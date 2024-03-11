use anyhow::anyhow;
use serde_json::json;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::rc::Rc;
use std::vec;

#[derive(Debug, Clone, Copy)]
enum CompOp {
    EQ,
    LT,
}
#[derive(Debug, Clone, Copy)]
enum Ordering {
    ASC,
    DESC,
}

type Name = String;

#[derive(Debug, Clone)]
struct VarRef {
    ref_: Name,
}
impl From<&str> for VarRef {
    fn from(s: &str) -> VarRef {
        VarRef {
            ref_: s.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    For {
        for_: Vec<(VarRef, Expr)>, // for $x in collection("captains")
        let_: Vec<(VarRef, Expr)>, // let $century := $x.century
        where_: Box<Expr>,         // where $x.name eq "Kathryn Janeway"
        order: Vec<(Expr, Ordering)>, // order by $x.name
        // TODO: group_by
        return_: Box<Expr>, // return $x
    },
    // FnCall(Option<Name>, Name, Vec<Box<Expr>>), //
    Comp(CompOp, Box<Expr>, Box<Expr>),
    ArrayUnbox(Box<Expr>),

    Sequence(Vec<Expr>),
    Array(Vec<Expr>),
    Literal(Value),
    VarRef(VarRef),
}

// I am thinking we'll have a few different implementations of
// Sequence. e.g. backed by a collection, backend by
// This probably could have been an alias IntoIterator ...
#[derive(Clone)]
enum Sequence {
    VecBackend(Vec<Value>),
}
impl Sequence {
    // This is not so great given that we might hit errors when
    // reading sequences... I think
    fn get_iter(self) -> Box<dyn Iterator<Item = Value>> {
        match self {
            Sequence::VecBackend(vs) => Box::new(vs.into_iter()),
        }
    }
}
impl IntoIterator for Sequence {
    type Item = Value;
    type IntoIter = Box<dyn Iterator<Item = Value>>;
    fn into_iter(self: Self) -> Self::IntoIter {
        self.get_iter()
    }
}


// We are not currently implementing TryInto<Value> as in many cases
// an empty sequence behaves differently from any value and thus
// it may be useful to preserve the distinction into the use site
#[derive(Clone)]
enum Data {
    // We use Box so that this can go into our bindings.
    //   we should probably add a condition that the IntoIter should
    //   implement Copy or Clone
    Sequence(Sequence),
    EmptySequence,
    Value(Value),
}

impl Data {
    /// force evaluates Sequences and turns them into Values or EmptySequences
    /// if they contain exactly one or 0 items
    fn force(self) -> Data {
        if let Data::Sequence(seq) = self {
            let mut iter = seq.get_iter();
            if let Some(v0) = iter.next() {
                if let Some(v1) = iter.next() {
                    let mut forced = vec![v0, v1];
                    forced.extend(iter);
                    Data::Sequence(Sequence::VecBackend(forced))
                } else {
                    Data::Value(v0)
                }
            } else {
                Data::EmptySequence
            }
        } else {
            self
        }
    }
}

// we have to implement Debug manually due to the Rc
impl Debug for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Data::Sequence(_rc) => {
                write!(f, "Sequence[..]")
            }
            Data::EmptySequence => {
                write!(f, "EmptySequence[..]")
            }
            Data::Value(v) => {
                write!(f, "Value[{}]", v)
            }
        }
    }
}

impl TryInto<bool> for Data {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<bool, Self::Error> {
        match self {
            Data::Sequence(seq) => {
                let mut iter = seq.get_iter();
                match (iter.next(), iter.next()) {
                    (None, None) => Ok(false),
                    (Some(v0), None) => Ok(json_to_bool(&v0)),
                    (Some(..), Some(..)) => {
                        Err(anyhow!("non-singleton sequence used as bool"))
                    }
                    (None, Some(..)) => panic!("broken iterator"),
                }
            }
            Data::EmptySequence => Ok(false),
            Data::Value(value) => Ok(json_to_bool(&value)),
        }
    }
}

fn json_to_bool(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => b.to_owned(),
        // This is broken for NaN or Infinity ... I think
        Value::Number(n) => n.as_f64().map(|n| n != 0.0).unwrap_or(false),
        Value::String(s) => !s.is_empty(),
        Value::Array(_) => true,
        Value::Object(_) => true,
    }
}

// BTreeMap feels a bit heavyweight, when we imagine that there will only
// be around a dozen variables probably
type Bindings = BTreeMap<String, Data>;

//
// Returns an iterator that gives the bindings produced by nesting all the
// for expressions
fn forexp_to_iter<'a>(
    for_: &'a [(VarRef, Expr)],
    bindings: Bindings,
) -> Box<dyn Iterator<Item = anyhow::Result<Bindings>> + 'a> {
    if for_.is_empty() {
        return Box::new(None.into_iter());
    }
    let (var_ref, exp) = &for_.first().unwrap();

    match force_seq(exp, &bindings) {
        Ok(it) => {
            let bind_var = move |v: Value| {
                // Create binding for the `for`
                let mut inner_bindings = bindings.clone();
                inner_bindings
                    .insert(var_ref.ref_.to_owned(), Data::Value(v.to_owned()));

                inner_bindings
            };

            if for_.len() == 1 {
                Box::new(it.map(bind_var).map(Ok))
            } else {
                let remaining = &for_[1..];
                Box::new(it.map(bind_var).flat_map(|inner_bindings| {
                    forexp_to_iter(remaining, inner_bindings)
                }))
            }
        }
        Err(err) => Box::new(Some(Err(err)).into_iter()),
    }
}

// In the next iteration of this eval_query, we will need to recursively
// chain together the fors.
// I think we may also want to have the notion of binding streams to
// variables...
//
// This ought to take a set of binding too, in case it's a nested query
//
// We ought to rewrite as multiple passes,
// one that does some sort of preparation of sources (e.g. collections)
fn eval_query(expr: &Expr) -> anyhow::Result<Box<dyn Iterator<Item = Value>>> {
    match expr {
        Expr::For {
            for_,
            let_,
            where_,
            order: _, // TODO: implement ordering
            return_,
        } => {
            // 1. evaluate the rhs of the for_, and convert to a sequence
            // 2. for each tuple in the sequence
            //    bind the value to the variable in a map
            // 2. evaluate the let_ with the map, for each tuple, producing a new map
            // 3. filter the tuples
            // 4. ignore the order clause for now
            // 5. evaluate the return_ expr and call consume on it

            // Start off basic
            if for_.is_empty() {
                anyhow::bail!("Need for for now");
            }

            let value_stream = forexp_to_iter(for_, BTreeMap::new())
                .map(|bindings_result| {
                    let mut bindings = bindings_result?;

                    // Let bindings
                    for let_binding in let_.iter() {
                        // I think we have to assume at this point that
                        // the the expressions have been checked
                        let data = eval_expr(&let_binding.1, &bindings)?;
                        bindings.insert(let_binding.0.ref_.to_owned(), data);
                    }
                    Ok(bindings)
                })
                .filter_map(|bindings_result| {
                    let bindings = match bindings_result {
                        Ok(bindings) => bindings,
                        Err(e) => return Some(Err(e)),
                    };
                    let get_cond_as_bool =
                        || Ok(eval_expr(&where_, &bindings)?.try_into()?);
                    match get_cond_as_bool() {
                        Err(e) => return Some(Err(e)),
                        Ok(false) => return None,
                        // we fall through so that evaluation of return_ is
                        // a bit separated from evaluation of where
                        Ok(true) => {}
                    }
                    Some(eval_expr(&return_, &bindings))
                })
                .flat_map(data_result_to_value_results)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter();

            Ok(Box::new(value_stream))
        }
        _ => Err(anyhow!("top level expression must be a FLWOR")),
    }
}

fn data_to_values(datum: Data) -> Box<dyn Iterator<Item = Value>> {
    match datum {
        Data::Value(value) => Box::new(Some(value).into_iter()),
        Data::Sequence(seq) => Box::new(seq.get_iter()),
        Data::EmptySequence => Box::new(None.into_iter()),
    }
}

fn data_result_to_value_results(
    data_result: anyhow::Result<Data>,
) -> Box<dyn Iterator<Item = anyhow::Result<Value>>> {
    match data_result {
        Ok(datum) => Box::new(data_to_values(datum).map(Ok)),
        Err(msg) => Box::new(Some(Err(msg)).into_iter()),
    }
}

fn force_seq(
    expr: &Expr,
    bindings: &Bindings,
) -> anyhow::Result<Box<dyn Iterator<Item = Value>>> {
    match expr {
        Expr::For { .. } => eval_query(expr),
        Expr::Sequence(..) => {
            let data = eval_expr(expr, bindings)?;
            match data {
                Data::Sequence(seq) => Ok(seq.get_iter()),
                _ => panic!("sequence did not evaluate to sequence"),
            }
        }
        Expr::Array(..) => {
            let data = eval_expr(expr, bindings)?;
            match data {
                // Array should always evaluate to array in the success case
                Data::Value(v) => Ok(Box::new(Some(v).into_iter())),
                _ => panic!("array did not evaluate to array"),
            }
        }

        // Atoms become sequences of length 1
        Expr::Comp(..) => {
            let data = eval_expr(expr, bindings)?;
            match data {
                Data::Value(v) => Ok(Box::new(Some(v).into_iter())),
                Data::EmptySequence => Ok(Box::new(None.into_iter())),
                Data::Sequence(..) => {
                    panic!("comparison produced nonempty sequence")
                }
            }
        }
        Expr::ArrayUnbox(..) => {
            // Since we know the values are in memory, I think it's ok
            // to cheat and use the conversion to vec
            let data = eval_expr(expr, bindings)?;
            Ok(Box::new(data_to_values(data)))
        }
        Expr::Literal(atom) => Ok(Box::new(Some(atom.clone()).into_iter())),
        Expr::VarRef(var_ref) => match bindings.get(&var_ref.ref_) {
            Some(data) => Ok(data_to_values(data.clone())),
            // TODO: we should come up with our own errors for certain
            // things
            None => Err(anyhow!("VarRef: {}", var_ref.ref_)),
        },
    }
}

// Should be some sort of checking function?
// i.e. / e.g that checks all the variable references are valid
// and that maybe some types are correct, if possible

// Should this return some sort of stream?
// Or take a pull and an error function
//
// Needs to take an environment, or a tuple or something
fn eval_expr(expr: &Expr, bindings: &Bindings) -> anyhow::Result<Data> {
    match expr {
        // This shouldn't be the case
        // it should just be the case that we consume the stream
        // TODO: actually nested FLWOR is supported
        Expr::For { .. } => Err(anyhow!("No FLWOR at this point")),
        Expr::Comp(CompOp::LT, lhs, rhs) => {
            // Or should we only evaluate rd if necessary?
            let ld: Data = eval_expr(&*lhs, bindings)?.force();
            let rd: Data = eval_expr(&*rhs, bindings)?.force();
            match (ld, rd) {
                (Data::Sequence(..), _) | (_, Data::Sequence(..)) => {
                    Err(anyhow!("comparison on multivalued sequences"))
                }
                (Data::Value(l), Data::Value(r)) => match (l, r) {
                    // In the future we may be able to guard on
                    // is_f64 and use some never type inference to
                    // avoid this ugliness
                    (Value::Number(nl), Value::Number(nr)) => {
                        let nl0 = nl.as_f64().ok_or(anyhow!("fml"))?;
                        let nr0 = nr.as_f64().ok_or(anyhow!("fmr"))?;
                        Ok(Data::Value(Value::Bool(nl0 < nr0)))
                    }
                    _ => Err(anyhow!(
                        "comparison can only be done between numbers"
                    )),
                },
                (Data::EmptySequence, _) | (_, Data::EmptySequence) => {
                    Ok(Data::EmptySequence)
                }
            }
        }
        Expr::Comp(CompOp::EQ, lhs, rhs) => {
            let _l: Data = eval_expr(&*lhs, bindings)?;
            let _r: Data = eval_expr(&*rhs, bindings)?;
            // I think we actually just need to derive Equal to get this for
            // free... except for some sequence cases.
            todo!("EQ")
        }
        Expr::ArrayUnbox(subexp) => {
            // Array unboxing turns an array into a sequence
            // an any other kind of value into the empty sequence
            let data = eval_expr(&subexp, bindings)?;

            match data {
                Data::Value(Value::Array(values)) => {
                    Ok(Data::Sequence(Sequence::VecBackend(values)))
                }
                _ => Ok(Data::EmptySequence),
            }
        }
        Expr::Sequence(exprs) => {
            // since a sequence could include multiple sources (in the future)
            // it would be nice to return some sort of lazy thing

            let evaluated: Vec<_> = exprs
                .iter()
                .map(|x| eval_expr(x, bindings))
                .flat_map(data_result_to_value_results)
                .collect::<Result<_, _>>()?;

            Ok(Data::Sequence(Sequence::VecBackend(evaluated)))
        }
        Expr::Array(exprs) => {
            //  loop through exprs and evaluate each
            let evaluated: Vec<Value> = exprs
                .iter()
                .map(|x| eval_expr(x, bindings))
                .flat_map(data_result_to_value_results)
                .collect::<Result<_, _>>()?;

            Ok(Data::Value(Value::Array(evaluated)))
        }
        Expr::Literal(atom) => Ok(Data::Value(atom.clone())),
        Expr::VarRef(VarRef { ref_ }) => match bindings.get(ref_) {
            Some(value) => Ok(value.clone()),
            None => Ok(Data::Value(Value::Null)),
        },
    }
}

pub fn run_example() {
    let source_x = Expr::Sequence(vec![
        Expr::Literal(json!(1.0)),
        Expr::Literal(json!(2.0)),
        Expr::Literal(json!(3.0)),
        Expr::Literal(json!(4.0)),
        Expr::Literal(json!(5.0)),
    ]);
    let source_y = Expr::Sequence(vec![
        Expr::Literal(json!(1.0)),
        Expr::Literal(json!(2.0)),
        Expr::Literal(json!(3.0)),
        Expr::Literal(json!(4.0)),
        Expr::Literal(json!(5.0)),
    ]);

    let example_expr = Expr::For {
        for_: vec![("x".into(), source_x), ("y".into(), source_y)],
        let_: Vec::new(),
        where_: Box::new(Expr::Comp(
            CompOp::LT,
            Box::new(Expr::VarRef("x".into())),
            Box::new(Expr::Literal(json!(3.0))),
        )),
        order: Vec::new(),
        return_: Box::new(Expr::Array(vec![
            Expr::VarRef("x".into()),
            Expr::VarRef("y".into()),
        ])),
    };

    match eval_query(&example_expr) {
        Ok(result_set) => result_set.for_each(|expr| {
            println!("{:?}", expr);
        }),
        Err(some_msg) => println!("{}", some_msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_two_for_expressions() {
        let source_x = Expr::Sequence(vec![
            Expr::Literal(json!(1.0)),
            Expr::Literal(json!(2.0)),
        ]);
        let source_y = Expr::Sequence(vec![
            Expr::Literal(json!(1.0)),
            Expr::Literal(json!(2.0)),
        ]);

        let fors_: Vec<(VarRef, Expr)> =
            vec![("x".into(), source_x), ("y".into(), source_y)];

        let it = forexp_to_iter(&fors_, BTreeMap::new());
        let res: Vec<(Option<Value>, Option<Value>)> = it
            .map(|bindings_result| {
                bindings_result.map(|bindings| {
                    let undata = |d| match d {
                        Data::Value(v) => v,
                        _ => panic!("expected value not sequence"),
                    };

                    let x: Option<Value> =
                        bindings.get("x").cloned().map(undata);
                    let y: Option<Value> =
                        bindings.get("y").cloned().map(undata);

                    (x, y)
                })
            })
            .collect::<Result<_, _>>()
            .unwrap();

        assert_eq!(
            res,
            vec![
                (Some(json!(1.0)), Some(json!(1.0))),
                (Some(json!(1.0)), Some(json!(2.0))),
                (Some(json!(2.0)), Some(json!(1.0))),
                (Some(json!(2.0)), Some(json!(2.0))),
            ]
        )
    }

    #[test]
    fn for_expansion_sub_iteration() {
        let source_x = Expr::Sequence(vec![
            Expr::Literal(json!([1.0, 2.0])),
            Expr::Literal(json!([3.0, 4.0])),
        ]);

        let fors_: Vec<(VarRef, Expr)> = vec![
            ("x".into(), source_x),
            (
                "y".into(),
                Expr::ArrayUnbox(Box::new(Expr::VarRef("x".into()))),
            ),
        ];

        let it = forexp_to_iter(&fors_, BTreeMap::new());
        let res: Vec<(Option<Value>, Option<Value>)> = it
            .map(|bindings_result| {
                bindings_result.map(|bindings| {
                    let undata = |d| match d {
                        Data::Value(v) => v,
                        _ => panic!("expected value not sequence"),
                    };
                    let x: Option<Value> =
                        bindings.get("x").cloned().map(undata);
                    let y: Option<Value> =
                        bindings.get("y").cloned().map(undata);

                    (x, y)
                })
            })
            .collect::<Result<_, _>>()
            .unwrap();

        assert_eq!(
            res,
            vec![
                (Some(json!([1.0, 2.0])), Some(json!(1.0))),
                (Some(json!([1.0, 2.0])), Some(json!(2.0))),
                (Some(json!([3.0, 4.0])), Some(json!(3.0))),
                (Some(json!([3.0, 4.0])), Some(json!(4.0))),
            ]
        )
    }
}
