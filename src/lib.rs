use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;
use std::vec;

#[derive(Debug, Clone, Copy)]
enum CompOp { EQ, LT }
#[derive(Debug, Clone, Copy)]
enum Ordering { ASC, DESC }

type Name = String;

#[derive(Debug, Clone)]
struct VarRef { ref_: Name }
impl From<&str> for VarRef {
    fn from(s: &str) -> VarRef {
        VarRef {
            ref_: s.to_owned()
        }
    }
}

#[derive(Debug, Clone)]
enum Expr {
    For{
        for_:  Vec<(VarRef, Expr)>, // for $x in collection("captains")
        let_: Vec<(VarRef, Expr)>, // let $century := $x.century
        where_: Box<Expr>, // where $x.name eq "Kathryn Janeway"
        order: Vec<(Expr, Ordering)>, // order by $x.name
        // TODO: group_by
        return_: Box<Expr>,// return $x
    },
    // FnCall(Option<Name>, Name, Vec<Box<Expr>>), //
    Comp(CompOp, Box<Expr>, Box<Expr>),
    Array(Vec<Expr>),
    Literal(Value),
    VarRef(VarRef),
}

// I am thinking we'll have a few different implementations of
// Sequence. e.g. backed by a collection, backend by
// This probably could have been an alias IntoIterator ...
trait Sequence: IntoIterator<Item=Value, IntoIter = dyn Iterator<Item = Value>> {
}

struct VecSequence {
    v: Vec<Value>
}



enum Data {
    // We use Box so that this can go into our bindings.
    //   we should probably add a condition that the IntoIter should
    //   implement Copy or Clone
    Sequence(Box<dyn Sequence>),
    Literal(Value),
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
type Bindings = BTreeMap<String, Value>;

//
// Returns an iterator that gives the bindings produced by nesting all the
// for expressions
fn forexp_to_iter(
    for_: &[(VarRef, Expr)],
    bindings: &Bindings,
) -> Box<dyn Iterator<Item = Result<Bindings, String>>> {

    if for_.is_empty() {
        return Box::new(vec![].into_iter());
    }

    if for_.len() == 1 {
        let (first_for_binding, first_for_expr) = &for_.first().unwrap();

        match force_seq(first_for_expr, bindings) {
            Ok(it) => {
                let res_stream = it.map(|v| {

                    // Create binding for the `for`
                    let mut inner_bindings = bindings.clone();
                    inner_bindings.insert(first_for_binding.ref_.to_owned(), v.to_owned());

                    Ok(inner_bindings)
                });
                let res_vec = res_stream.collect::<Vec<Result<_,String>>>();

                Box::new(res_vec.into_iter())
            },
            Err(err) => {
                Box::new(vec![Err(err)].into_iter())
            },
        }
    } else {
        let remaining = &for_[1..];

        let (first_for_binding, first_for_expr) = &for_.first().unwrap();

        match force_seq(first_for_expr, bindings) {
            Ok(it) => {
                let res_vec =
                    it.flat_map(|v| {

                        let mut inner_bindings = bindings.clone();
                        inner_bindings.insert(first_for_binding.ref_.to_owned(), v.to_owned());

                        forexp_to_iter(&remaining, &inner_bindings).into_iter()
                    }).collect::<Vec<Result<_,_>>>();

                Box::new(res_vec.into_iter())
            },
            Err(err) => {
                Box::new(vec![Err(err)].into_iter())
            },
        }

    }
}


// In the next iteration of this eval_query, we will need to recursively
// chain together the fors.
// I think we may also want to have the notion of binding streams to
// variables...
//
// This ought to take a set of binding too, in case it's a nested query
fn eval_query(expr: &Expr) -> Result<Box<dyn Iterator<Item = Value>>, String> {
    match expr {
        Expr::For{for_, let_, where_, order, return_} => {

            // 1. evaluate the rhs of the for_, and convert to a sequence
            // 2. for each tuple in the sequence
            //    bind the value to the variable in a map
            // 2. evaluate the let_ with the map, for each tuple, producing a new map
            // 3. filter the tuples
            // 4. ignore the order clause for now
            // 5. evaluate the return_ expr and call consume on it

            // Start off basic
            if for_.is_empty() {
                return Err("Need for for now".to_owned());
            }

            let res_stream = forexp_to_iter(for_, &BTreeMap::new())
                .filter_map(|bindings_result| {
                    if bindings_result.is_err() {
                        let err = bindings_result.unwrap_err();
                        return Some(Err(err));
                    }
                    let mut bindings = bindings_result.unwrap();

                    // Let bindings
                    for let_binding in let_.iter() {
                        // I think we have to assume at this point that
                        // the the expressions have been checked
                        let e_or_err = eval_expr(&let_binding.1, &bindings);
                        if e_or_err.is_err() {
                            let err = e_or_err.unwrap_err();
                            return Some(Err(err))
                        }
                        let e = e_or_err.unwrap();
                        bindings.insert(let_binding.0.ref_.to_owned(), e);
                    }

                    let where_result = eval_expr(&where_, &bindings);
                    if where_result.is_err() {
                        let err = where_result.unwrap_err();
                        return Some(Err(err))
                    }
                    let cond_as_bool: bool = json_to_bool(&where_result.unwrap());
                    if !cond_as_bool {
                        return None;
                    }

                    Some(eval_expr(&return_, &bindings))
                }).collect::<Result<Vec<_>,_>>()?.into_iter();

            Ok(Box::new(res_stream))
        },
        _ => {
            Err("top level expression must be a FLWOR".to_owned())
        }
    }
}

fn force_seq(expr: &Expr, bindings: &Bindings) -> Result<Box<dyn Iterator<Item = Value>>, String> {
    match expr {
        Expr::For{ .. } => {
            eval_query(expr)
        },
        // we should not do this...!
        // instead we need a sequence type
        // fixme: Arrays should be emitted as a single value
        Expr::Array(values) => {
            // Should this in fact evaluate the values as they are pulled?
            // Or should that be saved for Sequence?

            //  loop through values and evaluate each
            let evaluated: Vec<_> = values.iter()
                .map(|x| eval_expr(x, bindings))
                .collect::<Result<_,_>>()?;

            Ok(Box::new(evaluated.into_iter()))
        },

        // Should atoms be sequences of length 1 or errors?
        Expr::Comp(..) => {
            let e = eval_expr(expr, bindings)?;
            Ok(Box::new(Some(e).into_iter()))
        },
        Expr::Literal(atom) => Ok(Box::new(Some(atom.clone()).into_iter())),
        Expr::VarRef(var_ref) => {

            match bindings.get(&var_ref.ref_) {
                Some(value) => {
                    match value {
                        Value::Array(v) => {
                            let values = v.clone();
                            Ok(Box::new(values.into_iter()))
                        }
                        _ => Err(format!("${} not a sequence", var_ref.ref_).to_owned())
                    }
                    // FIXME: switch binding type to use Data as value
                    // return force_seq(value, bindings)
                },
                None => Err(format!("VarRef: {}", var_ref.ref_).to_owned()),
            }
        }
    }
}

// Should be some sort of checking function?
// i.e. / e.g that checks all the variable references are valid
// and that maybe some types are correct, if possible

// Should this return some sort of stream?
// Or take a pull and an error function
//
// Needs to take an environment, or a tuple or something
fn eval_expr(expr: &Expr, bindings: &Bindings) -> Result<Value, String> {
    match expr {
        // This shouldn't be the case
        // it should just be the case that we consume the stream
        Expr::For{ .. } => Err("No FLWOR at this point".to_owned()),
        Expr::Comp(CompOp::LT, lhs, rhs) => {
            let l: Value = eval_expr(&*lhs, bindings)?;
            let r: Value = eval_expr(&*rhs, bindings)?;
            match (l, r) {
                (Value::Number(nl), Value::Number(nr)) => {
                    let nl0 = nl.as_f64().ok_or("fml")?;
                    let nr0 = nr.as_f64().ok_or("fmr")?;
                    Ok(Value::Bool( nl0 < nr0 ))
                }
                _ => Err("comparison can only be done between numbers".to_owned())
            }
        },
        Expr::Comp(CompOp::EQ, lhs, rhs) => {
            let _l: Value = eval_expr(&*lhs, bindings)?;
            let _r: Value = eval_expr(&*rhs, bindings)?;
            // I think we actually just need to derive Equal to get this for
            // free
            Err("TODO: EQ".to_owned())
        },
        Expr::Array(values) => {
            //  loop through values and evaluate each
            let evaluated: Vec<Value> = values.iter()
                .map(|x| eval_expr(x, bindings))
                .collect::<Result<_,_>>()?;

            Ok(Value::Array(evaluated))
        },
        Expr::Literal(atom) => Ok(atom.clone()),
        Expr::VarRef(VarRef{ ref_ }) => {
            match bindings.get(ref_) {
                Some(value) => Ok(value.clone()),
                None => Ok(Value::Null)
            }
        }
    }
}


pub fn run_example() {

    let source_x = Expr::Array(vec![
        Expr::Literal(json!(1.0)),
        Expr::Literal(json!(2.0)),
        Expr::Literal(json!(3.0)),
        Expr::Literal(json!(4.0)),
        Expr::Literal(json!(5.0)),
    ]);
    let source_y = Expr::Array(vec![
        Expr::Literal(json!(1.0)),
        Expr::Literal(json!(2.0)),
        Expr::Literal(json!(3.0)),
        Expr::Literal(json!(4.0)),
        Expr::Literal(json!(5.0)),
    ]);

    let example_expr = Expr::For {
        for_: vec![
            ("x".into(), source_x),
            ("y".into(), source_y),
        ],
        let_: Vec::new(),
        where_: Box::new(Expr::Comp(
                CompOp::LT,
                Box::new(Expr::VarRef("x".into())),
                Box::new(Expr::Literal(json!(3.0))),
        )),
        order: Vec::new(),
        return_: Box::new(
            Expr::Array(vec![
                        Expr::VarRef("x".into()),
                        Expr::VarRef("y".into()),
            ])
        ),
    };

    match eval_query(&example_expr) {
        Ok(result_set) => result_set.for_each(|expr| {
            println!("{:?}", expr);
        }),
        Err(some_msg) => println!("{}", some_msg)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_two_for_expressions() {

        let source_x = Expr::Array(vec![
            Expr::Literal(json!(1.0)),
            Expr::Literal(json!(2.0)),
        ]);
        let source_y = Expr::Array(vec![
            Expr::Literal(json!(1.0)),
            Expr::Literal(json!(2.0)),
        ]);

        let fors_: Vec<(VarRef, Expr)> = vec![
            ("x".into(), source_x),
            ("y".into(), source_y),
        ];

        let it = forexp_to_iter(&fors_, &BTreeMap::new());
        let res: Vec<(Option<Value>, Option<Value>)> = it.map(|bindings_result| {
            bindings_result.map(|bindings|{
                let x: Option<Value> = bindings.get("x").cloned();
                let y: Option<Value> = bindings.get("y").cloned();

                (x, y)
            })
        }).collect::<Result<_, _>>().unwrap();

        assert_eq!(res, vec![
                   (Some(json!(1.0)), Some(json!(1.0))),
                   (Some(json!(1.0)), Some(json!(2.0))),
                   (Some(json!(2.0)), Some(json!(1.0))),
                   (Some(json!(2.0)), Some(json!(2.0))),
        ])
    }

    #[test]
    fn for_expansion_sub_iteration() {

        let source_x =
            Expr::Array(vec![
                        Expr::Array(vec![
                                    Expr::Literal(json!(1.0)),
                                    Expr::Literal(json!(2.0)),
                        ]),
                        Expr::Array(vec![
                                    Expr::Literal(json!(3.0)),
                                    Expr::Literal(json!(4.0)),
                        ]),
            ]);

        let fors_: Vec<(VarRef, Expr)> = vec![
            ("x".into(), source_x),
            ("y".into(), Expr::VarRef("x".into())),
        ];

        let it = forexp_to_iter(&fors_, &BTreeMap::new());
        let res: Vec<(Option<Value>, Option<Value>)> = it.map(|bindings_result| {
            bindings_result.map(|bindings|{
                let x: Option<Value> = bindings.get("x").cloned();
                let y: Option<Value> = bindings.get("y").cloned();

                (x, y)
            })
        }).collect::<Result<_, _>>().unwrap();

        assert_eq!(res, vec![
                   (Some(json!([1.0, 2.0])), Some(json!(1.0))),
                   (Some(json!([1.0, 2.0])), Some(json!(2.0))),
                   (Some(json!([3.0, 4.0])), Some(json!(3.0))),
                   (Some(json!([3.0, 4.0])), Some(json!(4.0))),
        ])
    }
}
