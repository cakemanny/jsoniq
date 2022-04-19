use serde_json::Value;
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug)]
enum CompOp { EQ, LT }
#[derive(Debug)]
enum Ordering { ASC, DESC }

type Name = String;

#[derive(Debug)]
struct VarRef { ref_: Name }
impl VarRef {
    fn from(s: &str) -> VarRef {
        VarRef {
            ref_: String::from(s)
        }
    }
}

#[derive(Debug)]
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

fn eval_query(expr: Expr) -> Result<Box<dyn Iterator<Item = Value>>, String> {
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
            // TODO: use decomposition
            let (first_for_binding, first_for_expr) = &for_[0];

            Ok(Box::new(force_seq(first_for_expr, &BTreeMap::new())?.map(|tuple| {

                // Create binding for the `for`
                let mut bindings = BTreeMap::new();
                bindings.insert(first_for_binding.ref_.clone(), tuple);

                // Let bindings
                for let_binding in let_.iter() {
                    // I think we have to assume at this point that
                    // the the expressions have been checked
                    let e = eval_expr(let_binding.1, &bindings).unwrap_or(Value::Null);
                    bindings.insert(let_binding.0.ref_, e);
                }

                // TODO: where

                //
                let e = eval_expr(*return_, &bindings).unwrap_or(Value::Null);
                e
            }).collect::<Vec<Value>>().into_iter()))
        },
        _ => {
            Err("top level expression must be a FLWOR".to_owned())
        }
    }
}

fn force_seq(expr: Expr, bindings: &BTreeMap<Name, Value>) -> Result<Box<dyn Iterator<Item = Value>>, String> {
    match expr {
        Expr::For{ .. } => {
            eval_query(expr)
        },
        // 
        Expr::Array(values) => {
            // Should this in fact evaluate the values as they are pulled?
            // Or should that be saved for Sequence?

            //  loop through values and evaluate each
            let mut evaluated: Vec<Value> = vec![];
            for x in values {
                let e = eval_expr(x, bindings)?;
                evaluated.push(e);
            }
            Ok(Box::new(evaluated.into_iter()))
        },

        // Should atoms be sequences of length 1 or errors?
        Expr::Comp(..) => {
            let e = eval_expr(expr, bindings)?;
            Ok(Box::new(Some(e).into_iter()))
        },
        Expr::Literal(atom) => Ok(Box::new(Some(atom).into_iter())),
        Expr::VarRef(var_ref) => {
            // Not sure if this is true or not
            Err("VarRef: A variable cannot contain a sequence".to_owned())
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
fn eval_expr(expr: Expr, bindings: &BTreeMap<Name, Value>) -> Result<Value, String> {
    match expr {
        // This shouldn't be the case
        // it should just be the case that we consume the stream
        Expr::For{ .. } => Err("No FLWOR at this point".to_owned()),
        Expr::Comp(CompOp::LT, lhs, rhs) => {
            let l: Value = eval_expr(*lhs, bindings)?;
            let r: Value = eval_expr(*rhs, bindings)?;
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
            let _l: Value = eval_expr(*lhs, bindings)?;
            let _r: Value = eval_expr(*rhs, bindings)?;
            // I think we actually just need to derive Equal to get this for
            // free
            Err("TODO: EQ".to_owned())
        },
        Expr::Array(values) => {
            //  loop through values and evaluate each
            let mut evaluated: Vec<Value> = vec![];
            for x in values {
                let e = eval_expr(x, bindings)?;
                evaluated.push(e);
            }
            Ok(Value::Array(evaluated))
        },
        Expr::Literal(atom) => Ok(atom),
        Expr::VarRef(VarRef{ ref_ }) => {
            match bindings.get(&ref_) {
                Some(&value) => Ok(value.clone()),
                None => Ok(Value::Null)
            }
        }
    }
}


pub fn run_example() {

    let source = Expr::Array(vec![
        Expr::Literal(json!(1.0)),
        Expr::Literal(json!(2.0)),
        Expr::Literal(json!(3.0)),
        Expr::Literal(json!(4.0)),
        Expr::Literal(json!(5.0)),
    ]);

    let example_expr = Expr::For {
        for_: vec![(VarRef::from("x"), source)],
        let_: Vec::new(),
        where_: Box::new(Expr::Comp(
                CompOp::LT,
                Box::new(Expr::VarRef(VarRef::from("x"))),
                Box::new(Expr::Literal(json!(3.0)))
        )),
        order: Vec::new(),
        return_: Box::new(Expr::VarRef(VarRef::from("x"))),
    };

    match eval_query(example_expr) {
        Ok(result_set) => result_set.for_each(|expr| {
            println!("{:?}", expr);
        }),
        Err(some_msg) => println!("{}", some_msg)
    }
}
