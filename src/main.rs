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

// Or should we use Serde's json types?
#[derive(Debug)]
enum Atom {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

#[derive(Debug)]
enum Literal {
    Atom(Atom),
    Array(Vec<Literal>),
    Object(BTreeMap<String, Literal>)
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
    Atom(Atom),
    VarRef(VarRef),
}

fn eval_query(expr: Expr, consume: Box<dyn Fn(&Expr) -> ()>) {
    match expr {
        Expr::For{for_, let_, where_, order, return_} => {

            // 1. evaluate the rhs of the for_, and convert to a stream
            // 2. for each tuple in the stream
            //    bind the value to the variable in a map
            // 2. evaluate the let_ with the map, for each tuple, producing a new map
            // 3. filter the tuples
            // 4. ignore the order clause for now
            // 5. evaluate the return_ expr and call consume on it
        },
        _ => {
            println!("top level expression must be a FLWOR")
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
fn eval_expr(expr: Expr) -> Result<Literal, String> {
    match expr {
        Expr::For{ .. } => Err("No FLWOR at this point".to_owned()),
        Expr::Comp(CompOp::LT, lhs, rhs) => {
            let l: Literal = eval_expr(*lhs)?;
            let r: Literal = eval_expr(*rhs)?;
            match (l, r) {
                (Literal::Atom(Atom::Num(nl)), Literal::Atom(Atom::Num(nr))) => {
                    Ok(Literal::Atom(Atom::Bool( nl < nr )))
                }
                _ => Err("comparison can only be done between numbers".to_owned())
            }
        },
        Expr::Comp(CompOp::EQ, lhs, rhs) => {
            let _l: Literal = eval_expr(*lhs)?;
            let _r: Literal = eval_expr(*rhs)?;
            // I think we actually just need to derive Equal to get this for
            // free
            Err("TODO: EQ".to_owned())
        },
        Expr::Array(values) => {
            //  loop through values and evaluate each
            let mut evaluated: Vec<Literal> = vec![];
            for x in values {
                let e = eval_expr(x)?;
                evaluated.push(e);
            }
            Ok(Literal::Array(evaluated))
        },
        Expr::Atom(atom) => Ok(Literal::Atom(atom)),
        Expr::VarRef(..) => {
            // This function needs to take a mapping containing the
            // variables
            Err("TODO: var ref".to_owned())
        }
    }
}


fn main() {

    let source = Expr::Array(vec![
        Expr::Atom(Atom::Num(1.0)),
        Expr::Atom(Atom::Num(2.0)),
        Expr::Atom(Atom::Num(3.0)),
        Expr::Atom(Atom::Num(4.0)),
        Expr::Atom(Atom::Num(5.0)),
    ]);

    let example_expr = Expr::For {
        for_: vec![(VarRef::from("x"), source)],
        let_: Vec::new(),
        where_: Box::new(Expr::Comp(
                CompOp::LT,
                Box::new(Expr::VarRef(VarRef::from("x"))),
                Box::new(Expr::Atom(Atom::Num(3.0)))
        )),
        order: Vec::new(),
        return_: Box::new(Expr::VarRef(VarRef::from("x"))),
    };

    eval_query(example_expr, Box::new(|expr| {
        println!("{:?}", expr);
    }));
}
