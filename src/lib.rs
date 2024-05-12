use anyhow::{anyhow, bail};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::fmt::Debug;
use std::io::Write;
use std::sync::Mutex;
use std::vec;

pub mod ast;
pub mod parse;
use crate::ast::{CompOp, Expr, VarRef};

// I am thinking we'll have a few different implementations of
// Sequence. e.g. backed by a collection, backend by
// This probably could have been an alias IntoIterator ...
#[derive(Clone)]
enum Sequence {
    VecBackend(Vec<Value>),
    // Want to do sth like this:
    // MappedVec(Vec<Value>, Box<dyn FnOnce(Value) -> Value>),
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
    // Later me: sequences may want to hold a reference to
    //   1. the syntax tree
    //   2. a possibly cached collection of "nodes" / file handle / ...
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Data::Sequence(_rc) => write!(f, "Sequence[..]"),
            Data::EmptySequence => write!(f, "EmptySequence[..]"),
            Data::Value(v) => write!(f, "Value[{}]", v),
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

// Do I need to learn about reference generics?
fn cast_json_to_string(value: Value) -> anyhow::Result<String> {
    match value {
        Value::String(s) => Ok(s),
        Value::Number(n) => Ok(format!("{}", n)),
        Value::Null => Ok("null".to_owned()),
        Value::Bool(b) => Ok(format!("{}", b)),
        Value::Array(..) => Err(anyhow!("cannot cast array to string")),
        Value::Object(..) => Err(anyhow!("cannot cast object to string")),
    }
}

// Some sort of execution context
// Why? we need a place to store the Available Documents and
// Available Node Collections. That is, any call to collection with the same
// parameter should give the same sequence - which implies that we have cached
// them. (or we're using a database that gives consistent read... unlikely).

// Since declarations are completely separated from the main query, we
// can put the function library and the modules in the execution context.
struct StaticContext {
    functions: FnTable,
}

// This will contain the available documents and node collections
// In the future maybe this wants to contain some sort of interface that
// talks to the correct kind of plugins
struct DynamicContext {
    stat_ctx: StaticContext,
    /// The key is a file path.
    /// In the future it shall also be possible to give a uri.
    json_file_contents: Mutex<HashMap<String, Vec<Value>>>,
}

// BTreeMap feels a bit heavyweight, when we imagine that there will only
// be around a dozen variables probably
// TODO: it's probably possible to have the key be a &str since
// the names are coming from the program text.
type Bindings = BTreeMap<String, Data>;

//
// Returns an iterator that gives the bindings produced by nesting all the
// for expressions
fn forexp_to_iter<'a, 'ctx>(
    for_: &'a [(VarRef, Expr)],
    bindings: Bindings,
    ctx: &'ctx DynamicContext,
) -> Box<dyn Iterator<Item = anyhow::Result<Bindings>> + 'a>
where
    'ctx: 'a,
{
    if for_.is_empty() {
        return Box::new(None.into_iter());
    }
    let (var_ref, exp) = &for_.first().unwrap();

    match force_seq(exp, &bindings, ctx) {
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
                    forexp_to_iter(remaining, inner_bindings, ctx)
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
// We ought to rewrite as multiple passes,
// one that does some sort of preparation of sources (e.g. collections)
fn eval_query(
    expr: &Expr,
    bindings: &Bindings,
    ctx: &DynamicContext,
) -> anyhow::Result<Box<dyn Iterator<Item = Value>>> {
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

            let value_stream = forexp_to_iter(for_, bindings.clone(), ctx)
                .map(|bindings_result| {
                    let mut bindings = bindings_result?;

                    // Let bindings
                    for let_binding in let_.iter() {
                        // I think we have to assume at this point that
                        // the the expressions have been checked
                        let data = eval_expr(&let_binding.1, &bindings, ctx)?;
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
                        || Ok(eval_expr(&where_, &bindings, ctx)?.try_into()?);
                    match get_cond_as_bool() {
                        Err(e) => return Some(Err(e)),
                        Ok(false) => return None,
                        // we fall through so that evaluation of return_ is
                        // a bit separated from evaluation of where
                        Ok(true) => {}
                    }
                    Some(eval_expr(&return_, &bindings, ctx))
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
    ctx: &DynamicContext,
) -> anyhow::Result<Box<dyn Iterator<Item = Value>>> {
    match expr {
        Expr::For { .. } => eval_query(expr, bindings, ctx),
        Expr::FnCall(..) | Expr::ObjectLookup { .. } => {
            let data = eval_expr(expr, bindings, ctx)?;
            Ok(data_to_values(data))
        }
        Expr::Sequence(..) => {
            let data = eval_expr(expr, bindings, ctx)?;
            match data {
                Data::Sequence(seq) => Ok(seq.get_iter()),
                _ => panic!("sequence did not evaluate to sequence"),
            }
        }
        Expr::Array(..) => {
            let data = eval_expr(expr, bindings, ctx)?;
            match data {
                // Array should always evaluate to array in the success case
                Data::Value(v) => Ok(Box::new(Some(v).into_iter())),
                _ => panic!("array did not evaluate to array"),
            }
        }
        // Atoms become sequences of length 1
        Expr::Comp(..) => {
            let data = eval_expr(expr, bindings, ctx)?;
            match data {
                Data::Value(v) => Ok(Box::new(Some(v).into_iter())),
                Data::EmptySequence => Ok(Box::new(None.into_iter())),
                Data::Sequence(..) => {
                    panic!("comparison produced nonempty sequence")
                }
            }
        }
        Expr::ArrayUnbox(..) => {
            let data = eval_expr(expr, bindings, ctx)?;
            Ok(data_to_values(data))
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
fn eval_expr(
    expr: &Expr,
    bindings: &Bindings,
    ctx: &DynamicContext,
) -> anyhow::Result<Data> {
    match expr {
        // This shouldn't be the case
        // it should just be the case that we consume the stream
        // TODO: actually nested FLWOR is supported
        Expr::For { .. } => Err(anyhow!("No FLWOR at this point")),
        Expr::FnCall((_nsopt, name), args) => {
            // We should define a static pass that checks this before
            // execution.
            let Some(jfn) = ctx.stat_ctx.functions.get(name.as_str()) else {
                bail!("no such function: {}", name);
            };
            if args.len() != jfn.num_args() {
                bail!(
                    "{} expects {} arguments but {} were given",
                    name,
                    jfn.num_args(),
                    args.len()
                )
            };

            // we could reverse it?
            let mut evaluated_args: Vec<_> = args
                .iter()
                .map(|x| eval_expr(x, bindings, ctx))
                .collect::<Result<_, _>>()?;

            // Maybe we can learn to write macros an we'll end up writing
            // this as macro.
            match jfn {
                JsoniqFn::Jfn1(f) => f(ctx, evaluated_args.remove(0)),
                JsoniqFn::Jfn2(f) => {
                    let arg1 = evaluated_args.remove(1);
                    let arg0 = evaluated_args.remove(0);
                    f(ctx, arg0, arg1)
                }
            }
        }
        Expr::Comp(CompOp::LT, lhs, rhs) => {
            // Or should we only evaluate rd if necessary?
            let ld: Data = eval_expr(&*lhs, bindings, ctx)?.force();
            let rd: Data = eval_expr(&*rhs, bindings, ctx)?.force();
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
            let _l: Data = eval_expr(&*lhs, bindings, ctx)?;
            let _r: Data = eval_expr(&*rhs, bindings, ctx)?;
            // I think we actually just need to derive Equal to get this for
            // free... except for some sequence cases.
            todo!("EQ")
        }
        Expr::ArrayUnbox(subexp) => {
            // Array unboxing turns an array into a sequence
            // and any other kind of value into the empty sequence
            let data = eval_expr(&subexp, bindings, ctx)?;

            match data {
                Data::Value(Value::Array(values)) => {
                    Ok(Data::Sequence(Sequence::VecBackend(values)))
                }
                _ => Ok(Data::EmptySequence),
            }
        }
        Expr::ObjectLookup {
            obj: obj_exp,
            lookup: lookup_exp,
        } => {
            let lookup_data = eval_expr(lookup_exp, bindings, ctx)?.force();
            let lookup = match lookup_data {
                Data::Sequence(..) | Data::EmptySequence => {
                    return Err(anyhow!(
                        "lookup expression cannot be a sequence"
                    ))
                }
                Data::Value(v) => cast_json_to_string(v)?,
            };

            let obj = eval_expr(&obj_exp, bindings, ctx)?;

            let perform_lookup = |value| match value {
                Value::Object(m) => {
                    let d = m
                        .get(lookup.as_str())
                        .map_or(Data::EmptySequence, |v| {
                            Data::Value(v.to_owned())
                        });
                    d
                }
                _ => Data::EmptySequence,
            };

            match obj {
                Data::Sequence(xs) => {
                    // and here a lazy sequence would be ideal
                    let mapped = xs
                        .get_iter()
                        .map(perform_lookup)
                        .flat_map(data_to_values)
                        .collect::<Vec<_>>();
                    Ok(Data::Sequence(Sequence::VecBackend(mapped)))
                }
                Data::EmptySequence => Ok(obj),
                Data::Value(value) => Ok(perform_lookup(value)),
            }
        }
        Expr::Sequence(exprs) => {
            // since a sequence could include multiple sources (in the future)
            // it would be nice to return some sort of lazy thing

            let evaluated: Vec<_> = exprs
                .iter()
                .map(|x| eval_expr(x, bindings, ctx))
                .flat_map(data_result_to_value_results)
                .collect::<Result<_, _>>()?;

            Ok(Data::Sequence(Sequence::VecBackend(evaluated)))
        }
        Expr::Array(exprs) => {
            //  loop through exprs and evaluate each
            let evaluated: Vec<Value> = exprs
                .iter()
                .map(|x| eval_expr(x, bindings, ctx))
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

struct Scope<'a> {
    parent_scope: Option<&'a Scope<'a>>,
    names: BTreeSet<String>,
    // todo: type ?
}
impl Scope<'_> {
    fn new() -> Scope<'static> {
        Scope {
            parent_scope: None,
            names: BTreeSet::new(),
        }
    }
}
impl Scope<'_> {
    fn with_parent<'a>(parent: &'a Scope) -> Scope<'a> {
        Scope {
            parent_scope: Some(parent),
            names: BTreeSet::new(),
        }
    }
}

#[derive(Debug)]
enum ValidationError {
    NoSuchFunction(String),
    IncorrectNumArgs {
        func: String,
        expected: usize,
        given: usize,
    },
    NoSuchVariable(String),
    Multiple {
        errs: Vec<ValidationError>,
    },
}
impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchFunction(func) => {
                write!(f, "no such function: {}", func)
            }
            Self::IncorrectNumArgs {
                func,
                expected,
                given,
            } => write!(
                f,
                "function {} expects {} arguments but {} were given",
                func, expected, given
            ),
            Self::NoSuchVariable(name) => {
                write!(f, "no variable ${} in scope", name)
            }
            Self::Multiple { errs } => {
                for err in errs.iter() {
                    write!(f, "{}\n", err)?
                }
                Ok(())
            }
        }
    }
}
impl std::error::Error for ValidationError {}

/// combines multiple errors together
macro_rules! check_multi {
    ($fst:expr, $snd:expr) => {
        match ($fst, $snd) {
            (Err(lhs), Err(rhs)) => {
                return Err(ValidationError::Multiple {
                    errs: vec![lhs, rhs],
                })
            }
            (Err(lhs), _) => return Err(lhs),
            (_, Err(rhs)) => return Err(rhs),
            (Ok(l), Ok(r)) => (l, r),
        }
    };
}

// one idea could be to return some sort of slightly transformed expression
// tree. e.g. a typed one or deduce contraints etc like max number of
// variables / ...
fn check_expr(
    expr: parse::Expr,
    scope: &Scope,
    ctx: &StaticContext,
) -> Result<Expr, ValidationError> {
    type E = ValidationError;

    match expr {
        parse::Expr::For {
            for_,
            let_,
            where_,
            order,
            return_,
        } => {
            // We work a bit non-functionaly in the sense that we are
            // mutating the scope as we work through the expression in
            // evaluation order.
            let mut new_scope = Scope::with_parent(scope);

            let checked_for = for_.into_iter().try_fold(
                Vec::new(),
                |mut acc, (var, expr)| {
                    let checked = check_expr(expr, &new_scope, ctx)?;
                    // When we do typing, this is where the difference between for
                    // and let below would show themselves. As the type of the
                    // bound variable in a for is the type of the sequence elements
                    // when the expression is a sequence...

                    new_scope.names.insert(var.ref_.clone());
                    acc.push((var.into(), checked));
                    Ok::<_, E>(acc)
                },
            )?;
            let checked_let = let_.into_iter().try_fold(
                Vec::new(),
                |mut acc, (var, expr)| {
                    let checked = check_expr(expr, &new_scope, ctx)?;
                    new_scope.names.insert(var.ref_.clone());
                    acc.push((var.into(), checked));
                    Ok::<_, E>(acc)
                },
            )?;
            let checked_where = Box::new(check_expr(*where_, &new_scope, ctx)?);
            // since order doesn't introduce any bindings it would definitely
            // be possible to check each part of the order clause individually
            // when we come to that.
            let checked_order = order
                .into_iter()
                .map(|(term, ord)| {
                    let checked_term = check_expr(term, &new_scope, ctx)?;
                    Ok::<_, E>((checked_term, ord.into()))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let checked_return =
                Box::new(check_expr(*return_, &new_scope, ctx)?);

            Ok(Expr::For {
                for_: checked_for,
                let_: checked_let,
                where_: checked_where,
                order: checked_order,
                return_: checked_return,
            })
        }
        parse::Expr::FnCall((ns, name), args) => {
            let Some(jfn) = ctx.functions.get(name.as_str()) else {
                // TODO: consider doing a fuzzy search?
                return Err(ValidationError::NoSuchFunction(name));
            };
            if args.len() != jfn.num_args() {
                return Err(ValidationError::IncorrectNumArgs {
                    func: name,
                    expected: jfn.num_args(),
                    given: args.len(),
                });
            }
            // TODO: consider having arg type specs where appropriate?

            let checked_args: Vec<_> = args
                .into_iter()
                .map(|arg| check_expr(arg, scope, ctx))
                .collect::<Result<_, _>>()?;

            Ok(Expr::FnCall((ns, name), checked_args))
        }
        parse::Expr::Comp(op, lhs, rhs) => {
            // TODO: if we can know the type and know that they are not
            // compatible then we can already return an error
            let (l, r) = check_multi!(
                check_expr(*lhs, scope, ctx),
                check_expr(*rhs, scope, ctx)
            );
            Ok(Expr::Comp(op.into(), Box::new(l), Box::new(r)))
        }
        parse::Expr::ArrayUnbox(expr) => {
            // Array unbox just returns an empty sequence when the expression
            // type doesn't match. So we do nothing special here.
            let expr = check_expr(*expr, scope, ctx)?;
            Ok(Expr::ArrayUnbox(Box::new(expr)))
        }
        parse::Expr::ObjectLookup { obj, lookup } => {
            let obj = check_expr(*obj, scope, ctx);
            let lookup = check_expr(*lookup, scope, ctx);
            let (obj, lookup) = check_multi!(obj, lookup);
            Ok(Expr::ObjectLookup {
                obj: Box::new(obj),
                lookup: Box::new(lookup),
            })
        }
        parse::Expr::Sequence(exprs) => {
            // TODO: try to write this in a 1. more functional 2. more generic
            // way
            let mut errs = Vec::new();
            let mut checked = Vec::new();
            for expr in exprs.into_iter() {
                match check_expr(expr, scope, ctx) {
                    Err(e) => errs.push(e),
                    Ok(expr) => checked.push(expr),
                }
            }
            match &*errs {
                [] => Ok(Expr::Sequence(checked)),
                [_one_err] => Err(errs.pop().unwrap()),
                _ => Err(E::Multiple { errs }),
            }
        }
        parse::Expr::Array(exprs) => {
            // TODO: factor this with Sequence
            let mut errs = Vec::new();
            let mut checked = Vec::new();
            for expr in exprs.into_iter() {
                match check_expr(expr, scope, ctx) {
                    Err(e) => errs.push(e),
                    Ok(expr) => checked.push(expr),
                }
            }
            match &*errs {
                [] => Ok(Expr::Array(checked)),
                [_one_err] => Err(errs.pop().unwrap()),
                _ => Err(E::Multiple { errs }),
            }
        }
        parse::Expr::Literal(v) => Ok(Expr::Literal(v)),
        parse::Expr::VarRef(var_ref) => {
            let mut maybe_scope = Some(scope);
            while maybe_scope.is_some() {
                let s = maybe_scope.unwrap();
                if s.names.contains(var_ref.ref_.as_str()) {
                    return Ok(Expr::VarRef(var_ref.into()));
                }
                maybe_scope = s.parent_scope;
            }
            Err(E::NoSuchVariable(var_ref.ref_.to_string()))
        }
    }
}

enum JsoniqFn {
    Jfn1(fn(&DynamicContext, Data) -> anyhow::Result<Data>),
    Jfn2(fn(&DynamicContext, Data, Data) -> anyhow::Result<Data>),
}

impl JsoniqFn {
    fn num_args(self: &Self) -> usize {
        match self {
            JsoniqFn::Jfn1(..) => 1,
            JsoniqFn::Jfn2(..) => 2,
        }
    }
}

/// The builtin function library
mod jfn {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    use anyhow::Context;

    use super::*;

    /// Casts it's arguments to strings and concatenates them.
    ///
    /// # Examples
    ///
    ///   fn:concat('Ciao!',()) -> "Ciao!"
    ///   fn:concat('un', 'grateful') -> "ungrateful"
    pub fn concat(
        _dyn_ctx: &DynamicContext,
        a1: Data,
        a2: Data,
    ) -> anyhow::Result<Data> {
        let to_str = |d: Data| match d.force() {
            Data::Sequence(..) => {
                Err(anyhow!("concat on multivalued sequence"))
            }
            Data::Value(v) => cast_json_to_string(v),
            Data::EmptySequence => Ok("".to_owned()),
        };
        let s1 = to_str(a1)?;
        let s2 = to_str(a2)?;
        Ok(Data::Value(Value::String(s1 + &s2)))
    }

    // Maybe we can end up writing some macros that validate / convert
    // the arguments...

    // json-file("captains.json")
    pub fn json_file(ctx: &DynamicContext, uri: Data) -> anyhow::Result<Data> {
        let to_str = |d: Data| match d.force() {
            Data::Sequence(..) => Err(anyhow!(
                "json-file with multivalued sequence, wanted string"
            )),
            Data::Value(v) => cast_json_to_string(v),
            Data::EmptySequence => Err(anyhow!(
                "json-file called with with empty sequence, wanted string"
            )),
        };
        let file_path = to_str(uri)?;
        let fp_str = file_path.as_str();

        let to_vec_seq =
            |values| Ok(Data::Sequence(Sequence::VecBackend(values)));

        let mut json_file_contents = ctx.json_file_contents.lock().unwrap();

        let Some(values) = json_file_contents.get(file_path.as_str()) else {
            let file = File::open(fp_str).with_context(|| {
                format!("Failed to read json from {}", fp_str)
            })?;
            let reader = BufReader::new(file);
            let mut values = Vec::new();
            for line in reader.lines() {
                let v: Value = serde_json::from_str(line?.as_str())?;
                values.push(v);
            }
            json_file_contents.insert(fp_str.to_owned(), values.clone());
            return to_vec_seq(values);
        };

        // TODO: we should have a backend that just refers to the values
        // in the map... instead of cloning...
        to_vec_seq(values.clone())
    }
}

type FnTable = HashMap<&'static str, JsoniqFn>;

fn base_fn_library() -> FnTable {
    HashMap::from([
        ("concat", JsoniqFn::Jfn2(jfn::concat)),
        ("json-file", JsoniqFn::Jfn1(jfn::json_file)),
    ])
}

pub fn run_program<W: Write>(program: &str, mut out: W) -> anyhow::Result<()> {
    let (_, expr) = parse::parse_main_module(program)
        .map_err(|e| anyhow::Error::new(e.to_owned()))?;

    let static_ctx = StaticContext {
        functions: base_fn_library(),
    };

    // There could now be a type checking phase that references static
    // context.

    let scope_chain = Scope::new();
    let expr = check_expr(expr, &scope_chain, &static_ctx)?;

    let ctx = DynamicContext {
        stat_ctx: static_ctx,
        json_file_contents: Mutex::new(HashMap::new()),
    };

    let bindings = Bindings::new();
    eval_query(&expr, &bindings, &ctx)?
        .try_for_each(|value| writeln!(out, "{value}"))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn combine_two_for_expressions() {
        let static_ctx = StaticContext {
            functions: base_fn_library(),
        };
        let ctx = DynamicContext {
            stat_ctx: static_ctx,
            json_file_contents: HashMap::new().into(),
        };
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

        let it = forexp_to_iter(&fors_, BTreeMap::new(), &ctx);
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
        let static_ctx = StaticContext {
            functions: base_fn_library(),
        };
        let ctx = DynamicContext {
            stat_ctx: static_ctx,
            json_file_contents: HashMap::new().into(),
        };
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

        let it = forexp_to_iter(&fors_, BTreeMap::new(), &ctx);
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
