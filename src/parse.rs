//
// Reference Grammar: https://www.jsoniq.org/grammars/jsoniq.xhtml
//
use std::vec;

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, multispace0, multispace1, satisfy},
    combinator::{all_consuming, cut, map, map_opt, opt, recognize, value},
    error::{context, make_error, ErrorKind, ParseError},
    multi::{separated_list0, separated_list1},
    number::complete::double,
    sequence::{delimited, pair, preceded, terminated, tuple},
    AsChar, Err, IResult, Parser,
};
use serde_json::{json, Number, Value};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompOp {
    EQ,
    LT,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ordering {
    ASC,
    DESC,
}

// is it possible to make Name a &str ? since it refers to the program text
// or maybe some static text.
type Name = String;
type QName = (Option<Name>, Name);

#[derive(Debug, Clone, PartialEq)]
pub struct VarRef {
    pub ref_: Name,
}
impl From<&str> for VarRef {
    fn from(s: &str) -> VarRef {
        VarRef {
            ref_: s.to_string(),
        }
    }
}

// TODO: include input position in here... or rebuild it out of tokens
#[derive(Debug, Clone, PartialEq)] // not sure what the PartialEq is about...
pub enum Expr {
    For {
        for_: Vec<(VarRef, Expr)>, // for $x in collection("captains")
        let_: Vec<(VarRef, Expr)>, // let $century := $x.century
        where_: Box<Expr>,         // where $x.name eq "Kathryn Janeway"
        order: Vec<(Expr, Ordering)>, // order by $x.name
        // TODO: group_by
        return_: Box<Expr>, // return $x
    },
    FnCall(QName, Vec<Expr>), // fn:concat("1","2")
    Comp(CompOp, Box<Expr>, Box<Expr>),
    ArrayUnbox(Box<Expr>),
    ObjectLookup{obj: Box<Expr>, lookup: Box<Expr>},

    Sequence(Vec<Expr>),
    Array(Vec<Expr>),
    Literal(Value),
    VarRef(VarRef),
}


//
// Lexical elements
//

/// Recognises a keyword
///
/// Fails if any alphanumeric characters follow the tag
fn kw<'a, 'tag: 'a, E: ParseError<&'a str>>(
    t: &'tag str,
) -> impl Fn(&'a str) -> IResult<&'a str, &'a str, E> {
    move |i: &'a str| {
        let (remaining, matched) = tag(t)(i)?;

        match satisfy::<_, &str, E>(|c| c.is_alphanum())(remaining) {
            Ok(_) => Err(Err::Error(make_error(i, ErrorKind::Tag))),
            Err(_) => Ok((remaining, matched)),
        }
    }
}

fn colon_eq<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    tag(":=")(i)
}
fn parse_comp_op<'a>(i: &'a str) -> IResult<&'a str, CompOp> {
    alt((
        value(CompOp::EQ, kw("eq")),
        // tag("ne"),
        value(CompOp::LT, kw("lt")),
        //tag("le"),
        //tag("gt"),
        //tag("ge"),
    ))(i)
}

fn parse_str<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    // FIXME: deal with escaped double quotes
    take_while(|c| c != '"')(i)
}

fn parse_var_ref<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    context("variable reference", preceded(char('$'), parse_name))(i)
}

fn parse_name(i: &str) -> IResult<&str, &str> {
    // differs xml name by not including ':'
    fn is_name_start_char(c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }
    // differs xml name, in that '.' is not allowed
    fn is_name_char(c: char) -> bool {
        is_name_start_char(c) || c == '-' || c.is_ascii_digit()
    }

    recognize(pair(
        take_while1(is_name_start_char),
        take_while(is_name_char),
    ))(i)
}

fn boolean<'a>(input: &'a str) -> IResult<&'a str, bool> {
    let parse_true = value(true, kw("true"));
    let parse_false = value(false, kw("false"));
    alt((parse_true, parse_false))(input)
}
fn null<'a>(input: &'a str) -> IResult<&'a str, ()> {
    value((), kw("null"))(input)
}

fn string<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    context(
        "string",
        preceded(char('\"'), cut(terminated(parse_str, char('\"')))),
    )(i)
}

fn parse_literal<'a>(i: &'a str) -> IResult<&'a str, Value> {
    // we don't include arrays or objects because
    // those might contain variable references or other expressions
    alt((
        map(null, |_| Value::Null),
        map(boolean, Value::Bool),
        map(string, |s| Value::String(s.to_string())),
        map_opt(double, |n| Number::from_f64(n).map(Value::Number)),
    ))(i)
}

// Adapted from: https://gist.github.com/eignnx/3c8444b8e2f4d8ce10fcd97815f29d2e
pub mod ws0 {
    use super::*;

    pub fn after<'a, F: 'a, O, E: ParseError<&'a str>>(
        inner: F,
    ) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
    where
        F: Fn(&'a str) -> IResult<&'a str, O, E>,
    {
        terminated(inner, multispace0)
    }

    pub fn around<'a, F: 'a, O, E: ParseError<&'a str>>(
        inner: F,
    ) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
    where
        F: Fn(&'a str) -> IResult<&'a str, O, E>,
    {
        delimited(multispace0, inner, multispace0)
    }

    pub fn before<'a, F: 'a, O, E: ParseError<&'a str>>(
        inner: F,
    ) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
    where
        F: Fn(&'a str) -> IResult<&'a str, O, E>,
    {
        preceded(multispace0, inner)
    }
}

//
// Compound Expressions
//

fn parenthesized_expr<'a>(i: &'a str) -> IResult<&'a str, Expr> {
    let p = preceded(
        ws0::after(char('(')),
        cut(terminated(opt(ws0::after(parse_sequence)), char(')'))),
    )
    .map(|opt_exprs| opt_exprs.unwrap_or_default());

    map(p, |exprs| {
        if exprs.len() == 1 {
            exprs.first().unwrap().to_owned()
        } else {
            Expr::Sequence(exprs)
        }
    })
    .parse(i)
}

fn parse_array(i: &str) -> IResult<&str, Vec<Expr>> {
    // Should we cut, or will it conflict with array unboxing?
    context(
        "array",
        delimited(
            ws0::after(char('[')),
            separated_list0(ws0::after(char(',')), ws0::after(parse_expr)),
            char(']'),
        ),
    )(i)
}

// TODO: support namespaces
fn parse_fn_call(i: &str) -> IResult<&str, (&str, Vec<Expr>)> {
    pair(
        ws0::after(parse_name),
        preceded(
            ws0::after(char('(')),
            cut(terminated(
                separated_list0(ws0::after(char(',')), ws0::after(parse_expr)),
                char(')'),
            )),
        ),
    )(i)
}

fn primary_expr<'a>(i: &'a str) -> IResult<&'a str, Expr> {
    // ... | array | object
    // TODO: rest
    alt((
        parenthesized_expr,
        map(parse_array, Expr::Array),
        map(parse_literal, Expr::Literal),
        map(parse_var_ref, |s| Expr::VarRef(s.into())),
        map(parse_fn_call, |(name, args)| {
            Expr::FnCall((None, name.to_owned()), args)
        }),
    ))(i)
}

// ::= PrimaryExpr ( Predicate | ObjectLookup | ArrayLookup | ArrayUnboxing )*
fn postfix_expr(i: &str) -> IResult<&str, Expr> {
    #[derive(Clone)]
    enum PostfixApply {
        ArrayUnbox,
        ObjectLookup(Expr),
    }

    fn apply_postfix_apply(pf_apply: PostfixApply, e: Expr) -> Expr {
        match pf_apply {
            PostfixApply::ArrayUnbox => Expr::ArrayUnbox(Box::new(e)),
            PostfixApply::ObjectLookup(lookup) => {
                Expr::ObjectLookup{obj: Box::new(e), lookup: Box::new(lookup)}
            }
        }
    }

    fn parse_array_unboxing(i: &str) -> IResult<&str, PostfixApply> {
        value(
            PostfixApply::ArrayUnbox,
            pair(ws0::after(char('[')), char(']')),
        )(i)
    }

    fn parse_object_lookup(i: &str) -> IResult<&str, PostfixApply> {
        // TODO: this should support expressions

        let alts = alt((
            map(string, |s| Value::String(s.to_string())),
            map(parse_name, |s| Value::String(s.to_string())),
        ));
        let alts_exp = map(alts, |s| Expr::Literal(s));

        map(preceded(ws0::after(char('.')), alts_exp), |a| {
            PostfixApply::ObjectLookup(a)
        })(i)
    }

    let (mut remaining, mut res) = primary_expr(i)?;

    let mut single_post_fix = alt((parse_array_unboxing, parse_object_lookup));

    loop {
        // This will turn into an alt that includes object lookup
        match single_post_fix(remaining) {
            Ok((r, apply)) => {
                res = apply_postfix_apply(apply, res);
                remaining = r;
            }
            Err(Err::Error(_)) => return Ok((remaining, res)),
            Err(e) => {
                return Err(e);
            }
        }
    }
}

fn comparison_expr<'a>(i: &'a str) -> IResult<&'a str, Expr> {
    let (remaining, (lhs, opt_rhs)) = pair(
        ws0::after(postfix_expr),
        opt(pair(ws0::after(parse_comp_op), ws0::after(postfix_expr))),
    )(i)?;

    match opt_rhs {
        None => Ok((remaining, lhs)),
        Some((op, rhs)) => {
            Ok((remaining, Expr::Comp(op, Box::new(lhs), Box::new(rhs))))
        }
    }
}

// ---

fn parse_var_ref_in<'a>(i: &'a str) -> IResult<&'a str, (VarRef, Expr)> {
    let (remaining, (var_ref, _, _, expr)) =
        tuple((parse_var_ref, multispace1, ws0::after(kw("in")), parse_expr))(
            i,
        )?;
    Ok((remaining, (var_ref.into(), expr)))
}

fn parse_for_line<'a>(i: &'a str) -> IResult<&'a str, Vec<(VarRef, Expr)>> {
    let (remaining, (_, ins)) = tuple((
        ws0::after(kw("for")),
        cut(separated_list1(ws0::after(char(',')), parse_var_ref_in)),
    ))(i)?;
    Ok((remaining, ins))
}

fn parse_let_binding<'a>(i: &'a str) -> IResult<&'a str, (VarRef, Expr)> {
    let (remaining, (var_ref, _, expr)) =
        tuple((parse_var_ref, ws0::around(colon_eq), parse_expr))(i)?;
    Ok((remaining, (var_ref.into(), expr)))
}

fn parse_let_line(i: &str) -> IResult<&str, Vec<(VarRef, Expr)>> {
    let (remaining, (_, bindings)) = tuple((
        ws0::after(kw("let")),
        separated_list1(ws0::after(char(',')), parse_let_binding),
    ))(i)?;
    Ok((remaining, bindings))
}

fn parse_where_line(i: &str) -> IResult<&str, Expr> {
    let (remaining, (_, bindings)) =
        tuple((ws0::after(kw("where")), parse_expr))(i)?;
    Ok((remaining, bindings))
}

fn parse_return(i: &str) -> IResult<&str, Expr> {
    preceded(kw("return"), cut(ws0::before(parse_expr)))(i)
}

fn parse_flwor(i: &str) -> IResult<&str, Expr> {
    // TODO: rest

    let (remaining, (for_line, let_line_opt, where_line_opt, ret_expr)) =
        tuple((
            ws0::after(parse_for_line),
            opt(ws0::after(parse_let_line)),
            opt(ws0::after(parse_where_line)),
            parse_return,
        ))(i)?;
    Ok((
        remaining,
        Expr::For {
            for_: for_line,
            let_: let_line_opt.unwrap_or_default(),
            where_: Box::new(
                where_line_opt.unwrap_or(Expr::Literal(json!(true))),
            ),
            order: vec![],
            return_: Box::new(ret_expr),
        },
    ))
}

fn parse_expr<'a>(i: &'a str) -> IResult<&'a str, Expr> {
    // TODO: logicals, arithmetic
    comparison_expr(i)
}

fn parse_sequence<'a>(i: &'a str) -> IResult<&'a str, Vec<Expr>> {
    separated_list1(ws0::after(char(',')), ws0::after(parse_expr)).parse(i)
}

// fncall identified , terminated('(', many0(expr)  ')')

pub fn parse_main_module(i: &str) -> IResult<&str, Expr> {
    all_consuming(parse_flwor)(i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::{Error, ErrorKind};
    use serde_json::json;

    // Lexical
    // -------

    #[test]
    fn test_parse_bool() {
        assert_eq!(boolean("true"), Ok(("", true)));
        assert_eq!(boolean("true "), Ok((" ", true)));
        assert_eq!(boolean("false"), Ok(("", false)));

        assert_eq!(
            boolean("truer"),
            Err(Err::Error(Error::new("truer", ErrorKind::Tag)))
        );
        assert_eq!(
            boolean("falsey"),
            Err(Err::Error(Error::new("falsey", ErrorKind::Tag)))
        );
    }

    #[test]
    fn test_parse_str() {
        assert_eq!(parse_str("captains"), Ok(("", "captains")));
        assert_eq!(parse_str("line1\\\nline2"), Ok(("", "line1\\\nline2")));
        // FIXME
        // assert_eq!(string("cap\\\"tains"), Ok(("", "cap\\\"tains")));
    }
    #[test]
    fn test_string() {
        assert_eq!(string("\"captains\""), Ok(("", "captains")));
    }

    #[test]
    fn test_parse_null() {
        assert_eq!(null("null"), Ok(("", ())));

        assert_eq!(
            null("nuller"),
            Err(Err::Error(Error::new("nuller", ErrorKind::Tag)))
        );
    }

    // Compound Exprs
    // --------------

    #[test]
    fn test_parse_parenthesized_expr() {
        assert_eq!(parse_expr("()"), Ok(("", Expr::Sequence(vec![]))));
        assert_eq!(parse_expr("(  )"), Ok(("", Expr::Sequence(vec![]))));

        assert_eq!(
            parenthesized_expr("(1)"),
            Ok(("", Expr::Literal(json!(1.0))))
        );
        assert_eq!(
            parenthesized_expr("( 1 )"),
            Ok(("", Expr::Literal(json!(1.0))))
        );

        assert_eq!(parse_expr("( )"), Ok(("", Expr::Sequence(vec![]))));
        assert_eq!(
            parse_expr("( 1 , 2 )"),
            Ok((
                "",
                Expr::Sequence(vec![
                    Expr::Literal(json!(1.0)),
                    Expr::Literal(json!(2.0))
                ])
            ))
        );
    }

    #[test]
    fn test_postfix_expr() {
        assert_eq!(
            parse_expr("$x[]"),
            Ok(("", Expr::ArrayUnbox(Box::new(Expr::VarRef("x".into())))))
        );
    }

    #[test]
    fn test_parse_array() {
        assert_eq!(parse_array("[]"), Ok(("", vec![])));
        assert_eq!(parse_array("[  ]"), Ok(("", vec![])));
        assert_eq!(
            parse_array("[1]"),
            Ok(("", vec![Expr::Literal(json!(1.0))]))
        );
        assert_eq!(
            parse_array("[ 1 ]"),
            Ok(("", vec![Expr::Literal(json!(1.0))]))
        );
        assert_eq!(
            parse_array("[ 1 , 2 ]"),
            Ok((
                "",
                vec![Expr::Literal(json!(1.0)), Expr::Literal(json!(2.0))]
            ))
        );
    }
    #[test]
    fn test_parse_fn_call() {
        assert_eq!(
            parse_fn_call("parse(1)"),
            Ok(("", ("parse", vec![Expr::Literal(json!(1.0))])))
        );
        assert_eq!(
            parse_fn_call("string-length7 ( 1 , 2 )"),
            Ok((
                "",
                (
                    "string-length7",
                    vec![Expr::Literal(json!(1.0)), Expr::Literal(json!(2.0))]
                )
            ))
        );
    }

    #[test]
    fn test_parse_var_ref() {
        assert_eq!(parse_var_ref("$x"), Ok(("", "x")));
        assert_eq!(parse_var_ref("$xxx"), Ok(("", "xxx")));
        assert_eq!(
            parse_var_ref("x"),
            Err(Err::Error(Error::new("x", ErrorKind::Char)))
        );
    }

    #[test]
    fn test_parse_for_line() {
        assert_eq!(
            parse_for_line("for $x in []"),
            Ok(("", vec![(VarRef::from("x"), Expr::Array(vec![]))]))
        );
        assert_eq!(
            parse_for_line("for $x in ()"),
            Ok(("", vec![(VarRef::from("x"), Expr::Sequence(vec![]))]))
        );
        assert_eq!(
            parse_for_line("for $x in (1,2)"),
            Ok((
                "",
                vec![(
                    VarRef::from("x"),
                    Expr::Sequence(vec![
                        Expr::Literal(json!(1.0)),
                        Expr::Literal(json!(2.0)),
                    ])
                )]
            ))
        );
        assert_eq!(
            parse_for_line("for$x in[]"),
            Ok(("", vec![(VarRef::from("x"), Expr::Array(vec![]))]))
        );
        assert_eq!(
            parse_for_line("for $x innull"),
            Err(Err::Failure(Error::new("innull", ErrorKind::Tag)))
        );
        assert_eq!(
            parse_for_line("for $x in [], $y in []"),
            Ok((
                "",
                vec![
                    (VarRef::from("x"), Expr::Array(vec![])),
                    (VarRef::from("y"), Expr::Array(vec![])),
                ]
            ))
        );
    }

    #[test]
    fn test_parse_let_line() {
        assert_eq!(
            parse_let_line("let $y := 5"),
            Ok(("", vec![(VarRef::from("y"), Expr::Literal(json!(5.0)))]))
        );
        assert_eq!(
            parse_let_line("let $y := 5, $x := $y"),
            Ok((
                "",
                vec![
                    (VarRef::from("y"), Expr::Literal(json!(5.0))),
                    (VarRef::from("x"), Expr::VarRef("y".into())),
                ]
            ))
        );
    }

    #[test]
    fn test_parse_where_line() {
        assert_eq!(
            parse_where_line("where $z eq 5"),
            Ok((
                "",
                Expr::Comp(
                    CompOp::EQ,
                    Box::new(Expr::VarRef("z".into())),
                    Box::new(Expr::Literal(json!(5.0)))
                )
            ))
        );
    }

    #[test]
    fn test_parse_return() {
        assert_eq!(
            parse_return("return $x"),
            Ok(("", Expr::VarRef("x".into())))
        );
        assert_eq!(parse_return("return$x"), parse_return("return $x"));
        assert_eq!(parse_return("return  $x"), parse_return("return $x"));
    }

    #[test]
    fn test_parse_flwor() {
        assert_eq!(
            parse_flwor(
                "for $x in ()
                let $y := $x
                where $x lt 3
                return [$x, $y]\n"
            ),
            Ok((
                "",
                Expr::For {
                    for_: vec![(VarRef::from("x"), Expr::Sequence(vec![]))],
                    let_: vec![(VarRef::from("y"), Expr::VarRef("x".into()))],
                    where_: Box::new(Expr::Comp(
                        CompOp::LT,
                        Box::new(Expr::VarRef("x".into())),
                        Box::new(Expr::Literal(json!(3.0))),
                    )),
                    order: vec![],
                    return_: Box::new(Expr::Array(vec![
                        Expr::VarRef("x".into()),
                        Expr::VarRef("y".into()),
                    ])),
                }
            ))
        );
        assert!(parse_flwor("for $x in () return $x\n").is_ok())
    }
}
