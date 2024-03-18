use std::vec;

use nom::{
    branch::alt,
    bytes::complete::{escaped, tag, take_while},
    character::complete::{
        alphanumeric1, char, multispace0, multispace1, one_of, satisfy,
    },
    combinator::{cut, map, map_opt, value},
    error::{context, make_error, ErrorKind, ParseError},
    multi::{separated_list0, separated_list1},
    number::complete::double,
    sequence::{delimited, preceded, terminated, tuple},
    AsChar, Err, IResult,
};
use serde_json::{json, Number, Value};

use crate::ast::{CompOp, Expr, VarRef};

fn sp<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    let chars = " \t\r\n";
    take_while(move |c| chars.contains(c))(i)
}

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
    escaped(alphanumeric1, '\\', one_of("\"n\\"))(i)
}

fn parse_var_ref<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    context("variable reference", preceded(char('$'), alphanumeric1))(i)
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
        terminated(inner, multispace0)
    }
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

fn parse_var_ref_in<'a>(i: &'a str) -> IResult<&'a str, (VarRef, Expr)> {
    let (remaining, (var_ref, _, _, _, expr)) = tuple((
        parse_var_ref,
        multispace1,
        kw("in"),
        multispace0,
        parse_expr,
    ))(i)?;
    Ok((remaining, (var_ref.into(), expr)))
}

fn parse_for_line<'a>(i: &'a str) -> IResult<&'a str, Vec<(VarRef, Expr)>> {
    let (remaining, (_, _, ins)) = tuple((
        kw("for"),
        multispace0,
        separated_list1(char(','), parse_var_ref_in),
    ))(i)?;
    Ok((remaining, ins))
}

fn parse_return(i: &str) -> IResult<&str, Expr> {
    preceded(kw("return"), cut(preceded(multispace0, parse_expr)))(i)
}

pub fn parse_flwor(i: &str) -> IResult<&str, Expr> {
    // TODO: rest

    let (remaining, (for_line, _, ret_expr)) =
        tuple((parse_for_line, multispace0, parse_return))(i)?;
    Ok((
        remaining,
        Expr::For {
            for_: for_line,
            let_: vec![],
            where_: Box::new(Expr::Literal(json!(true))),
            order: vec![],
            return_: Box::new(ret_expr),
        },
    ))
}

fn parse_expr<'a>(i: &'a str) -> IResult<&'a str, Expr> {
    // ... | array | object | literal
    // TODO: rest
    alt((
        map(parse_array, Expr::Array),
        map(parse_literal, Expr::Literal),
        map(parse_var_ref, |s| Expr::VarRef(s.into())),
    ))(i)
}

// fncall identified , terminated('(', many0(expr)  ')')

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::{Error, ErrorKind};
    use serde_json::json;

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
    }

    #[test]
    fn test_parse_null() {
        assert_eq!(null("null"), Ok(("", ())));

        assert_eq!(
            null("nuller"),
            Err(Err::Error(Error::new("nuller", ErrorKind::Tag)))
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
            parse_for_line("for$x in[]"),
            Ok(("", vec![(VarRef::from("x"), Expr::Array(vec![]))]))
        );
        assert_eq!(
            parse_for_line("for $x innull"),
            Err(Err::Error(Error::new("innull", ErrorKind::Tag)))
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
}
