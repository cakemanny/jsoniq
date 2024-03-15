use nom::{
    branch::alt,
    bytes::complete::{escaped, tag, take_while},
    character::complete::{alphanumeric1, char, multispace1, one_of},
    combinator::{cut, map, map_opt, opt, value},
    error::{context, Error},
    multi::separated_list1,
    number::complete::double,
    sequence::{delimited, preceded, separated_pair, terminated, tuple},
    Err, IResult,
};
use serde_json::{json, Number, Value};

use crate::ast::{CompOp, Expr, VarRef};

fn sp<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    let chars = " \t\r\n";
    take_while(move |c| chars.contains(c))(i)
}

fn colon_eq<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    tag(":=")(i)
}
fn parse_comp_op<'a>(i: &'a str) -> IResult<&'a str, CompOp> {
    alt((
        value(CompOp::EQ, tag("eq")),
        // tag("ne"),
        value(CompOp::LT, tag("lt")),
        //tag("le"),
        //tag("gt"),
        //tag("ge"),
    ))(i)
}

fn parse_str<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    escaped(alphanumeric1, '\\', one_of("\"n\\"))(i)
}

fn parse_var_ref<'a>(i: &'a str) -> IResult<&'a str, &'a str> {
    preceded(char('$'), alphanumeric1)(i)
}

fn boolean<'a>(input: &'a str) -> IResult<&'a str, bool> {
    let parse_true = value(true, tag("true"));
    let parse_false = value(false, tag("false"));
    alt((parse_true, parse_false))(input)
}
fn null<'a>(input: &'a str) -> IResult<&'a str, ()> {
    value((), tag("null"))(input)
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

fn parse_var_ref_in<'a>(i: &'a str) -> IResult<&'a str, (VarRef, Expr)> {
    let (remaining, (var_ref, _, _, _, expr)) = tuple((
        parse_var_ref,
        multispace1,
        tag("in"),
        multispace1,
        parse_expr,
    ))(i)?;
    Ok((remaining, (var_ref.into(), expr)))
}

fn parse_for_line<'a>(i: &'a str) -> IResult<&'a str, Vec<(VarRef, Expr)>> {
    let (remaining, (_, _, ins)) = tuple((
        tag("for"),
        multispace1,
        separated_list1(char(','), parse_var_ref_in),
    ))(i)?;
    Ok((remaining, ins))
}

fn parse_expr<'a>(i: &'a str) -> IResult<&'a str, Expr> {
    // ... | array | object | literal
    // TODO: rest
    map(parse_literal, Expr::Literal)(i)
}

// fncall identified , terminated('(', many0(expr)  ')')

#[cfg(test)]
mod tests {
    use super::*;
    use nom::error::ErrorKind;
    use serde_json::json;

    #[test]
    fn test_parse_str() {
        assert_eq!(parse_str("captains"), Ok(("", "captains")));
    }
    #[test]
    fn test_parse_null() {
        assert_eq!(null("null"), Ok(("", ())));
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
            parse_for_line("for $x in null"),
            Ok(("", vec![(VarRef::from("x"), Expr::Literal(json!(null)))]))
        );
    }
}
