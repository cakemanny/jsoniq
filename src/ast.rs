use serde_json::Value;

use crate::parse;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompOp {
    EQ,
    LT,
}
impl From<parse::CompOp> for CompOp {
    fn from(value: parse::CompOp) -> Self {
        match value {
            parse::CompOp::EQ => CompOp::EQ,
            parse::CompOp::LT => CompOp::LT,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Ordering {
    ASC,
    DESC,
}
impl From<parse::Ordering> for Ordering {
    fn from(value: parse::Ordering) -> Self {
        match value {
            parse::Ordering::ASC => Ordering::ASC,
            parse::Ordering::DESC => Ordering::DESC,
        }
    }
}

// is it possible to make Name a &str ? since it refers to the program text
// or maybe some static text.
type Name = String;
type QName = (Option<Name>, Name);

#[derive(Debug, Clone, PartialEq)]
pub struct VarRef {
    pub ref_: Name,
}
impl From<parse::VarRef> for VarRef {
    fn from(value: parse::VarRef) -> Self {
        VarRef { ref_: value.ref_ }
    }
}
impl From<&str> for VarRef {
    fn from(s: &str) -> VarRef {
        VarRef {
            ref_: s.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FLWORClause {
    // TODO: simplify so that these don't contain Vecs
    For(Vec<(VarRef, Expr)>), // for $x in collection("captains")
    Let(Vec<(VarRef, Expr)>), // let $century := $x.century
    Where(Box<Expr>),         // where $x.name eq "Kathryn Janeway"
    Order(Vec<(Expr, Ordering)>), // order by $x.name
                              // TODO GroupBy
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    For {
        clauses: Vec<FLWORClause>,
        return_: Box<Expr>, // return $x
    },
    // TODO: consider expanding these to fixed size known calls when possible?
    FnCall(QName, Vec<Expr>), // fn:concat("1","2")
    Comp(CompOp, Box<Expr>, Box<Expr>),
    ArrayUnbox(Box<Expr>),
    ObjectLookup{obj: Box<Expr>, lookup: Box<Expr>},

    Sequence(Vec<Expr>),
    Array(Vec<Expr>),
    Literal(Value),
    VarRef(VarRef),
}

// TODO: come up with a typed AST
