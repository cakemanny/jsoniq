use serde_json::Value;

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

type Name = String;

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

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
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
