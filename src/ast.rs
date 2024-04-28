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
    FnCall(QName, Vec<Expr>), // fn:concat("1","2")
    Comp(CompOp, Box<Expr>, Box<Expr>),
    ArrayUnbox(Box<Expr>),

    Sequence(Vec<Expr>),
    Array(Vec<Expr>),
    Literal(Value),
    VarRef(VarRef),
}

// TODO: come up with a typed AST
