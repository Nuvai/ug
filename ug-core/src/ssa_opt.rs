use crate::lang::ssa::{A, Instr, Kernel, VarId};
use crate::lang::{BinaryOp, Const, UnaryOp};
use crate::Result;

fn resolve_const(instrs: &[Instr], a: A) -> Option<Const> {
    match a {
        A::Const(c) => Some(c),
        A::Var(v) => match &instrs[v.as_usize()] {
            Instr::Const(c) => Some(*c),
            _ => None,
        },
    }
}

fn fold_binary(op: BinaryOp, lhs: Const, rhs: Const) -> Option<Const> {
    match (lhs, rhs) {
        (Const::I32(l), Const::I32(r)) => {
            let v = match op {
                BinaryOp::Add => l.checked_add(r)?,
                BinaryOp::Sub => l.checked_sub(r)?,
                BinaryOp::Mul => l.checked_mul(r)?,
                BinaryOp::Div => l.checked_div(r)?,
                BinaryOp::Mod => l.checked_rem(r)?,
                BinaryOp::Min => Some(l.min(r))?,
                BinaryOp::Max => Some(l.max(r))?,
                BinaryOp::Eq => return Some(Const::I32(if l == r { 1 } else { 0 })),
                BinaryOp::Ne => return Some(Const::I32(if l != r { 1 } else { 0 })),
                BinaryOp::Lt => return Some(Const::I32(if l < r { 1 } else { 0 })),
                BinaryOp::Le => return Some(Const::I32(if l <= r { 1 } else { 0 })),
                BinaryOp::Gt => return Some(Const::I32(if l > r { 1 } else { 0 })),
                BinaryOp::Ge => return Some(Const::I32(if l >= r { 1 } else { 0 })),
            };
            Some(Const::I32(v))
        }
        (Const::F32(l), Const::F32(r)) => {
            let (l, r) = (*l, *r);
            let v = match op {
                BinaryOp::Add => l + r,
                BinaryOp::Sub => l - r,
                BinaryOp::Mul => l * r,
                BinaryOp::Div => l / r,
                BinaryOp::Mod => l % r,
                BinaryOp::Min => l.min(r),
                BinaryOp::Max => l.max(r),
                BinaryOp::Eq => return Some(Const::I32(if l == r { 1 } else { 0 })),
                BinaryOp::Ne => return Some(Const::I32(if l != r { 1 } else { 0 })),
                BinaryOp::Lt => return Some(Const::I32(if l < r { 1 } else { 0 })),
                BinaryOp::Le => return Some(Const::I32(if l <= r { 1 } else { 0 })),
                BinaryOp::Gt => return Some(Const::I32(if l > r { 1 } else { 0 })),
                BinaryOp::Ge => return Some(Const::I32(if l >= r { 1 } else { 0 })),
            };
            Const::try_from(v).ok()
        }
        _ => None,
    }
}

fn fold_unary(op: UnaryOp, arg: Const) -> Option<Const> {
    match arg {
        Const::F32(v) => {
            let v = *v;
            let r = match op {
                UnaryOp::Neg => -v,
                UnaryOp::Exp => v.exp(),
                UnaryOp::Sin => v.sin(),
                UnaryOp::Cos => v.cos(),
                UnaryOp::Sqrt => v.sqrt(),
                UnaryOp::Id => v,
                UnaryOp::Cast(dt) => match dt {
                    crate::DType::I32 => return Some(Const::I32(v as i32)),
                    crate::DType::I64 => return Some(Const::I64(v as i64)),
                    _ => return None,
                },
            };
            Const::try_from(r).ok()
        }
        Const::I32(v) => match op {
            UnaryOp::Neg => Some(Const::I32(-v)),
            UnaryOp::Id => Some(Const::I32(v)),
            UnaryOp::Cast(dt) => match dt {
                crate::DType::F32 => Const::try_from(v as f32).ok(),
                crate::DType::I64 => Some(Const::I64(v as i64)),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    }
}

pub fn constant_fold(instrs: &mut Vec<Instr>) {
    for i in 0..instrs.len() {
        let replacement = match &instrs[i] {
            Instr::Binary { op, lhs, rhs, dtype: _ } => {
                let lc = resolve_const(instrs, *lhs);
                let rc = resolve_const(instrs, *rhs);
                match (lc, rc) {
                    (Some(l), Some(r)) => fold_binary(*op, l, r).map(Instr::Const),
                    _ => None,
                }
            }
            Instr::Unary { op, arg, dtype: _ } => {
                resolve_const(instrs, *arg).and_then(|c| fold_unary(*op, c).map(Instr::Const))
            }
            _ => None,
        };
        if let Some(new_instr) = replacement {
            instrs[i] = new_instr;
        }
    }
}

pub fn dead_code_elimination(instrs: &mut Vec<Instr>) {
    let n = instrs.len();
    let mut live = vec![false; n];

    fn mark_a(a: A, live: &mut [bool]) {
        if let A::Var(v) = a {
            live[v.as_usize()] = true;
        }
    }

    fn mark_var(v: VarId, live: &mut [bool]) {
        live[v.as_usize()] = true;
    }

    for i in (0..n).rev() {
        if live[i] {
            mark_deps(&instrs[i], &mut live);
            continue;
        }
        match &instrs[i] {
            Instr::Store { dst, offset, value, .. } => {
                live[i] = true;
                mark_var(*dst, &mut live);
                mark_a(*offset, &mut live);
                mark_a(*value, &mut live);
            }
            Instr::Assign { dst, src } => {
                live[i] = true;
                mark_var(*dst, &mut live);
                mark_a(*src, &mut live);
            }
            Instr::DefineGlobal { .. } | Instr::DefineLocal { .. } => {
                live[i] = true;
            }
            Instr::Range { end_idx, .. } => {
                live[i] = true;
                live[end_idx.as_usize()] = true;
                mark_deps(&instrs[i], &mut live);
            }
            Instr::EndRange { start_idx } => {
                live[i] = true;
                live[start_idx.as_usize()] = true;
            }
            Instr::If { end_idx, .. } => {
                live[i] = true;
                live[end_idx.as_usize()] = true;
                mark_deps(&instrs[i], &mut live);
            }
            Instr::EndIf => {
                live[i] = true;
            }
            Instr::Barrier => {
                live[i] = true;
            }
            Instr::Special(_) | Instr::DefineAcc(_) => {
                live[i] = true;
            }
            Instr::ReduceLocal { arg, .. } => {
                live[i] = true;
                mark_a(*arg, &mut live);
            }
            _ => {}
        }
    }

    for i in (0..n).rev() {
        if live[i] {
            mark_deps(&instrs[i], &mut live);
        }
    }

    for i in 0..n {
        if !live[i] {
            instrs[i] = Instr::Const(Const::I32(0));
        }
    }
}

fn mark_deps(instr: &Instr, live: &mut [bool]) {
    fn mark_a(a: A, live: &mut [bool]) {
        if let A::Var(v) = a {
            live[v.as_usize()] = true;
        }
    }
    match instr {
        Instr::Binary { lhs, rhs, .. } => {
            mark_a(*lhs, live);
            mark_a(*rhs, live);
        }
        Instr::Unary { arg, .. } => mark_a(*arg, live),
        Instr::Load { src, offset, .. } => {
            live[src.as_usize()] = true;
            mark_a(*offset, live);
        }
        Instr::Store { dst, offset, value, .. } => {
            live[dst.as_usize()] = true;
            mark_a(*offset, live);
            mark_a(*value, live);
        }
        Instr::Assign { dst, src } => {
            live[dst.as_usize()] = true;
            mark_a(*src, live);
        }
        Instr::Range { lo, up, .. } => {
            mark_a(*lo, live);
            mark_a(*up, live);
        }
        Instr::If { cond, .. } => mark_a(*cond, live),
        Instr::Where { cond, on_true, on_false, .. } => {
            mark_a(*cond, live);
            mark_a(*on_true, live);
            mark_a(*on_false, live);
        }
        Instr::ReduceLocal { arg, .. } => mark_a(*arg, live),
        Instr::DefineAcc(_)
        | Instr::DefineGlobal { .. }
        | Instr::DefineLocal { .. }
        | Instr::Special(_)
        | Instr::Const(_)
        | Instr::EndRange { .. }
        | Instr::EndIf
        | Instr::Barrier => {}
    }
}

impl Kernel {
    pub fn optimize(mut self) -> Result<Self> {
        constant_fold(self.instrs_mut());
        dead_code_elimination(self.instrs_mut());
        Ok(self)
    }
}
