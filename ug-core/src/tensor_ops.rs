use crate::block::Block;
use crate::lang::ssa::Instr as I;
use crate::lang::{BinaryOp as B, DType, ReduceOp, UnaryOp};
use crate::{Device, LazyBuffer, Result};

impl<D: Device> LazyBuffer<D> {
    pub fn pad(&self, dim: usize, pad_before: usize, pad_after: usize, pad_val: f32) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            crate::bail!("pad dim {dim} out of range for shape {dims:?}")
        }
        let src_dim = dims[dim];
        let dst_dim = src_dim + pad_before + pad_after;
        let mut dst_dims = dims.to_vec();
        dst_dims[dim] = dst_dim;
        let outer: usize = dims[..dim].iter().product();
        let inner: usize = dims[dim + 1..].iter().product();
        let dtype = self.dtype();

        let mut b = Block::empty();
        let src = b.push(I::DefineGlobal { index: 0, dtype }).to_varid();
        let dst = b.push(I::DefineGlobal { index: 1, dtype }).to_varid();
        let r_outer = b.range(0, outer as i32, 1);
        let r_dst_dim = b.range(0, dst_dim as i32, 1);
        let r_inner = b.range(0, inner as i32, 1);

        let dst_off = b.mul(r_outer.id(), (dst_dim * inner) as i32);
        let dst_off = b.add(dst_off, 0);
        let tmp = b.mul(r_dst_dim.id(), inner as i32);
        let dst_off = b.binary(B::Add, dst_off, tmp, DType::I32);
        let dst_off = b.binary(B::Add, dst_off, r_inner.id(), DType::I32);

        let pad_before_cst = b.cst(pad_before as i32);
        let pad_end_cst = b.cst((pad_before + src_dim) as i32);
        let in_range_lo = b.binary(B::Ge, r_dst_dim.id(), pad_before_cst, DType::I32);
        let in_range_hi = b.binary(B::Lt, r_dst_dim.id(), pad_end_cst, DType::I32);
        let in_range = b.binary(B::Mul, in_range_lo, in_range_hi, DType::I32);

        let src_idx = b.binary(B::Sub, r_dst_dim.id(), pad_before_cst, DType::I32);
        let src_off = b.mul(r_outer.id(), (src_dim * inner) as i32);
        let tmp = b.mul(src_idx, inner as i32);
        let src_off = b.binary(B::Add, src_off, tmp, DType::I32);
        let src_off = b.binary(B::Add, src_off, r_inner.id(), DType::I32);

        let load_val = b.push(I::Load { src, offset: src_off.to_a(), dtype });
        let pad_cst = b.push(I::Const(crate::Const::try_from(pad_val)?));
        let pad_cst = b.unary(UnaryOp::Cast(dtype), pad_cst, dtype);
        let result = b.push(I::Where {
            cond: in_range.to_a(),
            on_true: load_val.to_a(),
            on_false: pad_cst.to_a(),
            dtype,
        });
        b.push(I::Store { dst, offset: dst_off.to_a(), value: result.to_a(), dtype });

        b.end_range(r_inner)?;
        b.end_range(r_dst_dim)?;
        b.end_range(r_outer)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(ssa, vec![self.clone()], dst_dims, dtype, self.device())
    }

    pub fn gather(&self, indices: &Self, dim: usize) -> Result<Self> {
        let src_dims = self.dims();
        let idx_dims = indices.dims();
        if src_dims.len() != idx_dims.len() {
            crate::bail!("gather rank mismatch: src {:?} vs idx {:?}", src_dims, idx_dims)
        }
        if dim >= src_dims.len() {
            crate::bail!("gather dim {dim} out of range for shape {src_dims:?}")
        }
        let dtype = self.dtype();
        let total_elems: usize = idx_dims.iter().product();

        let mut strides_src = vec![0usize; src_dims.len()];
        let mut strides_idx = vec![0usize; idx_dims.len()];
        let mut s = 1;
        for i in (0..src_dims.len()).rev() {
            strides_src[i] = s;
            s *= src_dims[i];
        }
        s = 1;
        for i in (0..idx_dims.len()).rev() {
            strides_idx[i] = s;
            s *= idx_dims[i];
        }

        let mut b = Block::empty();
        let src = b.push(I::DefineGlobal { index: 0, dtype }).to_varid();
        let idx = b.push(I::DefineGlobal { index: 1, dtype: DType::I32 }).to_varid();
        let dst_buf = b.push(I::DefineGlobal { index: 2, dtype }).to_varid();

        let r = b.range(0, total_elems as i32, 1);
        let idx_val = b.push(I::Load { src: idx, offset: r.id().to_a(), dtype: DType::I32 });

        let mut src_off = b.cst(0i32);
        let mut remaining = r.id();
        for i in 0..idx_dims.len() {
            let coord = if i + 1 < idx_dims.len() {
                let stride_cst = b.cst(strides_idx[i] as i32);
                let c = b.binary(B::Div, remaining, stride_cst, DType::I32);
                let used = b.binary(B::Mul, c, stride_cst, DType::I32);
                remaining = b.binary(B::Sub, remaining, used, DType::I32);
                c
            } else {
                remaining
            };

            let dim_coord = if i == dim { idx_val } else { coord };
            let s = b.cst(strides_src[i] as i32);
            let off = b.binary(B::Mul, dim_coord, s, DType::I32);
            src_off = b.binary(B::Add, src_off, off, DType::I32);
        }

        let val = b.push(I::Load { src, offset: src_off.to_a(), dtype });
        b.push(I::Store { dst: dst_buf, offset: r.id().to_a(), value: val.to_a(), dtype });
        b.end_range(r)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(ssa, vec![self.clone(), indices.clone()], idx_dims.to_vec(), dtype, self.device())
    }

    pub fn scatter(&self, indices: &Self, src_vals: &Self, dim: usize) -> Result<Self> {
        let dst_dims = self.dims();
        let idx_dims = indices.dims();
        let src_dims = src_vals.dims();
        if idx_dims != src_dims {
            crate::bail!("scatter: indices and src must have same shape, got {:?} vs {:?}", idx_dims, src_dims)
        }
        if dim >= dst_dims.len() {
            crate::bail!("scatter dim {dim} out of range for shape {dst_dims:?}")
        }
        let dtype = self.dtype();
        let total_dst: usize = dst_dims.iter().product();
        let total_idx: usize = idx_dims.iter().product();

        let mut strides_dst = vec![0usize; dst_dims.len()];
        let mut strides_idx = vec![0usize; idx_dims.len()];
        let mut s = 1;
        for i in (0..dst_dims.len()).rev() {
            strides_dst[i] = s;
            s *= dst_dims[i];
        }
        s = 1;
        for i in (0..idx_dims.len()).rev() {
            strides_idx[i] = s;
            s *= idx_dims[i];
        }

        let mut b = Block::empty();
        let self_buf = b.push(I::DefineGlobal { index: 0, dtype }).to_varid();
        let idx_buf = b.push(I::DefineGlobal { index: 1, dtype: DType::I32 }).to_varid();
        let src_buf = b.push(I::DefineGlobal { index: 2, dtype }).to_varid();
        let dst_buf = b.push(I::DefineGlobal { index: 3, dtype }).to_varid();

        // Copy self to dst
        let r_copy = b.range(0, total_dst as i32, 1);
        let v = b.push(I::Load { src: self_buf, offset: r_copy.id().to_a(), dtype });
        b.push(I::Store { dst: dst_buf, offset: r_copy.id().to_a(), value: v.to_a(), dtype });
        b.end_range(r_copy)?;

        // Scatter src into dst at index positions
        let r = b.range(0, total_idx as i32, 1);
        let idx_val = b.push(I::Load { src: idx_buf, offset: r.id().to_a(), dtype: DType::I32 });
        let src_val = b.push(I::Load { src: src_buf, offset: r.id().to_a(), dtype });

        let mut dst_off = b.cst(0i32);
        let mut remaining = r.id();
        for i in 0..idx_dims.len() {
            let coord = if i + 1 < idx_dims.len() {
                let stride_cst = b.cst(strides_idx[i] as i32);
                let c = b.binary(B::Div, remaining, stride_cst, DType::I32);
                let used = b.binary(B::Mul, c, stride_cst, DType::I32);
                remaining = b.binary(B::Sub, remaining, used, DType::I32);
                c
            } else {
                remaining
            };
            let dim_coord = if i == dim { idx_val } else { coord };
            let s = b.cst(strides_dst[i] as i32);
            let off = b.binary(B::Mul, dim_coord, s, DType::I32);
            dst_off = b.binary(B::Add, dst_off, off, DType::I32);
        }

        b.push(I::Store { dst: dst_buf, offset: dst_off.to_a(), value: src_val.to_a(), dtype });
        b.end_range(r)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(
            ssa,
            vec![self.clone(), indices.clone(), src_vals.clone()],
            dst_dims.to_vec(),
            dtype,
            self.device(),
        )
    }

    pub fn cumsum(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            crate::bail!("cumsum dim {dim} out of range for shape {dims:?}")
        }
        let dtype = self.dtype();
        let outer: usize = dims[..dim].iter().product();
        let axis_len = dims[dim];
        let inner: usize = dims[dim + 1..].iter().product();

        let mut b = Block::empty();
        let src = b.push(I::DefineGlobal { index: 0, dtype }).to_varid();
        let dst = b.push(I::DefineGlobal { index: 1, dtype }).to_varid();

        let r_outer = b.range(0, outer as i32, 1);
        let r_inner = b.range(0, inner as i32, 1);

        let acc = b.push(I::DefineAcc(crate::Const::zero(dtype)));
        let r_axis = b.range(0, axis_len as i32, 1);

        let off = b.mul(r_outer.id(), (axis_len * inner) as i32);
        let tmp = b.mul(r_axis.id(), inner as i32);
        let off = b.binary(B::Add, off, tmp, DType::I32);
        let off = b.binary(B::Add, off, r_inner.id(), DType::I32);

        let val = b.push(I::Load { src, offset: off.to_a(), dtype });
        let new_acc = b.binary(B::Add, acc, val, dtype);
        b.push(I::Assign { dst: acc.to_varid(), src: new_acc.to_a() });
        b.push(I::Store { dst, offset: off.to_a(), value: acc.to_a(), dtype });

        b.end_range(r_axis)?;
        b.end_range(r_inner)?;
        b.end_range(r_outer)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(ssa, vec![self.clone()], dims.to_vec(), dtype, self.device())
    }

    pub fn cumprod(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            crate::bail!("cumprod dim {dim} out of range for shape {dims:?}")
        }
        let dtype = self.dtype();
        let outer: usize = dims[..dim].iter().product();
        let axis_len = dims[dim];
        let inner: usize = dims[dim + 1..].iter().product();

        let one = match dtype {
            DType::F32 => crate::Const::try_from(1.0f32)?,
            DType::I32 => crate::Const::I32(1),
            DType::I64 => crate::Const::I64(1),
            dt => crate::bail!("cumprod unsupported dtype {dt:?}"),
        };

        let mut b = Block::empty();
        let src = b.push(I::DefineGlobal { index: 0, dtype }).to_varid();
        let dst = b.push(I::DefineGlobal { index: 1, dtype }).to_varid();

        let r_outer = b.range(0, outer as i32, 1);
        let r_inner = b.range(0, inner as i32, 1);

        let acc = b.push(I::DefineAcc(one));
        let r_axis = b.range(0, axis_len as i32, 1);

        let off = b.mul(r_outer.id(), (axis_len * inner) as i32);
        let tmp = b.mul(r_axis.id(), inner as i32);
        let off = b.binary(B::Add, off, tmp, DType::I32);
        let off = b.binary(B::Add, off, r_inner.id(), DType::I32);

        let val = b.push(I::Load { src, offset: off.to_a(), dtype });
        let new_acc = b.binary(B::Mul, acc, val, dtype);
        b.push(I::Assign { dst: acc.to_varid(), src: new_acc.to_a() });
        b.push(I::Store { dst, offset: off.to_a(), value: acc.to_a(), dtype });

        b.end_range(r_axis)?;
        b.end_range(r_inner)?;
        b.end_range(r_outer)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(ssa, vec![self.clone()], dims.to_vec(), dtype, self.device())
    }

    pub fn argmax(&self, dim: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            crate::bail!("argmax dim {dim} out of range for shape {dims:?}")
        }
        let dtype = self.dtype();
        let outer: usize = dims[..dim].iter().product();
        let axis_len = dims[dim];
        let inner: usize = dims[dim + 1..].iter().product();
        let mut dst_dims = dims.to_vec();
        dst_dims[dim] = 1;

        let mut b = Block::empty();
        let src = b.push(I::DefineGlobal { index: 0, dtype }).to_varid();
        let dst = b.push(I::DefineGlobal { index: 1, dtype: DType::I32 }).to_varid();

        let r_outer = b.range(0, outer as i32, 1);
        let r_inner = b.range(0, inner as i32, 1);

        let best_val = b.push(I::DefineAcc(crate::Const::min_value(dtype)));
        let best_idx = b.push(I::DefineAcc(crate::Const::I32(0)));

        let r_axis = b.range(0, axis_len as i32, 1);

        let off = b.mul(r_outer.id(), (axis_len * inner) as i32);
        let tmp = b.mul(r_axis.id(), inner as i32);
        let off = b.binary(B::Add, off, tmp, DType::I32);
        let off = b.binary(B::Add, off, r_inner.id(), DType::I32);

        let val = b.push(I::Load { src, offset: off.to_a(), dtype });
        let is_better = b.binary(B::Gt, val, best_val, dtype);
        let new_best_val = b.push(I::Where {
            cond: is_better.to_a(),
            on_true: val.to_a(),
            on_false: best_val.to_a(),
            dtype,
        });
        let new_best_idx = b.push(I::Where {
            cond: is_better.to_a(),
            on_true: r_axis.id().to_a(),
            on_false: best_idx.to_a(),
            dtype: DType::I32,
        });
        b.push(I::Assign { dst: best_val.to_varid(), src: new_best_val.to_a() });
        b.push(I::Assign { dst: best_idx.to_varid(), src: new_best_idx.to_a() });

        b.end_range(r_axis)?;

        let dst_off = b.mul(r_outer.id(), inner as i32);
        let dst_off = b.binary(B::Add, dst_off, r_inner.id(), DType::I32);
        b.push(I::Store { dst, offset: dst_off.to_a(), value: best_idx.to_a(), dtype: DType::I32 });

        b.end_range(r_inner)?;
        b.end_range(r_outer)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(ssa, vec![self.clone()], dst_dims, DType::I32, self.device())
    }

    pub fn topk(&self, dim: usize, k: usize) -> Result<Self> {
        let dims = self.dims();
        if dim >= dims.len() {
            crate::bail!("topk dim {dim} out of range for shape {dims:?}")
        }
        if k > dims[dim] {
            crate::bail!("topk k={k} > axis length {}", dims[dim])
        }
        let sorted = self.reduce(crate::lang::ReduceOp::Max, dim)?;
        if k == 1 {
            return Ok(sorted);
        }
        let mut results = vec![sorted];
        let mut src = self.clone();
        for _ in 1..k {
            let max_val = src.reduce(crate::lang::ReduceOp::Max, dim)?;
            let max_broadcast = max_val.broadcast(src.shape())?;
            let is_max = src.binary(B::Eq, max_broadcast.clone())?;
            let neg_inf = Self::cst(f32::NEG_INFINITY, (), self.device())?;
            let neg_inf = neg_inf.broadcast(src.shape())?;
            src = is_max.where_(neg_inf, src.clone())?;
            let next_max = src.reduce(crate::lang::ReduceOp::Max, dim)?;
            results.push(next_max);
        }
        let refs: Vec<&Self> = results.iter().collect();
        Self::cat(&refs, dim)
    }

    pub fn reduce_dims(&self, op: ReduceOp, dims: &[usize]) -> Result<Self> {
        if dims.is_empty() {
            return Ok(self.clone());
        }
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort();
        sorted_dims.dedup();
        let rank = self.rank();
        for &d in &sorted_dims {
            if d >= rank {
                crate::bail!("reduce_dims: dim {d} out of range for rank {rank}")
            }
        }
        let mut result = self.clone();
        for &d in sorted_dims.iter().rev() {
            result = result.reduce(op, d)?;
        }
        Ok(result)
    }
}
