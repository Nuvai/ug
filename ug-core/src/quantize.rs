use crate::block::Block;
use crate::lang::ssa::Instr as I;
use crate::lang::{BinaryOp as B, DType, UnaryOp};
use crate::{Device, LazyBuffer, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantType {
    Q8_0,
    Q4_0,
}

impl QuantType {
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q8_0 => 32,
            Self::Q4_0 => 32,
        }
    }

    pub fn bits(&self) -> usize {
        match self {
            Self::Q8_0 => 8,
            Self::Q4_0 => 4,
        }
    }
}

impl<D: Device> LazyBuffer<D> {
    pub fn dequantize_q8(
        packed: &Self,
        scales: &Self,
        num_elements: usize,
        group_size: usize,
    ) -> Result<Self> {
        let _n_groups = num_elements / group_size;
        let dtype = DType::F32;

        let mut b = Block::empty();
        let src = b.push(I::DefineGlobal { index: 0, dtype: DType::I32 }).to_varid();
        let scale_buf = b.push(I::DefineGlobal { index: 1, dtype: DType::F32 }).to_varid();
        let dst = b.push(I::DefineGlobal { index: 2, dtype }).to_varid();

        let r = b.range(0, num_elements as i32, 1);

        let group_cst = b.cst(group_size as i32);
        let group_id = b.binary(B::Div, r.id(), group_cst, DType::I32);
        let scale = b.push(I::Load { src: scale_buf, offset: group_id.to_a(), dtype: DType::F32 });

        let packed_val = b.push(I::Load { src, offset: r.id().to_a(), dtype: DType::I32 });
        let as_float = b.unary(UnaryOp::Cast(DType::F32), packed_val, DType::F32);
        let result = b.binary(B::Mul, as_float, scale, DType::F32);

        b.push(I::Store { dst, offset: r.id().to_a(), value: result.to_a(), dtype });
        b.end_range(r)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(ssa, vec![packed.clone(), scales.clone()], num_elements, dtype, packed.device())
    }

    pub fn dequantize_q4(
        packed: &Self,
        scales: &Self,
        num_elements: usize,
        group_size: usize,
    ) -> Result<Self> {
        let n_packed = num_elements / 2;
        let dtype = DType::F32;

        let mut b = Block::empty();
        let src = b.push(I::DefineGlobal { index: 0, dtype: DType::I32 }).to_varid();
        let scale_buf = b.push(I::DefineGlobal { index: 1, dtype: DType::F32 }).to_varid();
        let dst = b.push(I::DefineGlobal { index: 2, dtype }).to_varid();

        let r = b.range(0, n_packed as i32, 1);

        let group_cst = b.cst((group_size / 2) as i32);
        let group_id = b.binary(B::Div, r.id(), group_cst, DType::I32);
        let scale = b.push(I::Load { src: scale_buf, offset: group_id.to_a(), dtype: DType::F32 });

        let packed_val = b.push(I::Load { src, offset: r.id().to_a(), dtype: DType::I32 });

        // Low nibble: packed & 0xF
        let mask_f = b.cst(0xFi32);
        let lo = b.binary(B::Mod, packed_val, mask_f, DType::I32);
        let eight = b.cst(8i32);
        let lo_centered = b.binary(B::Sub, lo, eight, DType::I32);
        let lo_f = b.unary(UnaryOp::Cast(DType::F32), lo_centered, DType::F32);
        let lo_result = b.binary(B::Mul, lo_f, scale, DType::F32);

        // High nibble: (packed >> 4) & 0xF
        let sixteen = b.cst(16i32);
        let hi_raw = b.binary(B::Div, packed_val, sixteen, DType::I32);
        let hi = b.binary(B::Mod, hi_raw, mask_f, DType::I32);
        let hi_centered = b.binary(B::Sub, hi, eight, DType::I32);
        let hi_f = b.unary(UnaryOp::Cast(DType::F32), hi_centered, DType::F32);
        let hi_result = b.binary(B::Mul, hi_f, scale, DType::F32);

        let two = b.cst(2i32);
        let dst_lo = b.binary(B::Mul, r.id(), two, DType::I32);
        let one = b.cst(1i32);
        let dst_hi = b.binary(B::Add, dst_lo, one, DType::I32);

        b.push(I::Store { dst, offset: dst_lo.to_a(), value: lo_result.to_a(), dtype });
        b.push(I::Store { dst, offset: dst_hi.to_a(), value: hi_result.to_a(), dtype });

        b.end_range(r)?;

        let instrs = b.relocate()?;
        let ssa = crate::lang::ssa::Kernel::from_instrs(instrs)?;
        Self::ssa(ssa, vec![packed.clone(), scales.clone()], num_elements, dtype, packed.device())
    }

    pub fn quantize_to_q8(&self, group_size: usize) -> Result<(Self, Self)> {
        let num_elements = self.shape().num_elements();
        if num_elements % group_size != 0 {
            crate::bail!("quantize_q8: num_elements {} not divisible by group_size {}", num_elements, group_size)
        }
        let n_groups = num_elements / group_size;

        let reshaped = self.reshape((n_groups, group_size))?;
        let abs_vals = reshaped.unary(UnaryOp::Neg)?.binary(B::Max, reshaped.clone())?;
        let scales = abs_vals.reduce(crate::lang::ReduceOp::Max, 1)?;
        let inv_scale = {
            let one = Self::cst(127.0f32, (), self.device())?;
            let one = one.broadcast(scales.shape())?;
            one.binary(B::Div, scales.clone())?
        };
        let inv_broadcast = inv_scale.broadcast(reshaped.shape())?;
        let quantized_f = reshaped.binary(B::Mul, inv_broadcast)?;
        let quantized = quantized_f.unary(UnaryOp::Cast(DType::I32))?;
        let quantized = quantized.reshape(num_elements)?;

        Ok((quantized, scales.reshape(n_groups)?))
    }
}
