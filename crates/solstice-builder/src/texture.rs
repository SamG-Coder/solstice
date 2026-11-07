//! Texture preparation.

/// Box-filters a square RGBA8 image down to 1x1. The result includes the input as mip 0.
pub fn mip_chain(base: Vec<u8>, size: u32) -> Vec<Vec<u8>> {
    let mut mips = vec![base];
    let mut edge = size as usize;
    while edge > 1 {
        let previous = mips.last().expect("chain starts non-empty");
        let half = edge / 2;
        let mut next = Vec::with_capacity(half * half * 4);
        for y in 0..half {
            for x in 0..half {
                for channel in 0..4 {
                    let at =
                        |px: usize, py: usize| u32::from(previous[(py * edge + px) * 4 + channel]);
                    let sum = at(x * 2, y * 2)
                        + at(x * 2 + 1, y * 2)
                        + at(x * 2, y * 2 + 1)
                        + at(x * 2 + 1, y * 2 + 1);
                    next.push((sum / 4) as u8);
                }
            }
        }
        mips.push(next);
        edge = half;
    }
    mips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_reaches_one_texel_with_correct_sizes() {
        let mips = mip_chain(vec![200; 16 * 16 * 4], 16);
        assert_eq!(mips.len(), 5);
        for (level, mip) in mips.iter().enumerate() {
            let edge = 16usize >> level;
            assert_eq!(mip.len(), edge * edge * 4);
        }
        assert_eq!(mips[4], vec![200; 4]);
    }
}
