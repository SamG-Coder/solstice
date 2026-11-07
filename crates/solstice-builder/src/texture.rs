//! Texture preparation: mip chains and block compression.
//!
//! Colour goes to BC7 through the `block_compression` crate, a pure Rust port of Intel's ISPC
//! Texture Compressor. Its slower settings measured no better on scanned textures, so this
//! uses its "basic" one. Normals and material data go to BC5 through [`encode_bc4`] here, as
//! that crate's BC4 stops at the channel's extremes, which measured 1 dB worse on normals.

use block_compression::{encode::compress_rgba8, BC7Settings, CompressionVariant};
use rayon::prelude::*;
use solstice_asset::{SceneAsset, TextureEncoding};

/// Box-filters a square image down to 1x1. The result includes the input as mip 0.
pub fn mip_chain(base: Vec<u8>, size: u32, channels: usize) -> Vec<Vec<u8>> {
    let mut mips = vec![base];
    let mut edge = size as usize;
    while edge > 1 {
        let previous = mips.last().expect("chain starts non-empty");
        let half = edge / 2;
        let mut next = Vec::with_capacity(half * half * channels);
        for y in 0..half {
            for x in 0..half {
                for channel in 0..channels {
                    let at = |px: usize, py: usize| {
                        u32::from(previous[(py * edge + px) * channels + channel])
                    };
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

/// Converts a scene's textures to BC7 and BC5, unless they are already compressed.
/// Textures below 4x4 cannot be, since blocks are 4x4, and those stay as they are.
pub fn compress_textures(scene: &mut SceneAsset) {
    if scene.texture_encoding != TextureEncoding::Raw || scene.texture_size < 4 {
        return;
    }
    let size = scene.texture_size;
    for layer in &mut scene.textures {
        for (level, mip) in layer.color_mips.iter_mut().enumerate() {
            *mip = encode_bc7(mip, size >> level);
        }
        for (level, mip) in layer.normal_mips.iter_mut().enumerate() {
            *mip = encode_bc5(mip, size >> level);
        }
        for (level, mip) in layer.material_mips.iter_mut().enumerate() {
            *mip = encode_bc5(mip, size >> level);
        }
    }
    scene.texture_encoding = TextureEncoding::Block;
}

/// Encodes a square RGBA8 mip level, of any edge, to BC7.
fn encode_bc7(texels: &[u8], edge: u32) -> Vec<u8> {
    let (texels, edge) = fill_blocks(texels, edge, 4);
    let edge = edge as usize;
    let blocks_wide = edge / 4;
    let mut out = vec![0; blocks_wide * blocks_wide * 16];
    let variant = CompressionVariant::BC7(BC7Settings::alpha_basic());
    out.par_chunks_mut(blocks_wide * 16)
        .zip(texels.par_chunks(edge * 4 * 4))
        .for_each(|(row_out, row_in)| {
            let (width, stride) = (edge as u32, (edge * 4) as u32);
            compress_rgba8(variant, row_in, row_out, width, 4, stride);
        });
    out
}

/// Encodes a square RG8 mip level, of any edge, to BC5.
fn encode_bc5(texels: &[u8], edge: u32) -> Vec<u8> {
    let (texels, edge) = fill_blocks(texels, edge, 2);
    let edge = edge as usize;
    let blocks_wide = edge / 4;
    let mut out = vec![0; blocks_wide * blocks_wide * 16];
    out.par_chunks_mut(blocks_wide * 16)
        .zip(texels.par_chunks(edge * 2 * 4))
        .for_each(|(row_out, row_in)| {
            for (x, block_out) in row_out.chunks_exact_mut(16).enumerate() {
                for (channel, half) in block_out.chunks_exact_mut(8).enumerate() {
                    let values = std::array::from_fn(|i| {
                        row_in[((i / 4) * edge + x * 4 + i % 4) * 2 + channel]
                    });
                    half.copy_from_slice(&encode_bc4(&values));
                }
            }
        });
    out
}

/// A mip smaller than one block repeats to fill it, as sampling only reads the part in use.
fn fill_blocks(texels: &[u8], edge: u32, channels: usize) -> (std::borrow::Cow<'_, [u8]>, u32) {
    if edge >= 4 {
        return (texels.into(), edge);
    }
    let n = edge as usize;
    let mut filled = Vec::with_capacity(16 * channels);
    for i in 0..16 {
        let at = ((i / 4) % n * n + (i % 4) % n) * channels;
        filled.extend_from_slice(&texels[at..at + channels]);
    }
    (filled.into(), 4)
}

/// How far inside a block's range to look for better endpoints. Noise-like blocks do best
/// with endpoints inside the range, and searching further than this measured no better.
const BC4_SEARCH: i32 = 8;

/// Encodes 16 values to a BC4 block by testing endpoint pairs near the block's extremes.
fn encode_bc4(values: &[u8; 16]) -> [u8; 8] {
    let min = i32::from(*values.iter().min().expect("16 values"));
    let max = i32::from(*values.iter().max().expect("16 values"));
    if min == max {
        // Equal endpoints select the six-value mode, whose index 0 is the first endpoint.
        return [min as u8, min as u8, 0, 0, 0, 0, 0, 0];
    }

    let mut best = (i64::MAX, min, max);
    for low in min..=(min + BC4_SEARCH).min(max - 1) {
        for high in (max - BC4_SEARCH).max(low + 1)..=max {
            let palette = bc4_palette(low, high);
            let mut error = 0;
            for &value in values {
                let difference = palette
                    .iter()
                    .map(|entry| (entry - i32::from(value)).abs())
                    .min()
                    .expect("8 entries");
                error += i64::from(difference * difference);
                if error >= best.0 {
                    break;
                }
            }
            if error < best.0 {
                best = (error, low, high);
            }
        }
    }

    let (_, low, high) = best;
    let palette = bc4_palette(low, high);
    let mut indices = 0u64;
    for (i, &value) in values.iter().enumerate() {
        let (nearest, _) = palette
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| (*entry - i32::from(value)).abs())
            .expect("8 entries");
        // Position 0 of the palette is the first endpoint, 7 the second, and the rest run
        // between them, which is the order of index values 0, 2, 3, 4, 5, 6, 7, 1.
        let code = match nearest {
            0 => 0,
            7 => 1,
            between => between as u64 + 1,
        };
        indices |= code << (3 * i);
    }
    let mut block = [0; 8];
    block[0] = high as u8;
    block[1] = low as u8;
    block[2..].copy_from_slice(&indices.to_le_bytes()[..6]);
    block
}

/// The eight values a block can hold when its first endpoint is the larger, from `high` to `low`.
fn bc4_palette(low: i32, high: i32) -> [i32; 8] {
    std::array::from_fn(|i| {
        let i = i as i32;
        ((7 - i) * high + i * low + 3) / 7
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use block_compression::decode::{decode_block_bc5, decode_block_bc7};

    #[test]
    fn mip_chain_reaches_one_texel_with_correct_sizes() {
        for channels in [2, 4] {
            let mips = mip_chain(vec![200; 16 * 16 * channels], 16, channels);
            assert_eq!(mips.len(), 5);
            for (level, mip) in mips.iter().enumerate() {
                let edge = 16usize >> level;
                assert_eq!(mip.len(), edge * edge * channels);
            }
            assert_eq!(mips[4], vec![200; channels]);
        }
    }

    /// Texture-like data: a slope with repeating noise on top.
    fn test_value(x: usize, y: usize, seed: usize) -> u8 {
        let noise = ((x * 7 + y * 13 + seed * 5).wrapping_mul(2654435761) >> 7) & 15;
        (60 + (x + y) * 3 + noise + seed * 20).min(255) as u8
    }

    #[test]
    fn bc5_stays_close_to_its_source() {
        let texels: Vec<u8> = (0..16 * 16)
            .flat_map(|i| [test_value(i % 16, i / 16, 0), test_value(i % 16, i / 16, 1)])
            .collect();
        let encoded = encode_bc5(&texels, 16);
        assert_eq!(encoded.len(), 4 * 4 * 16);
        let mut out = [0u8; 32];
        for (index, block) in encoded.chunks_exact(16).enumerate() {
            decode_block_bc5(block, &mut out, 8);
            let (bx, by) = (index % 4 * 4, index / 4 * 4);
            for y in 0..4 {
                for x in 0..8 {
                    let source = texels[((by + y) * 16 + bx) * 2 + x];
                    let error = (i32::from(out[y * 8 + x]) - i32::from(source)).abs();
                    assert!(error <= 4, "block {index}: {error} at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn bc5_represents_flat_blocks_exactly() {
        let encoded = encode_bc5(&[77, 200].repeat(16), 4);
        let mut out = [0u8; 32];
        decode_block_bc5(&encoded, &mut out, 8);
        assert_eq!(out[..], [[77, 200]; 16].concat()[..]);
    }

    #[test]
    fn bc7_stays_close_to_its_source_and_keeps_alpha_separate() {
        let texels: Vec<u8> = (0..16 * 16)
            .flat_map(|i| {
                let (x, y) = (i % 16, i / 16);
                [
                    test_value(x, y, 0),
                    test_value(x, y, 0) / 2,
                    test_value(x, y, 0) / 3,
                    test_value(y, x, 4),
                ]
            })
            .collect();
        let encoded = encode_bc7(&texels, 16);
        let mut out = [0u8; 64];
        for (index, block) in encoded.chunks_exact(16).enumerate() {
            decode_block_bc7(block, &mut out, 16);
            let (bx, by) = (index % 4 * 4, index / 4 * 4);
            for y in 0..4 {
                for x in 0..16 {
                    let source = texels[((by + y) * 16 + bx) * 4 + x];
                    let error = (i32::from(out[y * 16 + x]) - i32::from(source)).abs();
                    assert!(error <= 12, "block {index}: {error} at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn mips_below_a_block_fill_one_whole_block() {
        assert_eq!(encode_bc7(&[9; 4], 1).len(), 16);
        assert_eq!(encode_bc7(&[9; 16], 2).len(), 16);
        assert_eq!(encode_bc5(&[9; 2], 1).len(), 16);
        let two_by_two = [10, 11, 40, 41, 40, 41, 10, 11];
        let mut out = [0u8; 32];
        decode_block_bc5(&encode_bc5(&two_by_two, 2), &mut out, 8);
        assert_eq!(&out[..4], &[10, 11, 40, 41]);
        assert_eq!(&out[8..12], &[40, 41, 10, 11]);
    }

    #[test]
    fn compressing_a_scene_fills_every_level_and_is_idempotent() {
        let chain = |channels| mip_chain(vec![90; 16 * 16 * channels], 16, channels);
        let mut scene = SceneAsset {
            texture_size: 16,
            textures: vec![solstice_asset::TextureLayer {
                color_mips: chain(4),
                normal_mips: chain(2),
                material_mips: chain(2),
            }],
            ..Default::default()
        };
        compress_textures(&mut scene);
        assert_eq!(scene.texture_encoding, TextureEncoding::Block);
        let layer = &scene.textures[0];
        let sizes: Vec<usize> = layer.normal_mips.iter().map(Vec::len).collect();
        assert_eq!(sizes, [256, 64, 16, 16, 16]);
        assert_eq!(layer.color_mips[4].len(), 16);

        let before = layer.color_mips[0].clone();
        compress_textures(&mut scene);
        assert_eq!(scene.textures[0].color_mips[0], before);
    }
}
