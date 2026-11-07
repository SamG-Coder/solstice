//! Procedural, seamlessly tiling material textures.

use crate::noise::tiling_fbm;
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use solstice_asset::TextureLayer;
use solstice_builder::mip_chain;

/// Description of one surface at a point on the texture.
struct Texel {
    color: Vec3,
    height: f32,
    roughness: f32,
    cavity: f32,
}

/// Generates a texture layer of `size`x`size` with a full mip chain.
fn build_layer(
    size: u32,
    normal_strength: f32,
    sample: impl Fn(f32, f32) -> Texel + Sync,
) -> TextureLayer {
    let n = size as usize;
    let texels: Vec<Texel> = (0..n * n)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % n, i / n);
            sample(
                (x as f32 + 0.5) / size as f32,
                (y as f32 + 0.5) / size as f32,
            )
        })
        .collect();

    let mut color = Vec::with_capacity(n * n * 4);
    let mut normals = Vec::with_capacity(n * n * 2);
    let mut material = Vec::with_capacity(n * n * 2);
    let to_byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    for y in 0..n {
        for x in 0..n {
            let texel = &texels[y * n + x];
            let height_at = |dx: i32, dy: i32| {
                let px = (x as i32 + dx).rem_euclid(n as i32) as usize;
                let py = (y as i32 + dy).rem_euclid(n as i32) as usize;
                texels[py * n + px].height
            };
            let slope = Vec2::new(
                height_at(1, 0) - height_at(-1, 0),
                height_at(0, 1) - height_at(0, -1),
            ) * normal_strength
                * size as f32;
            let normal = Vec3::new(-slope.x, -slope.y, 1.0).normalize();

            let srgb = texel.color.powf(1.0 / 2.2);
            color.extend_from_slice(&[
                to_byte(srgb.x),
                to_byte(srgb.y),
                to_byte(srgb.z),
                to_byte(texel.cavity),
            ]);
            normals
                .extend_from_slice(&[to_byte(normal.x * 0.5 + 0.5), to_byte(normal.y * 0.5 + 0.5)]);
            material.extend_from_slice(&[
                to_byte(texel.roughness * 0.5),
                255, // Metallic multiplier: these materials set metalness by factor alone.
            ]);
        }
    }

    TextureLayer {
        color_mips: mip_chain(color, size, 4),
        normal_mips: mip_chain(normals, size, 2),
        material_mips: mip_chain(material, size, 2),
    }
}

fn mix(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a.lerp(b, t.clamp(0.0, 1.0))
}

/// Warm, sedimentary stone with horizontal strata.
pub fn sandstone(size: u32) -> TextureLayer {
    build_layer(size, 0.16, |u, v| {
        let broad = tiling_fbm(u, v, 3, 5, 10);
        let strata = tiling_fbm(u * 0.25, v, 8, 4, 20) * 0.5 + 0.5;
        let grain = tiling_fbm(u, v, 48, 3, 30);
        let pitting = (tiling_fbm(u, v, 18, 4, 40) - 0.42).max(0.0) * 1.6;
        let tone = broad * 0.5 + 0.5;
        let color = mix(
            Vec3::new(0.46, 0.33, 0.21),
            Vec3::new(0.71, 0.56, 0.39),
            tone,
        );
        let color = mix(color, Vec3::new(0.60, 0.43, 0.27), strata * 0.5) * (0.93 + grain * 0.1);
        Texel {
            color: color * (1.0 - pitting * 0.5),
            height: 0.5 + broad * 0.12 + grain * 0.1 - pitting * 0.35 + strata * 0.06,
            roughness: 0.9 - grain * 0.08,
            cavity: 1.0 - pitting * 0.8,
        }
    })
}

/// Pale stone with dark veins; polishes to a soft sheen.
pub fn marble(size: u32) -> TextureLayer {
    build_layer(size, 0.035, |u, v| {
        let warp = tiling_fbm(u, v, 4, 5, 50);
        let veins = 1.0
            - (tiling_fbm(u + warp * 0.08, v - warp * 0.08, 6, 6, 60) * 3.5)
                .abs()
                .min(1.0);
        let fine = 1.0 - (tiling_fbm(u, v, 16, 4, 70) * 6.0).abs().min(1.0);
        let vein = veins.powi(7) * 0.75 + fine.powi(9) * 0.25;
        let cloud = tiling_fbm(u, v, 3, 4, 80) * 0.5 + 0.5;
        let color = mix(
            Vec3::new(0.80, 0.77, 0.71),
            Vec3::new(0.90, 0.88, 0.84),
            cloud,
        );
        Texel {
            color: mix(color, Vec3::new(0.52, 0.48, 0.44), vein),
            height: 0.5 - vein * 0.05,
            roughness: 0.35 + vein * 0.25 + cloud * 0.08,
            cavity: 1.0 - vein * 0.25,
        }
    })
}

/// Wind-blown desert ground with small stones.
pub fn ground(size: u32) -> TextureLayer {
    build_layer(size, 0.2, |u, v| {
        let dunes = tiling_fbm(u, v, 4, 4, 90);
        let pebbles = (tiling_fbm(u, v, 28, 3, 100) - 0.4).max(0.0) * 3.0;
        let dust = tiling_fbm(u, v, 80, 3, 110);
        let base = mix(
            Vec3::new(0.50, 0.37, 0.25),
            Vec3::new(0.68, 0.53, 0.38),
            dunes * 0.5 + 0.5,
        );
        let color =
            mix(base, Vec3::new(0.34, 0.27, 0.21), pebbles.min(1.0) * 0.3) * (0.95 + dust * 0.1);
        Texel {
            color,
            height: 0.45 + dunes * 0.1 + pebbles.min(1.0) * 0.3 + dust * 0.05,
            roughness: 0.95,
            cavity: 1.0 - (0.28 - pebbles).max(0.0) * 0.6,
        }
    })
}

/// Dark, weathered rock for boulders and cliffs.
pub fn basalt(size: u32) -> TextureLayer {
    build_layer(size, 0.22, |u, v| {
        let broad = tiling_fbm(u, v, 5, 6, 120);
        let cracks = 1.0 - (tiling_fbm(u, v, 9, 5, 130) * 5.0).abs().min(1.0);
        let grain = tiling_fbm(u, v, 56, 3, 140);
        let color = mix(
            Vec3::new(0.21, 0.17, 0.14),
            Vec3::new(0.46, 0.36, 0.27),
            broad * 0.5 + 0.5,
        );
        let crack = cracks.powi(5);
        Texel {
            color: color * (1.0 - crack * 0.6) * (0.92 + grain * 0.14),
            height: 0.5 + broad * 0.2 + grain * 0.06 - crack * 0.3,
            roughness: 0.82 + grain * 0.1,
            cavity: 1.0 - crack * 0.85,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_has_matching_chains() {
        let layer = sandstone(16);
        assert_eq!(layer.color_mips.len(), layer.normal_mips.len());
        assert_eq!(layer.color_mips.len(), layer.material_mips.len());
        assert_eq!(layer.color_mips[0].len(), 16 * 16 * 4);
    }
}
