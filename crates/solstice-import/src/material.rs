//! Converts glTF materials and images to the renderer's texture format.
//!
//! The renderer's layers hold, per texel:
//! * colour: RGB base colour (sRGB), A cavity, from the occlusion map.
//! * normal: RG tangent-space normal.
//! * material: R roughness multiplier, G metallic multiplier.

use glam::Vec2;
use gltf::image::{Data as ImageData, Format};
use gltf::Document;
use rayon::prelude::*;
use solstice_asset::{Material, TextureLayer};
use solstice_builder::mip_chain;
use std::collections::HashMap;

/// A `KHR_texture_transform`, applied to UVs while reading geometry.
///
/// The renderer has one UV set per mesh, so all of a material's maps must share a transform.
/// glTF allows them to differ, which [`MaterialTable`] warns about.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvTransform {
    pub offset: Vec2,
    pub rotation: f32,
    pub scale: Vec2,
}

impl UvTransform {
    pub const IDENTITY: Self = Self {
        offset: Vec2::ZERO,
        rotation: 0.0,
        scale: Vec2::ONE,
    };

    /// The extension's definition: scale, then rotate clockwise in texture space, then offset.
    pub fn apply(&self, uv: Vec2) -> Vec2 {
        let (sin, cos) = self.rotation.sin_cos();
        let scaled = uv * self.scale;
        Vec2::new(
            self.offset.x + cos * scaled.x + sin * scaled.y,
            self.offset.y - sin * scaled.x + cos * scaled.y,
        )
    }
}

fn to_uv_transform(transform: Option<gltf::texture::TextureTransform<'_>>) -> UvTransform {
    transform.map_or(UvTransform::IDENTITY, |t| UvTransform {
        offset: Vec2::from(t.offset()),
        rotation: t.rotation(),
        scale: Vec2::from(t.scale()),
    })
}

/// The transforms of a material's maps. Only colour and metallic-roughness can report one
/// through the glTF crate, so normal and occlusion maps are assumed to match them.
fn map_transforms(material: &gltf::Material<'_>) -> Vec<UvTransform> {
    let pbr = material.pbr_metallic_roughness();
    let mut transforms = Vec::new();
    if let Some(t) = pbr.base_color_texture() {
        transforms.push(to_uv_transform(t.texture_transform()));
    }
    if let Some(t) = pbr.metallic_roughness_texture() {
        transforms.push(to_uv_transform(t.texture_transform()));
    }
    transforms
}

/// The transform to apply to a primitive's UVs.
pub fn uv_transform(material: &gltf::Material<'_>) -> UvTransform {
    map_transforms(material)
        .first()
        .copied()
        .unwrap_or(UvTransform::IDENTITY)
}

/// Which source images make up one texture layer.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LayerKey {
    base_color: Option<usize>,
    normal: Option<usize>,
    metallic_roughness: Option<usize>,
    occlusion: Option<usize>,
}

/// Collects materials and texture layers, sharing them where the source does.
pub struct MaterialTable<'a> {
    images: &'a [ImageData],
    texture_size: u32,
    materials: Vec<Material>,
    slots: HashMap<Option<usize>, u32>,
    layer_slots: HashMap<LayerKey, i32>,
    layers: Vec<TextureLayer>,
}

impl<'a> MaterialTable<'a> {
    pub fn new(images: &'a [ImageData], texture_size: u32) -> Self {
        Self {
            images,
            texture_size,
            materials: Vec::new(),
            slots: HashMap::new(),
            layer_slots: HashMap::new(),
            layers: Vec::new(),
        }
    }

    /// Returns the slot for a glTF material index, where `None` is glTF's default material.
    pub fn slot(
        &mut self,
        document: &Document,
        index: Option<usize>,
        warnings: &mut Vec<String>,
    ) -> u32 {
        if let Some(&slot) = self.slots.get(&index) {
            return slot;
        }
        let material = match index.and_then(|i| document.materials().nth(i)) {
            Some(source) => self.convert(&source, warnings),
            None => Material {
                base_color: [1.0; 4],
                roughness: 1.0,
                metallic: 1.0,
                ..Default::default()
            },
        };
        let slot = self.materials.len() as u32;
        self.materials.push(material);
        self.slots.insert(index, slot);
        slot
    }

    pub fn finish(self) -> (Vec<Material>, Vec<TextureLayer>) {
        (self.materials, self.layers)
    }

    fn convert(&mut self, source: &gltf::Material<'_>, warnings: &mut Vec<String>) -> Material {
        let name = source.name().unwrap_or("unnamed");
        match source.alpha_mode() {
            gltf::material::AlphaMode::Opaque => {}
            mode => warnings.push(format!(
                "material '{name}': alpha mode {mode:?} is drawn as opaque"
            )),
        }
        if source.emissive_texture().is_some() {
            warnings.push(format!(
                "material '{name}': emissive texture is ignored, only its factor is used"
            ));
        }

        let pbr = source.pbr_metallic_roughness();
        let mut uses_other_uv_set = false;
        let mut image_of = |set: u32, index: usize| {
            uses_other_uv_set |= set != 0;
            index
        };
        let key = LayerKey {
            base_color: pbr
                .base_color_texture()
                .map(|t| image_of(t.tex_coord(), t.texture().source().index())),
            normal: source
                .normal_texture()
                .map(|t| image_of(t.tex_coord(), t.texture().source().index())),
            metallic_roughness: pbr
                .metallic_roughness_texture()
                .map(|t| image_of(t.tex_coord(), t.texture().source().index())),
            occlusion: source
                .occlusion_texture()
                .map(|t| image_of(t.tex_coord(), t.texture().source().index())),
        };
        let transforms = map_transforms(source);
        if transforms.windows(2).any(|pair| pair[0] != pair[1]) {
            warnings.push(format!(
                "material '{name}': its maps use different texture transforms, only the first is applied"
            ));
        }
        if uses_other_uv_set {
            warnings.push(format!(
                "material '{name}': only the first UV set is supported"
            ));
        }

        let has_textures = key.base_color.is_some()
            || key.normal.is_some()
            || key.metallic_roughness.is_some()
            || key.occlusion.is_some();
        let texture_layer = if has_textures {
            self.layer_slot(key)
        } else {
            -1
        };

        let [er, eg, eb] = source.emissive_factor();
        Material {
            base_color: pbr.base_color_factor(),
            emissive: [er, eg, eb],
            flags: if source.double_sided() {
                Material::TWO_SIDED
            } else {
                0
            },
            roughness: pbr.roughness_factor(),
            metallic: pbr.metallic_factor(),
            uv_scale: 1.0,
            normal_strength: source.normal_texture().map_or(0.0, |t| t.scale()),
            texture_layer,
            ..Default::default()
        }
    }

    fn layer_slot(&mut self, key: LayerKey) -> i32 {
        if let Some(&slot) = self.layer_slots.get(&key) {
            return slot;
        }
        let layer = compose_layer(self.images, key, self.texture_size);
        let slot = self.layers.len() as i32;
        self.layers.push(layer);
        self.layer_slots.insert(key, slot);
        slot
    }
}

fn compose_layer(images: &[ImageData], key: LayerKey, size: u32) -> TextureLayer {
    let image = |index: Option<usize>| index.map(|i| &images[i]);
    let (base, normal, metallic_roughness, occlusion) = (
        image(key.base_color),
        image(key.normal),
        image(key.metallic_roughness),
        image(key.occlusion),
    );
    let n = size as usize;
    let texels: Vec<([u8; 4], [u8; 2], [u8; 2])> = (0..n * n)
        .into_par_iter()
        .map(|i| {
            let (u, v) = (
                ((i % n) as f32 + 0.5) / size as f32,
                ((i / n) as f32 + 0.5) / size as f32,
            );
            let base = base.map_or([255; 4], |im| sample(im, u, v, size));
            let cavity = occlusion.map_or(255, |im| sample(im, u, v, size)[0]);
            let normal = normal.map_or([128, 128, 255, 255], |im| sample(im, u, v, size));
            // glTF stores roughness in G and metallic in B. Roughness is halved because the
            // layer format lets a texture raise roughness as well as lower it.
            let orm = metallic_roughness.map_or([255; 4], |im| sample(im, u, v, size));
            (
                [base[0], base[1], base[2], cavity],
                // glTF's green points towards decreasing V, and the renderer's derives its
                // frame from the direction of increasing V.
                [normal[0], 255 - normal[1]],
                [orm[1].div_ceil(2), orm[2]],
            )
        })
        .collect();

    let mut color = Vec::with_capacity(n * n * 4);
    let mut normals = Vec::with_capacity(n * n * 2);
    let mut material = Vec::with_capacity(n * n * 2);
    for (c, nor, mat) in &texels {
        color.extend_from_slice(c);
        normals.extend_from_slice(nor);
        material.extend_from_slice(mat);
    }
    TextureLayer {
        color_mips: mip_chain(color, size, 4),
        normal_mips: mip_chain(normals, size, 2),
        material_mips: mip_chain(material, size, 2),
    }
}

/// Reads a source image at output resolution `size`, averaging when shrinking so that a
/// large image does not alias. Wraps at the borders, as glTF's default sampler does.
pub(crate) fn sample(image: &ImageData, u: f32, v: f32, size: u32) -> [u8; 4] {
    let taps = (image.width.max(image.height) / size).clamp(1, 8);
    let mut sum = [0.0f32; 4];
    for ty in 0..taps {
        for tx in 0..taps {
            let du = (tx as f32 + 0.5) / taps as f32 - 0.5;
            let dv = (ty as f32 + 0.5) / taps as f32 - 0.5;
            let texel = bilinear(image, u + du / size as f32, v + dv / size as f32);
            for channel in 0..4 {
                sum[channel] += texel[channel];
            }
        }
    }
    let count = (taps * taps) as f32;
    sum.map(|value| (value / count + 0.5) as u8)
}

fn bilinear(image: &ImageData, u: f32, v: f32) -> [f32; 4] {
    let x = u * image.width as f32 - 0.5;
    let y = v * image.height as f32 - 0.5;
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |dx: i64, dy: i64| texel(image, x0 as i64 + dx, y0 as i64 + dy);
    let (a, b, c, d) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
    std::array::from_fn(|channel| {
        let top = a[channel] + (b[channel] - a[channel]) * fx;
        let bottom = c[channel] + (d[channel] - c[channel]) * fx;
        top + (bottom - top) * fy
    })
}

/// One texel as RGBA in `0.0..=255.0`, whatever the source format.
fn texel(image: &ImageData, x: i64, y: i64) -> [f32; 4] {
    let x = x.rem_euclid(i64::from(image.width)) as usize;
    let y = y.rem_euclid(i64::from(image.height)) as usize;
    let index = y * image.width as usize + x;
    let p = &image.pixels;
    let u8_at = |offset: usize| f32::from(p[offset]);
    // 16-bit data is little-endian, so its high byte holds the most significant bits.
    let u16_at = |offset: usize| f32::from(p[offset * 2 + 1]);
    let f32_at = |offset: usize| {
        let bytes = [
            p[offset * 4],
            p[offset * 4 + 1],
            p[offset * 4 + 2],
            p[offset * 4 + 3],
        ];
        f32::from_le_bytes(bytes).clamp(0.0, 1.0) * 255.0
    };
    match image.format {
        Format::R8 => [u8_at(index), u8_at(index), u8_at(index), 255.0],
        Format::R8G8 => [u8_at(index * 2), u8_at(index * 2 + 1), 0.0, 255.0],
        Format::R8G8B8 => [
            u8_at(index * 3),
            u8_at(index * 3 + 1),
            u8_at(index * 3 + 2),
            255.0,
        ],
        Format::R8G8B8A8 => [
            u8_at(index * 4),
            u8_at(index * 4 + 1),
            u8_at(index * 4 + 2),
            u8_at(index * 4 + 3),
        ],
        Format::R16 => [u16_at(index), u16_at(index), u16_at(index), 255.0],
        Format::R16G16 => [u16_at(index * 2), u16_at(index * 2 + 1), 0.0, 255.0],
        Format::R16G16B16 => [
            u16_at(index * 3),
            u16_at(index * 3 + 1),
            u16_at(index * 3 + 2),
            255.0,
        ],
        Format::R16G16B16A16 => [
            u16_at(index * 4),
            u16_at(index * 4 + 1),
            u16_at(index * 4 + 2),
            u16_at(index * 4 + 3),
        ],
        Format::R32G32B32FLOAT => [
            f32_at(index * 3),
            f32_at(index * 3 + 1),
            f32_at(index * 3 + 2),
            255.0,
        ],
        Format::R32G32B32A32FLOAT => [
            f32_at(index * 4),
            f32_at(index * 4 + 1),
            f32_at(index * 4 + 2),
            f32_at(index * 4 + 3),
        ],
    }
}
