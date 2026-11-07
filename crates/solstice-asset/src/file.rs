//! A flat, versioned binary encoding of [`SceneAsset`].
//!
//! The format is deliberately trivial: little-endian lengths followed by raw
//! `Pod` slices, so loading is one bounds-checked copy per array and the data
//! can go straight to the GPU.

use crate::gpu::{PackedCluster, PackedNode, PackedVertex};
use crate::scene::{
    Instance, Light, Material, SceneAsset, TextureEncoding, TextureLayer, Viewpoint, VirtualMesh,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::fmt;

const MAGIC: [u8; 8] = *b"SOLSTICE";
const VERSION: u32 = 3;

/// Why a scene file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    BadInstanceReference,
    BadTextures,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a Solstice scene file"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported file version {v} (expected {VERSION})")
            }
            Self::Truncated => write!(f, "file is truncated"),
            Self::BadInstanceReference => {
                write!(
                    f,
                    "an instance refers to a mesh or material that does not exist"
                )
            }
            Self::BadTextures => write!(f, "the texture data does not match its stated format"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// On-disk form of an [`Instance`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawInstance {
    mesh: u32,
    material: u32,
    transform: [f32; 16],
}

/// Serialises a scene to bytes.
pub fn encode(scene: &SceneAsset) -> Vec<u8> {
    let mut w = Writer::default();
    w.bytes(&MAGIC);
    w.pod(&VERSION);

    w.pod(&(scene.meshes.len() as u32));
    for mesh in &scene.meshes {
        w.slice(&mesh.clusters);
        w.slice(&mesh.nodes);
        w.slice(&mesh.vertices);
        w.slice(&mesh.triangles);
        w.pod(&mesh.root_node);
        w.pod(&mesh.hierarchy_depth);
        w.pod(&mesh.quant_origin.to_array());
        w.pod(&mesh.quant_step);
        w.pod(&mesh.uv_origin);
        w.pod(&mesh.uv_step);
        w.pod(&mesh.bounds_center.to_array());
        w.pod(&mesh.bounds_radius);
        w.pod(&mesh.source_triangles);
    }

    w.slice(&scene.materials);
    w.slice(&scene.lights);

    let instances: Vec<RawInstance> = scene
        .instances
        .iter()
        .map(|i| RawInstance {
            mesh: i.mesh,
            material: i.material,
            transform: i.transform.to_cols_array(),
        })
        .collect();
    w.slice(&instances);

    w.pod(&scene.texture_size);
    w.pod(&match scene.texture_encoding {
        TextureEncoding::Raw => 0u32,
        TextureEncoding::Block => 1u32,
    });
    w.pod(&(scene.textures.len() as u32));
    for layer in &scene.textures {
        w.pod(&(layer.color_mips.len() as u32));
        for mip in layer
            .color_mips
            .iter()
            .chain(&layer.normal_mips)
            .chain(&layer.material_mips)
        {
            w.slice(mip);
        }
    }

    w.pod(&u32::from(scene.viewpoint.is_some()));
    w.pod(&scene.viewpoint.unwrap_or_default());

    w.finish()
}

/// Parses bytes produced by [`encode`].
pub fn decode(bytes: &[u8]) -> Result<SceneAsset, DecodeError> {
    let mut r = Reader { bytes };
    if r.take(MAGIC.len()).ok() != Some(MAGIC.as_slice()) {
        return Err(DecodeError::BadMagic);
    }
    let version: u32 = r.pod()?;
    if version != VERSION {
        return Err(DecodeError::UnsupportedVersion(version));
    }

    let mesh_count: u32 = r.pod()?;
    let mut meshes = Vec::new();
    for _ in 0..mesh_count {
        let clusters: Vec<PackedCluster> = r.slice()?;
        let nodes: Vec<PackedNode> = r.slice()?;
        let vertices: Vec<PackedVertex> = r.slice()?;
        let triangles: Vec<u32> = r.slice()?;
        meshes.push(VirtualMesh {
            clusters,
            nodes,
            vertices,
            triangles,
            root_node: r.pod()?,
            hierarchy_depth: r.pod()?,
            quant_origin: Vec3::from_array(r.pod()?),
            quant_step: r.pod()?,
            uv_origin: r.pod()?,
            uv_step: r.pod()?,
            bounds_center: Vec3::from_array(r.pod()?),
            bounds_radius: r.pod()?,
            source_triangles: r.pod()?,
        });
    }

    let materials: Vec<Material> = r.slice()?;
    let lights: Vec<Light> = r.slice()?;

    let raw_instances: Vec<RawInstance> = r.slice()?;
    let instances: Vec<Instance> = raw_instances
        .iter()
        .map(|raw| Instance {
            mesh: raw.mesh,
            material: raw.material,
            transform: Mat4::from_cols_array(&raw.transform),
        })
        .collect();
    let references_valid = instances
        .iter()
        .all(|i| (i.mesh as usize) < meshes.len() && (i.material as usize) < materials.len());
    if !references_valid {
        return Err(DecodeError::BadInstanceReference);
    }

    let texture_size: u32 = r.pod()?;
    let texture_encoding = match r.pod::<u32>()? {
        0 => TextureEncoding::Raw,
        1 => TextureEncoding::Block,
        _ => return Err(DecodeError::BadTextures),
    };
    let layer_count: u32 = r.pod()?;
    let mut textures = Vec::new();
    for _ in 0..layer_count {
        let mip_count: u32 = r.pod()?;
        let mut layer = TextureLayer::default();
        for _ in 0..mip_count {
            layer.color_mips.push(r.slice()?);
        }
        for _ in 0..mip_count {
            layer.normal_mips.push(r.slice()?);
        }
        for _ in 0..mip_count {
            layer.material_mips.push(r.slice()?);
        }
        if !layer_is_well_formed(&layer, texture_size, texture_encoding) {
            return Err(DecodeError::BadTextures);
        }
        textures.push(layer);
    }

    let has_viewpoint: u32 = r.pod()?;
    let viewpoint: Viewpoint = r.pod()?;

    Ok(SceneAsset {
        meshes,
        materials,
        instances,
        lights,
        texture_size,
        texture_encoding,
        textures,
        viewpoint: (has_viewpoint != 0).then_some(viewpoint),
    })
}

/// Checks that every mip is the size the GPU will expect, which a bad file could otherwise
/// turn into a validation error much later.
fn layer_is_well_formed(layer: &TextureLayer, size: u32, encoding: TextureEncoding) -> bool {
    if !size.is_power_of_two() || (encoding == TextureEncoding::Block && size < 4) {
        return false;
    }
    let levels = size.ilog2() as usize + 1;
    let edge = |level: usize| size >> level;
    layer.color_mips.len() == levels
        && (0..levels).all(|level| {
            layer.color_mips[level].len() == encoding.color_mip_bytes(edge(level))
                && layer.normal_mips[level].len() == encoding.data_mip_bytes(edge(level))
                && layer.material_mips[level].len() == encoding.data_mip_bytes(edge(level))
        })
}

#[derive(Default)]
struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn bytes(&mut self, data: &[u8]) {
        self.out.extend_from_slice(data);
    }

    fn pod<T: Pod>(&mut self, value: &T) {
        self.bytes(bytemuck::bytes_of(value));
    }

    fn slice<T: Pod>(&mut self, values: &[T]) {
        self.pod(&(values.len() as u64));
        self.bytes(bytemuck::cast_slice(values));
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], DecodeError> {
        if count > self.bytes.len() {
            return Err(DecodeError::Truncated);
        }
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }

    fn pod<T: Pod>(&mut self) -> Result<T, DecodeError> {
        let raw = self.take(std::mem::size_of::<T>())?;
        Ok(bytemuck::pod_read_unaligned(raw))
    }

    fn slice<T: Pod>(&mut self) -> Result<Vec<T>, DecodeError> {
        let len: u64 = self.pod()?;
        let byte_len = usize::try_from(len)
            .ok()
            .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
            .ok_or(DecodeError::Truncated)?;
        let raw = self.take(byte_len)?;
        Ok(raw
            .chunks_exact(std::mem::size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_scene() -> SceneAsset {
        SceneAsset {
            meshes: vec![VirtualMesh {
                clusters: vec![PackedCluster {
                    radius: 2.0,
                    counts: 3 | 1 << 8,
                    ..Default::default()
                }],
                nodes: vec![PackedNode::default()],
                vertices: vec![PackedVertex { data: [7; 8] }; 3],
                triangles: vec![crate::gpu::pack_triangle(0, 1, 2)],
                quant_step: 0.5,
                source_triangles: 1,
                ..Default::default()
            }],
            materials: vec![Material::default()],
            lights: vec![Light {
                range: 4.0,
                ..Default::default()
            }],
            instances: vec![Instance {
                mesh: 0,
                material: 0,
                transform: Mat4::from_translation(Vec3::X),
            }],
            texture_size: 1,
            texture_encoding: TextureEncoding::Raw,
            textures: vec![TextureLayer {
                color_mips: vec![vec![1, 2, 3, 4]],
                normal_mips: vec![vec![5, 6]],
                material_mips: vec![vec![7, 8]],
            }],
            viewpoint: Some(Viewpoint {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.25,
            }),
        }
    }

    #[test]
    fn round_trip_preserves_everything() {
        let scene = sample_scene();
        let decoded = decode(&encode(&scene)).unwrap();
        assert_eq!(decoded.meshes[0].clusters, scene.meshes[0].clusters);
        assert_eq!(decoded.meshes[0].vertices, scene.meshes[0].vertices);
        assert_eq!(decoded.meshes[0].triangles, scene.meshes[0].triangles);
        assert_eq!(decoded.materials, scene.materials);
        assert_eq!(decoded.lights, scene.lights);
        assert_eq!(decoded.viewpoint, scene.viewpoint);
        assert_eq!(decoded.instances[0].transform, scene.instances[0].transform);
        assert_eq!(decoded.textures[0].normal_mips, vec![vec![5, 6]]);
        assert_eq!(decoded.textures[0].material_mips, vec![vec![7, 8]]);
    }

    #[test]
    fn rejects_foreign_and_truncated_data() {
        assert_eq!(decode(b"not a scene").unwrap_err(), DecodeError::BadMagic);
        let bytes = encode(&sample_scene());
        assert_eq!(
            decode(&bytes[..bytes.len() - 1]).unwrap_err(),
            DecodeError::Truncated
        );
    }

    #[test]
    fn block_encoding_round_trips_with_small_mips_filling_a_block() {
        let mut scene = sample_scene();
        scene.texture_size = 4;
        scene.texture_encoding = TextureEncoding::Block;
        let chain = vec![vec![9; 16]; 3];
        scene.textures = vec![TextureLayer {
            color_mips: chain.clone(),
            normal_mips: chain.clone(),
            material_mips: chain,
        }];
        let decoded = decode(&encode(&scene)).unwrap();
        assert_eq!(decoded.texture_encoding, TextureEncoding::Block);
        assert_eq!(decoded.textures[0].color_mips[2], vec![9; 16]);
    }

    #[test]
    fn rejects_textures_of_the_wrong_size() {
        let mut scene = sample_scene();
        scene.textures[0].normal_mips[0].pop();
        assert_eq!(
            decode(&encode(&scene)).unwrap_err(),
            DecodeError::BadTextures
        );
    }

    #[test]
    fn rejects_dangling_instance_references() {
        let mut scene = sample_scene();
        scene.instances[0].mesh = 9;
        assert_eq!(
            decode(&encode(&scene)).unwrap_err(),
            DecodeError::BadInstanceReference
        );
    }
}
