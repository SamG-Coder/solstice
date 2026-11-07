//! Uploads a [`SceneAsset`] into one shared pool of GPU buffers.
//!
//! WebGPU has no bindless resources. Everything geometric lives in a few big buffers
//! that all passes share, and each instance records where its mesh begins.

use crate::gpu::Gpu;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use solstice_asset::{
    Material, PackedCluster, PackedNode, PackedVertex, SceneAsset, TextureEncoding,
};

/// Twin of `Instance` in `common/scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuInstance {
    world: [Vec4; 3],
    prev_world: [Vec4; 3],
    normal: [Vec4; 3],
    quant_origin: Vec3,
    quant_step: f32,
    uv_origin: Vec2,
    uv_step: Vec2,
    root_node: u32,
    material: u32,
    max_scale: f32,
    _pad: u32,
}

const _: () = assert!(std::mem::size_of::<GpuInstance>() == 192);

/// Stores the top three rows of an affine matrix.
fn affine_rows(m: Mat4) -> [Vec4; 3] {
    [m.row(0), m.row(1), m.row(2)]
}

/// A scene resident on the GPU.
pub struct GpuScene {
    pub instances: wgpu::Buffer,
    pub nodes: wgpu::Buffer,
    pub clusters: wgpu::Buffer,
    pub vertices: wgpu::Buffer,
    pub triangles: wgpu::Buffer,
    pub materials: wgpu::Buffer,
    pub lights: wgpu::Buffer,
    pub color_textures: wgpu::TextureView,
    pub normal_textures: wgpu::TextureView,
    pub material_textures: wgpu::TextureView,
    pub texture_summary: String,
    pub material_sampler: wgpu::Sampler,
    pub instance_count: u32,
    pub light_count: u32,
    /// Longest root-to-leaf path over all meshes, which sets the traversal dispatch count.
    pub max_hierarchy_depth: u32,
    /// Triangles drawn if every instance used its finest LOD.
    pub source_triangles: u64,
    pub unique_triangles: u64,
}

impl GpuScene {
    pub fn new(gpu: &Gpu, asset: &SceneAsset) -> Self {
        let pool = GeometryPool::build(asset);
        let fallback_material = [Material::default()];
        let materials: &[Material] = if asset.materials.is_empty() {
            &fallback_material
        } else {
            &asset.materials
        };
        let textures = upload_textures(gpu, asset);

        let material_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 8,
            ..Default::default()
        });

        Self {
            instances: gpu.storage_buffer_init("instances", &pool.instances),
            nodes: gpu.storage_buffer_init("nodes", &pool.nodes),
            clusters: gpu.storage_buffer_init("clusters", &pool.clusters),
            vertices: gpu.storage_buffer_init("vertices", &pool.vertices),
            triangles: gpu.storage_buffer_init("triangles", &pool.triangles),
            materials: gpu.storage_buffer_init("materials", materials),
            lights: gpu.storage_buffer_init("lights", &asset.lights),
            color_textures: textures.color,
            normal_textures: textures.normal,
            material_textures: textures.material,
            texture_summary: textures.summary,
            material_sampler,
            instance_count: pool.instances.len() as u32,
            light_count: asset.lights.len() as u32,
            max_hierarchy_depth: pool.max_hierarchy_depth,
            source_triangles: asset.total_source_triangles(),
            unique_triangles: asset.unique_source_triangles(),
        }
    }
}

/// All meshes concatenated, with offsets rebased to the shared arrays.
struct GeometryPool {
    instances: Vec<GpuInstance>,
    nodes: Vec<PackedNode>,
    clusters: Vec<PackedCluster>,
    vertices: Vec<PackedVertex>,
    triangles: Vec<u32>,
    max_hierarchy_depth: u32,
}

impl GeometryPool {
    fn build(asset: &SceneAsset) -> Self {
        let mut pool = Self {
            instances: Vec::with_capacity(asset.instances.len()),
            nodes: Vec::new(),
            clusters: Vec::new(),
            vertices: Vec::new(),
            triangles: Vec::new(),
            max_hierarchy_depth: 1,
        };

        let mut root_nodes = Vec::with_capacity(asset.meshes.len());
        for mesh in &asset.meshes {
            let node_base = pool.nodes.len() as u32;
            let cluster_base = pool.clusters.len() as u32;
            let vertex_base = pool.vertices.len() as u32;
            let triangle_base = pool.triangles.len() as u32;

            pool.nodes.extend(mesh.nodes.iter().map(|node| PackedNode {
                first: node.first
                    + if node.is_leaf != 0 {
                        cluster_base
                    } else {
                        node_base
                    },
                ..*node
            }));
            pool.clusters
                .extend(mesh.clusters.iter().map(|cluster| PackedCluster {
                    vertex_offset: cluster.vertex_offset + vertex_base,
                    triangle_offset: cluster.triangle_offset + triangle_base,
                    ..*cluster
                }));
            pool.vertices.extend_from_slice(&mesh.vertices);
            pool.triangles.extend_from_slice(&mesh.triangles);
            pool.max_hierarchy_depth = pool.max_hierarchy_depth.max(mesh.hierarchy_depth);
            root_nodes.push(mesh.root_node + node_base);
        }

        for instance in &asset.instances {
            let mesh = &asset.meshes[instance.mesh as usize];
            if mesh.clusters.is_empty() {
                continue;
            }
            let world = instance.transform;
            let linear = Mat3::from_mat4(world);
            let normal = Mat4::from_mat3(linear.inverse().transpose());
            let max_scale = linear
                .x_axis
                .length()
                .max(linear.y_axis.length())
                .max(linear.z_axis.length());
            pool.instances.push(GpuInstance {
                world: affine_rows(world),
                prev_world: affine_rows(world),
                normal: affine_rows(normal),
                quant_origin: mesh.quant_origin,
                quant_step: mesh.quant_step,
                uv_origin: Vec2::from_array(mesh.uv_origin),
                uv_step: Vec2::from_array(mesh.uv_step),
                root_node: root_nodes[instance.mesh as usize],
                material: instance.material,
                max_scale,
                _pad: 0,
            });
        }
        pool
    }
}

/// The largest edge kept when textures must be unpacked, because the GPU can't sample BC.
/// Unpacked textures take 8 bytes per texel rather than 3, so this keeps a scene baked at a
/// larger size from taking far more memory on the devices least able to spare it.
const UNPACKED_MAX_EDGE: u32 = 512;

struct UploadedTextures {
    color: wgpu::TextureView,
    normal: wgpu::TextureView,
    material: wgpu::TextureView,
    summary: String,
}

/// Creates the three texture arrays, or 1x1 stand-ins if the scene has no textures.
fn upload_textures(gpu: &Gpu, asset: &SceneAsset) -> UploadedTextures {
    let has_textures = !asset.textures.is_empty();
    // The stand-ins for a scene without textures are never blocks, whatever it says it holds.
    let stored_as_blocks = has_textures && asset.texture_encoding == TextureEncoding::Block;
    let block = stored_as_blocks && gpu.block_compressed_textures;
    let unpack = stored_as_blocks && !gpu.block_compressed_textures;

    // Unpacking can drop a chain's largest levels, so it no longer starts at `texture_size`.
    let mut skip = 0;
    if unpack {
        while asset.texture_size >> skip > UNPACKED_MAX_EDGE {
            skip += 1;
        }
    }
    let size = if has_textures {
        asset.texture_size >> skip
    } else {
        1
    };
    let layers = asset.textures.len().max(1) as u32;
    let mips = asset
        .textures
        .first()
        .map_or(1, |t| t.color_mips.len() - skip)
        .max(1) as u32;

    let create = |label: &str, format: wgpu::TextureFormat| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let (color_format, data_format) = if block {
        (
            wgpu::TextureFormat::Bc7RgbaUnormSrgb,
            wgpu::TextureFormat::Bc5RgUnorm,
        )
    } else {
        (
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Rg8Unorm,
        )
    };
    let color = create("material color", color_format);
    let normal = create("material normal", data_format);
    let material = create("material data", data_format);

    let mut bytes = 0u64;
    let mut write = |texture: &wgpu::Texture, layer: usize, mip: usize, data: &[u8]| {
        bytes += data.len() as u64;
        let edge = (size >> mip).max(1);
        // Copies of block formats cover whole blocks, even when the mip itself is smaller.
        let (copy_edge, rows, bytes_per_row) = if block {
            let blocks = edge.div_ceil(4);
            (blocks * 4, blocks, blocks * 16)
        } else {
            let texel_bytes = data.len() as u32 / (edge * edge);
            (edge, edge, edge * texel_bytes)
        };
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip as u32,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: copy_edge,
                height: copy_edge,
                depth_or_array_layers: 1,
            },
        );
    };

    if has_textures {
        for (layer, set) in asset.textures.iter().enumerate() {
            for (mip, data) in set.color_mips[skip..].iter().enumerate() {
                if unpack {
                    write(&color, layer, mip, &unpack_bc7((size >> mip).max(1), data));
                } else {
                    write(&color, layer, mip, data);
                }
            }
            for (texture, chain) in [(&normal, &set.normal_mips), (&material, &set.material_mips)] {
                for (mip, data) in chain[skip..].iter().enumerate() {
                    if unpack {
                        write(texture, layer, mip, &unpack_bc5((size >> mip).max(1), data));
                    } else {
                        write(texture, layer, mip, data);
                    }
                }
            }
        }
    } else {
        write(&color, 0, 0, &[255; 4]);
        write(&normal, 0, 0, &[128; 2]);
        write(&material, 0, 0, &[128; 2]);
    }

    let view = |texture: &wgpu::Texture| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    };
    let form = if block {
        "BC7 + BC5"
    } else if unpack {
        "RGBA8 + RG8 unpacked from BC"
    } else {
        "RGBA8 + RG8"
    };
    let summary = format!("{layers} x {size}, {form}, {:.0} MB", bytes as f64 / 1.0e6);
    log::info!("textures: {summary}");
    UploadedTextures {
        color: view(&color),
        normal: view(&normal),
        material: view(&material),
        summary,
    }
}

/// Unpacks a BC7 mip level to RGBA8.
fn unpack_bc7(edge: u32, blocks: &[u8]) -> Vec<u8> {
    unpack_blocks::<4>(edge, blocks, block_compression::decode::decode_block_bc7)
}

/// Unpacks a BC5 mip level to RG8.
fn unpack_bc5(edge: u32, blocks: &[u8]) -> Vec<u8> {
    unpack_blocks::<2>(edge, blocks, block_compression::decode::decode_block_bc5)
}

/// Runs a 16-byte block decoder over a whole mip level, cropping the blocks that a mip
/// smaller than 4x4 only partly uses.
fn unpack_blocks<const TEXEL_BYTES: usize>(
    edge: u32,
    blocks: &[u8],
    decode: impl Fn(&[u8], &mut [u8], usize),
) -> Vec<u8> {
    let edge = edge as usize;
    let blocks_wide = edge.div_ceil(4);
    let pitch = edge * TEXEL_BYTES;
    let mut out = vec![0; edge * pitch];
    let mut texels = [0u8; 64];
    for (index, block) in blocks.chunks_exact(16).enumerate() {
        let (x, y) = (index % blocks_wide * 4, index / blocks_wide * 4);
        decode(block, &mut texels, 4 * TEXEL_BYTES);
        let row_bytes = (edge - x).min(4) * TEXEL_BYTES;
        for row in 0..(edge - y).min(4) {
            let at = (y + row) * pitch + x * TEXEL_BYTES;
            let from = row * 4 * TEXEL_BYTES;
            out[at..at + row_bytes].copy_from_slice(&texels[from..from + row_bytes]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A BC5 block whose two channels are constant: both endpoints equal, all indices zero.
    fn flat_bc5_block(red: u8, green: u8) -> [u8; 16] {
        let mut block = [0; 16];
        block[0] = red;
        block[1] = red;
        block[8] = green;
        block[9] = green;
        block
    }

    #[test]
    fn unpacking_places_blocks_and_crops_small_mips() {
        let row = [flat_bc5_block(10, 20), flat_bc5_block(30, 40)].concat();
        let unpacked = unpack_bc5(8, &[row.clone(), row].concat());
        assert_eq!(unpacked.len(), 8 * 8 * 2);
        assert_eq!(&unpacked[..2], &[10, 20]);
        assert_eq!(&unpacked[4 * 2..4 * 2 + 2], &[30, 40]);
        assert_eq!(&unpacked[(7 * 8 + 7) * 2..], &[30, 40]);

        let cropped = unpack_bc5(2, &flat_bc5_block(50, 60));
        assert_eq!(cropped, [50, 60, 50, 60, 50, 60, 50, 60]);
    }
}
