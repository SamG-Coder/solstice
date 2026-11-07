//! Uploads a [`SceneAsset`] into one shared pool of GPU buffers.
//!
//! WebGPU has no bindless resources. Everything geometric lives in a few big buffers
//! that all passes share, and each instance records where its mesh begins.

use crate::gpu::Gpu;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};
use solstice_asset::{Material, PackedCluster, PackedNode, PackedVertex, SceneAsset};

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
    pub surface_textures: wgpu::TextureView,
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
        let (color_textures, surface_textures) = upload_textures(gpu, asset);

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
            color_textures,
            surface_textures,
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

/// Creates the two texture arrays, or 1x1 stand-ins if the scene has no textures.
fn upload_textures(gpu: &Gpu, asset: &SceneAsset) -> (wgpu::TextureView, wgpu::TextureView) {
    let has_textures = !asset.textures.is_empty();
    let size = if has_textures { asset.texture_size } else { 1 };
    let layers = asset.textures.len().max(1) as u32;
    let mips = asset
        .textures
        .first()
        .map_or(1, |t| t.color_mips.len())
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
    let color = create("material color", wgpu::TextureFormat::Rgba8UnormSrgb);
    let surface = create("material surface", wgpu::TextureFormat::Rgba8Unorm);

    let write = |texture: &wgpu::Texture, layer: u32, mip: u32, data: &[u8]| {
        let edge = (size >> mip).max(1);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(edge * 4),
                rows_per_image: Some(edge),
            },
            wgpu::Extent3d {
                width: edge,
                height: edge,
                depth_or_array_layers: 1,
            },
        );
    };

    if has_textures {
        for (layer, set) in asset.textures.iter().enumerate() {
            for (mip, data) in set.color_mips.iter().enumerate() {
                write(&color, layer as u32, mip as u32, data);
            }
            for (mip, data) in set.surface_mips.iter().enumerate() {
                write(&surface, layer as u32, mip as u32, data);
            }
        }
    } else {
        write(&color, 0, 0, &[255; 4]);
        write(&surface, 0, 0, &[128; 4]);
    }

    let view = |texture: &wgpu::Texture| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    };
    (view(&color), view(&surface))
}
