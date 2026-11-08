//! Converts the visibility buffer into a G-buffer.

use super::culling::CullWork;
use super::raster::VisibilityBuffer;
use crate::gpu::{groups_for, Gpu, LayoutBuilder, Target};
use crate::scene::GpuScene;
use crate::shaders;

const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const EMISSIVE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Surface properties per pixel.
pub struct GBuffer {
    /// RGB base colour, A cavity.
    pub albedo: Target,
    /// RG octahedral normal, B roughness, A metallic.
    pub normal: Target,
    pub emissive: Target,
    /// UV offset from this pixel to its position in the previous frame.
    pub motion: Target,
    /// Reversed-Z depth; zero where there is no geometry.
    pub depth: Target,
}

impl GBuffer {
    pub fn new(gpu: &Gpu, size: (u32, u32)) -> Self {
        let none = wgpu::TextureUsages::empty();
        // Solid particles draw their depth and motion into these when there is no water, and
        // tests read them back.
        let render = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        Self {
            albedo: Target::new(gpu, "gbuffer albedo", size, ALBEDO_FORMAT, none),
            normal: Target::new(gpu, "gbuffer normal", size, NORMAL_FORMAT, none),
            emissive: Target::new(gpu, "gbuffer emissive", size, EMISSIVE_FORMAT, none),
            motion: Target::new(gpu, "gbuffer motion", size, MOTION_FORMAT, render),
            depth: Target::new(gpu, "gbuffer depth", size, DEPTH_FORMAT, render),
        }
    }
}

/// The resolve compute pass.
pub struct Resolve {
    pipeline: wgpu::ComputePipeline,
    data_layout: wgpu::BindGroupLayout,
    output_layout: wgpu::BindGroupLayout,
}

/// The bind groups of one resolve configuration, rebuilt on resize.
pub struct ResolveBindings {
    data: wgpu::BindGroup,
    output: wgpu::BindGroup,
}

impl Resolve {
    /// `wetness` is the layout of the third bind group, from [`crate::ocean::waterline`].
    pub fn new(gpu: &Gpu, wetness: &wgpu::BindGroupLayout, skinned: bool) -> Self {
        let data_layout = LayoutBuilder::new()
            .uniform()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .texture_unfilterable()
            .build(gpu, "resolve data");
        let output_layout = LayoutBuilder::new()
            .storage_texture(ALBEDO_FORMAT)
            .storage_texture(NORMAL_FORMAT)
            .storage_texture(EMISSIVE_FORMAT)
            .storage_texture(MOTION_FORMAT)
            .storage_texture(DEPTH_FORMAT)
            .sampler()
            .texture_array()
            .texture_array()
            .texture_array()
            .texture_array()
            .texture_array()
            .texture_array()
            .texture_array()
            .texture_array()
            .texture_array()
            .build(gpu, "resolve output");
        let module = shaders::module_for(&gpu.device, "vg/resolve.wgsl", skinned);
        Self {
            pipeline: gpu.compute_pipeline(
                "resolve",
                &module,
                "main",
                &[&data_layout, &output_layout, wetness],
            ),
            data_layout,
            output_layout,
        }
    }

    pub fn bindings(
        &self,
        gpu: &Gpu,
        frame: &wgpu::Buffer,
        scene: &GpuScene,
        visibility: &VisibilityBuffer,
        work: &CullWork,
        gbuffer: &GBuffer,
    ) -> ResolveBindings {
        ResolveBindings {
            data: gpu.bind_group(
                "resolve data",
                &self.data_layout,
                &[
                    frame.as_entire_binding(),
                    visibility.depth.as_entire_binding(),
                    visibility.ids.as_entire_binding(),
                    work.visible.as_entire_binding(),
                    scene.instances.as_entire_binding(),
                    scene.clusters.as_entire_binding(),
                    scene.vertices.as_entire_binding(),
                    scene.triangles.as_entire_binding(),
                    scene.materials.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&scene.skins.bones_view),
                ],
            ),
            output: gpu.bind_group(
                "resolve output",
                &self.output_layout,
                &[
                    wgpu::BindingResource::TextureView(&gbuffer.albedo.view),
                    wgpu::BindingResource::TextureView(&gbuffer.normal.view),
                    wgpu::BindingResource::TextureView(&gbuffer.emissive.view),
                    wgpu::BindingResource::TextureView(&gbuffer.motion.view),
                    wgpu::BindingResource::TextureView(&gbuffer.depth.view),
                    wgpu::BindingResource::Sampler(&scene.material_sampler),
                    wgpu::BindingResource::TextureView(&scene.textures.base.color_view),
                    wgpu::BindingResource::TextureView(&scene.textures.base.normal_view),
                    wgpu::BindingResource::TextureView(&scene.textures.base.material_view),
                    wgpu::BindingResource::TextureView(&scene.textures.middle.color_view),
                    wgpu::BindingResource::TextureView(&scene.textures.middle.normal_view),
                    wgpu::BindingResource::TextureView(&scene.textures.middle.material_view),
                    wgpu::BindingResource::TextureView(&scene.textures.top.color_view),
                    wgpu::BindingResource::TextureView(&scene.textures.top.normal_view),
                    wgpu::BindingResource::TextureView(&scene.textures.top.material_view),
                ],
            ),
        }
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &ResolveBindings,
        wetness: &wgpu::BindGroup,
        size: (u32, u32),
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resolve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.data, &[]);
        pass.set_bind_group(1, &bindings.output, &[]);
        pass.set_bind_group(2, wetness, &[]);
        pass.dispatch_workgroups(groups_for(size.0, 8), groups_for(size.1, 8), 1);
    }
}
