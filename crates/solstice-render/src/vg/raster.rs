//! Rasterizes selected clusters into the visibility buffer.

use super::culling::{CullWork, ViewSlots};
use crate::gpu::{Gpu, LayoutBuilder};
use crate::scene::GpuScene;
use crate::shaders;

/// Format of the colour attachment that exists only to satisfy WebGPU's rule that a
/// render pass have at least one attachment. Nothing is ever written to it.
const DUMMY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Per-pixel depth and triangle ID, as storage buffers both rasterizers can write.
pub struct VisibilityBuffer {
    pub depth: wgpu::Buffer,
    pub ids: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    dummy_target: wgpu::TextureView,
    size: (u32, u32),
}

impl VisibilityBuffer {
    pub fn new(gpu: &Gpu, rasterizer: &Rasterizer, size: (u32, u32)) -> Self {
        let bytes = u64::from(size.0) * u64::from(size.1) * 4;
        let depth = gpu.storage_buffer("visibility depth", bytes, wgpu::BufferUsages::COPY_DST);
        let ids = gpu.storage_buffer("visibility ids", bytes, wgpu::BufferUsages::COPY_DST);
        let bind_group = gpu.bind_group(
            "visibility",
            &rasterizer.vis_layout,
            &[depth.as_entire_binding(), ids.as_entire_binding()],
        );
        let dummy_target = gpu
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("visibility dummy target"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DUMMY_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        Self {
            depth,
            ids,
            bind_group,
            dummy_target,
            size,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Resets every pixel to "no geometry". Depth 0 is the far plane in reversed Z,
    /// and resolve checks depth before reading an ID, so IDs need no clearing.
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.depth, 0, None);
    }
}

/// Bindings for rasterizing the output of one [`CullWork`].
pub struct RasterBindings {
    pub group: wgpu::BindGroup,
}

/// The software and hardware rasterization pipelines.
pub struct Rasterizer {
    pub raster_layout: wgpu::BindGroupLayout,
    vis_layout: wgpu::BindGroupLayout,
    software: wgpu::ComputePipeline,
    hardware: wgpu::RenderPipeline,
    pub hardware_module: wgpu::ShaderModule,
}

impl Rasterizer {
    pub fn new(gpu: &Gpu) -> Self {
        let raster_layout = LayoutBuilder::new()
            .visibility(wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX_FRAGMENT)
            .uniform_dynamic()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .build(gpu, "raster");
        let vis_layout = LayoutBuilder::new()
            .visibility(wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT)
            .storage_write()
            .storage_write()
            .build(gpu, "visibility");

        let software_module = shaders::module(&gpu.device, "vg/raster_sw.wgsl");
        let software = gpu.compute_pipeline(
            "software raster",
            &software_module,
            "main",
            &[&raster_layout, &vis_layout],
        );

        let hardware_module = shaders::module(&gpu.device, "vg/raster_hw_fragment.wgsl");
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hardware raster"),
                bind_group_layouts: &[Some(&raster_layout), Some(&vis_layout)],
                immediate_size: 0,
            });
        let hardware = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hardware raster"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &hardware_module,
                    entry_point: Some("vs_visibility"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    // Facing is decided per instance in `raster_hw.wgsl`.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &hardware_module,
                    entry_point: Some("fs_visibility"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: DUMMY_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        Self {
            raster_layout,
            vis_layout,
            software,
            hardware,
            hardware_module,
        }
    }

    /// Binds a scene and a culling run's output for rasterization.
    pub fn bindings(
        &self,
        gpu: &Gpu,
        slots: &ViewSlots,
        scene: &GpuScene,
        work: &CullWork,
    ) -> RasterBindings {
        RasterBindings {
            group: gpu.bind_group(
                "raster",
                &self.raster_layout,
                &[
                    slots.binding(),
                    scene.instances.as_entire_binding(),
                    scene.clusters.as_entire_binding(),
                    scene.vertices.as_entire_binding(),
                    scene.triangles.as_entire_binding(),
                    work.visible.as_entire_binding(),
                    work.state().as_entire_binding(),
                ],
            ),
        }
    }

    /// Rasterizes everything the most recent cull of `work` selected.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &RasterBindings,
        slot: u32,
        work: &CullWork,
        target: &VisibilityBuffer,
    ) {
        let offset = [ViewSlots::offset(slot)];
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("software raster"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.software);
            pass.set_bind_group(0, &bindings.group, &offset);
            pass.set_bind_group(1, &target.bind_group, &[]);
            pass.dispatch_workgroups_indirect(work.args(), CullWork::software_args_offset());
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hardware raster"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.dummy_target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&self.hardware);
        pass.set_bind_group(0, &bindings.group, &offset);
        pass.set_bind_group(1, &target.bind_group, &[]);
        pass.draw_indirect(work.args(), CullWork::draw_args_offset());
    }
}
