//! Screen-space passes of the GI system: probes and reflections.

use crate::gpu::{groups_for, Gpu, LayoutBuilder, Target};
use crate::shaders;
use crate::vg::GBuffer;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Must match the constants of the same names in `gi/probe_common.wgsl`.
const PROBE_SPACING: u32 = 16;
const PROBE_RAYS: u32 = 6;
/// Size of `Probe` in `gi/probe_common.wgsl`: nine SH vectors, position and normal.
const PROBE_BYTES: u64 = 11 * 16;

/// Pipelines of the screen-space GI passes.
pub struct ScreenGi {
    trace_layout: wgpu::BindGroupLayout,
    project_layout: wgpu::BindGroupLayout,
    integrate_layout: wgpu::BindGroupLayout,
    reflect_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    filter: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
    reflect: wgpu::ComputePipeline,
}

/// Size-dependent targets and bind groups. Index 0 or 1 of each pair is chosen by frame
/// parity: the pair is written on one frame and read as history on the next.
pub struct ScreenGiBindings {
    pub diffuse: [Target; 2],
    pub reflections: [Target; 2],
    trace_groups: [wgpu::BindGroup; 2],
    project_group: wgpu::BindGroup,
    filter_group: wgpu::BindGroup,
    integrate_groups: [wgpu::BindGroup; 2],
    reflect_groups: [wgpu::BindGroup; 2],
    atlas_size: (u32, u32),
    grid_size: (u32, u32),
    size: (u32, u32),
    half_size: (u32, u32),
}

impl ScreenGi {
    pub fn new(gpu: &Gpu, environment_layout: &wgpu::BindGroupLayout) -> Self {
        let trace_layout = LayoutBuilder::new()
            .texture()
            .texture_unfilterable()
            .texture()
            .storage_texture(FORMAT)
            .build(gpu, "probe trace");
        let project_layout = LayoutBuilder::new()
            .texture()
            .texture_unfilterable()
            .texture()
            .storage_read()
            .storage_write()
            .build(gpu, "probe project");
        let integrate_layout = LayoutBuilder::new()
            .texture()
            .texture_unfilterable()
            .texture_unfilterable()
            .texture()
            .storage_read()
            .storage_texture(FORMAT)
            .build(gpu, "probe integrate");
        let reflect_layout = LayoutBuilder::new()
            .texture()
            .texture_unfilterable()
            .texture_unfilterable()
            .texture()
            .texture()
            .storage_texture(FORMAT)
            .build(gpu, "reflections");

        let trace_module = shaders::module(&gpu.device, "gi/probe_trace.wgsl");
        let project_module = shaders::module(&gpu.device, "gi/probe_project.wgsl");
        let integrate_module = shaders::module(&gpu.device, "gi/probe_integrate.wgsl");
        let reflect_module = shaders::module(&gpu.device, "gi/reflections.wgsl");

        Self {
            trace: gpu.compute_pipeline(
                "probe trace",
                &trace_module,
                "main",
                &[&trace_layout, environment_layout],
            ),
            project: gpu.compute_pipeline(
                "probe project",
                &project_module,
                "project",
                &[&project_layout, environment_layout],
            ),
            filter: gpu.compute_pipeline(
                "probe filter",
                &project_module,
                "filter_spatial",
                &[&project_layout, environment_layout],
            ),
            integrate: gpu.compute_pipeline(
                "probe integrate",
                &integrate_module,
                "main",
                &[&integrate_layout, environment_layout],
            ),
            reflect: gpu.compute_pipeline(
                "reflections",
                &reflect_module,
                "main",
                &[&reflect_layout, environment_layout],
            ),
            trace_layout,
            project_layout,
            integrate_layout,
            reflect_layout,
        }
    }

    /// Creates targets and bind groups for a render `size`. `scene_history` holds last
    /// frame's final colour for each parity.
    pub fn bindings(
        &self,
        gpu: &Gpu,
        size: (u32, u32),
        gbuffer: &GBuffer,
        scene_history: [&wgpu::TextureView; 2],
    ) -> ScreenGiBindings {
        let none = wgpu::TextureUsages::empty();
        let grid_size = (
            size.0.div_ceil(PROBE_SPACING),
            size.1.div_ceil(PROBE_SPACING),
        );
        let atlas_size = (grid_size.0 * PROBE_RAYS, grid_size.1 * PROBE_RAYS);
        let half_size = ((size.0 / 2).max(1), (size.1 / 2).max(1));

        let atlas = Target::new(gpu, "probe atlas", atlas_size, FORMAT, none);
        let probe_bytes = u64::from(grid_size.0) * u64::from(grid_size.1) * PROBE_BYTES;
        let probes_raw = gpu.storage_buffer("probes raw", probe_bytes, wgpu::BufferUsages::empty());
        let probes =
            gpu.storage_buffer("probes filtered", probe_bytes, wgpu::BufferUsages::empty());
        let diffuse = [
            Target::new(gpu, "gi diffuse 0", size, FORMAT, none),
            Target::new(gpu, "gi diffuse 1", size, FORMAT, none),
        ];
        let reflections = [
            Target::new(gpu, "gi reflections 0", half_size, FORMAT, none),
            Target::new(gpu, "gi reflections 1", half_size, FORMAT, none),
        ];

        let tex = |view| wgpu::BindingResource::TextureView(view);
        let trace_group = |parity: usize| {
            gpu.bind_group(
                "probe trace",
                &self.trace_layout,
                &[
                    tex(&gbuffer.normal.view),
                    tex(&gbuffer.depth.view),
                    tex(scene_history[parity]),
                    tex(&atlas.view),
                ],
            )
        };
        let project_bindings = |input: &wgpu::Buffer, output: &wgpu::Buffer| {
            gpu.bind_group(
                "probe project",
                &self.project_layout,
                &[
                    tex(&gbuffer.normal.view),
                    tex(&gbuffer.depth.view),
                    tex(&atlas.view),
                    input.as_entire_binding(),
                    output.as_entire_binding(),
                ],
            )
        };
        let integrate_group = |parity: usize| {
            gpu.bind_group(
                "probe integrate",
                &self.integrate_layout,
                &[
                    tex(&gbuffer.normal.view),
                    tex(&gbuffer.depth.view),
                    tex(&gbuffer.motion.view),
                    tex(&diffuse[1 - parity].view),
                    probes.as_entire_binding(),
                    tex(&diffuse[parity].view),
                ],
            )
        };
        let reflect_group = |parity: usize| {
            gpu.bind_group(
                "reflections",
                &self.reflect_layout,
                &[
                    tex(&gbuffer.normal.view),
                    tex(&gbuffer.depth.view),
                    tex(&gbuffer.motion.view),
                    tex(scene_history[parity]),
                    tex(&reflections[1 - parity].view),
                    tex(&reflections[parity].view),
                ],
            )
        };

        ScreenGiBindings {
            trace_groups: [trace_group(0), trace_group(1)],
            project_group: project_bindings(&probes, &probes_raw),
            filter_group: project_bindings(&probes_raw, &probes),
            integrate_groups: [integrate_group(0), integrate_group(1)],
            reflect_groups: [reflect_group(0), reflect_group(1)],
            diffuse,
            reflections,
            atlas_size,
            grid_size,
            size,
            half_size,
        }
    }

    /// Records all passes. Afterwards `diffuse[parity]` and `reflections[parity]` are ready.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &ScreenGiBindings,
        environment: &wgpu::BindGroup,
        parity: usize,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("screen gi"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, environment, &[]);

        pass.set_pipeline(&self.trace);
        pass.set_bind_group(0, &bindings.trace_groups[parity], &[]);
        pass.dispatch_workgroups(
            groups_for(bindings.atlas_size.0, 8),
            groups_for(bindings.atlas_size.1, 8),
            1,
        );

        let grid = (
            groups_for(bindings.grid_size.0, 8),
            groups_for(bindings.grid_size.1, 8),
        );
        pass.set_pipeline(&self.project);
        pass.set_bind_group(0, &bindings.project_group, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, 1);

        pass.set_pipeline(&self.filter);
        pass.set_bind_group(0, &bindings.filter_group, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, 1);

        pass.set_pipeline(&self.integrate);
        pass.set_bind_group(0, &bindings.integrate_groups[parity], &[]);
        pass.dispatch_workgroups(
            groups_for(bindings.size.0, 8),
            groups_for(bindings.size.1, 8),
            1,
        );

        pass.set_pipeline(&self.reflect);
        pass.set_bind_group(0, &bindings.reflect_groups[parity], &[]);
        pass.dispatch_workgroups(
            groups_for(bindings.half_size.0, 8),
            groups_for(bindings.half_size.1, 8),
            1,
        );
    }
}
