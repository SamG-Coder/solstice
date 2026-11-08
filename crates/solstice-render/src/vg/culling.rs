//! GPU-driven hierarchy traversal. See `shaders/vg/cull.wgsl` for the algorithm.

use crate::gpu::{groups_for, Gpu, LayoutBuilder};
use crate::scene::GpuScene;
use crate::shaders;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Twin of `PassView` in `vg/pass_view.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PassView {
    pub raster_view_proj: Mat4,
    pub occlusion_view: Mat4,
    pub planes: [Vec4; 6],
    pub camera_position: Vec3,
    pub lod_threshold: f32,
    pub viewport: Vec2,
    pub proj_scale: Vec2,
    pub hzb_size: Vec2,
    pub near: f32,
    pub lod_scale: f32,
    pub is_ortho: u32,
    pub phase: u32,
    pub occlusion_enabled: u32,
    pub allow_software: u32,
    /// Which instances the view draws: an [`InstanceFilter`].
    pub instance_filter: u32,
    pub _pad: [u32; 3],
}

/// Which instances a view draws. Twin of the `INSTANCES_` constants in `vg/pass_view.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum InstanceFilter {
    All = 0,
    /// Only those that cannot move, for views kept from frame to frame, which would otherwise
    /// keep a moving instance where it was.
    Static = 1,
    /// Only those that can move, drawn each frame over a cached view.
    Movable = 2,
    /// Static instances, and movable ones that stay in the GI voxels.
    Gi = 3,
}

impl PassView {
    /// Extracts normalised frustum planes from a view-projection matrix, with depth
    /// running 0 to 1. An infinite far plane yields a degenerate plane that passes all.
    pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
        let (r0, r1, r2, r3) = (
            view_proj.row(0),
            view_proj.row(1),
            view_proj.row(2),
            view_proj.row(3),
        );
        [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(|plane| {
            let length = plane.truncate().length();
            if length > 1.0e-6 {
                plane / length
            } else {
                Vec4::new(0.0, 0.0, 0.0, 1.0)
            }
        })
    }

    /// A view with no occlusion testing and no software rasterization, for orthographic
    /// volumes such as shadow cascades. `pixels_per_unit` converts LOD error to pixels. These
    /// views are kept from frame to frame, so they draw only static instances; see
    /// [`Self::drawing`].
    pub fn orthographic(
        view_proj: Mat4,
        center: Vec3,
        viewport: Vec2,
        pixels_per_unit: f32,
        lod_threshold: f32,
    ) -> Self {
        Self {
            raster_view_proj: view_proj,
            occlusion_view: Mat4::IDENTITY,
            planes: Self::frustum_planes(view_proj),
            camera_position: center,
            lod_threshold,
            viewport,
            proj_scale: Vec2::ONE,
            hzb_size: Vec2::ONE,
            near: 0.0,
            lod_scale: pixels_per_unit,
            is_ortho: 1,
            phase: 0,
            occlusion_enabled: 0,
            allow_software: 0,
            instance_filter: InstanceFilter::Static as u32,
            _pad: [0; 3],
        }
    }

    /// The same view, drawing other instances.
    pub fn drawing(mut self, filter: InstanceFilter) -> Self {
        self.instance_filter = filter as u32;
        self
    }
}

/// Bytes between view slots. Comfortably above `PassView` and any device's offset alignment.
const SLOT_STRIDE: u32 = 512;
const _: () = assert!(std::mem::size_of::<PassView>() <= SLOT_STRIDE as usize);

/// A uniform buffer holding several [`PassView`]s, selected by dynamic offset.
pub struct ViewSlots {
    pub buffer: wgpu::Buffer,
}

impl ViewSlots {
    pub fn new(gpu: &Gpu, count: u32) -> Self {
        Self {
            buffer: gpu.uniform_buffer("view slots", u64::from(count * SLOT_STRIDE)),
        }
    }

    pub fn write(&self, gpu: &Gpu, slot: u32, view: &PassView) {
        gpu.queue.write_buffer(
            &self.buffer,
            u64::from(Self::offset(slot)),
            bytemuck::bytes_of(view),
        );
    }

    pub fn offset(slot: u32) -> u32 {
        slot * SLOT_STRIDE
    }

    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: 0,
            size: wgpu::BufferSize::new(std::mem::size_of::<PassView>() as u64),
        })
    }
}

/// Byte offsets into the indirect arguments buffer. See `Args` in `cull_control.wgsl`.
const ARGS_QUEUE_OFFSET: u64 = 0;
const ARGS_SOFTWARE_OFFSET: u64 = 12;
const ARGS_DRAW_OFFSET: u64 = 24;
const ARGS_SIZE: u64 = 40;
/// Size of `CullState` in bytes.
const STATE_SIZE: u64 = 48;

/// Which occlusion phase a cull runs. See the module docs of [`crate::vg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CullPhase {
    /// Starts a frame's cull: tests the previous depth and collects retries. Views
    /// without occlusion simply run this phase alone.
    First,
    /// Re-tests the retries against this frame's depth.
    Second,
}

/// The transient buffers of one culling run.
pub struct CullWork {
    state: wgpu::Buffer,
    pub visible: wgpu::Buffer,
    args: wgpu::Buffer,
    first_phase: wgpu::BindGroup,
    second_phase: wgpu::BindGroup,
    control: wgpu::BindGroup,
    stats: CullStats,
}

impl CullWork {
    pub fn new(
        gpu: &Gpu,
        pipelines: &CullPipelines,
        label: &str,
        queue_capacity: u64,
        visible_capacity: u64,
        page_requests: &wgpu::Buffer,
    ) -> Self {
        let state = gpu.storage_buffer(
            label,
            STATE_SIZE,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let queue_a = gpu.storage_buffer(label, queue_capacity * 8, wgpu::BufferUsages::empty());
        let queue_b = gpu.storage_buffer(label, queue_capacity * 8, wgpu::BufferUsages::empty());
        let visible = gpu.storage_buffer(label, visible_capacity * 8, wgpu::BufferUsages::empty());
        let args = gpu.storage_buffer(label, ARGS_SIZE, wgpu::BufferUsages::INDIRECT);

        let bind = |queue: &wgpu::Buffer, retry: &wgpu::Buffer| {
            gpu.bind_group(
                label,
                &pipelines.work_layout,
                &[
                    state.as_entire_binding(),
                    queue.as_entire_binding(),
                    retry.as_entire_binding(),
                    visible.as_entire_binding(),
                    page_requests.as_entire_binding(),
                ],
            )
        };
        let first_phase = bind(&queue_a, &queue_b);
        let second_phase = bind(&queue_b, &queue_a);
        let control = gpu.bind_group(
            label,
            &pipelines.control_layout,
            &[state.as_entire_binding(), args.as_entire_binding()],
        );
        let stats = CullStats::new(gpu);

        Self {
            state,
            visible,
            args,
            first_phase,
            second_phase,
            control,
            stats,
        }
    }

    pub fn state(&self) -> &wgpu::Buffer {
        &self.state
    }

    pub fn args(&self) -> &wgpu::Buffer {
        &self.args
    }

    pub const fn software_args_offset() -> u64 {
        ARGS_SOFTWARE_OFFSET
    }

    pub const fn draw_args_offset() -> u64 {
        ARGS_DRAW_OFFSET
    }

    /// Queues a copy of the counters. Call at the end of a frame's encoding.
    pub fn capture_stats(&self, encoder: &mut wgpu::CommandEncoder) {
        self.stats.capture(encoder, &self.state);
    }

    /// Starts reading back any counters captured this frame. Call after submitting.
    pub fn map_stats(&self) {
        self.stats.request_map();
    }

    pub fn stats(&self) -> &CullStats {
        &self.stats
    }
}

/// Asynchronous readback of how much recent culling selected.
///
/// A buffer can only map while no command uses it, so a capture cycle takes several
/// frames: copy, map, wait for the callback, read, unmap. Numbers lag slightly behind,
/// which is fine for statistics, and the render loop never blocks on them.
pub struct CullStats {
    staging: wgpu::Buffer,
    shared: Arc<StatsShared>,
}

#[derive(Default)]
struct StatsShared {
    clusters: AtomicU32,
    software_clusters: AtomicU32,
    triangles: AtomicU32,
    /// Set when a copy was recorded and still needs its map request.
    copied: AtomicBool,
    /// Set from the map request until the data has been consumed.
    busy: AtomicBool,
    /// Set by the map callback once the buffer is readable.
    mapped: AtomicBool,
}

impl CullStats {
    fn new(gpu: &Gpu) -> Self {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cull stats"),
            size: STATE_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            staging,
            shared: Default::default(),
        }
    }

    fn capture(&self, encoder: &mut wgpu::CommandEncoder, state: &wgpu::Buffer) {
        if self.shared.mapped.swap(false, Ordering::AcqRel) {
            self.consume_mapped_data();
            self.shared.busy.store(false, Ordering::Release);
        }
        if !self.shared.busy.load(Ordering::Acquire) {
            encoder.copy_buffer_to_buffer(state, 0, &self.staging, 0, STATE_SIZE);
            self.shared.copied.store(true, Ordering::Release);
        }
    }

    fn consume_mapped_data(&self) {
        {
            let data = self
                .staging
                .slice(..)
                .get_mapped_range()
                .expect("the map has completed");
            let words: &[u32] = bytemuck::cast_slice(&data);
            // Word positions follow `CullState` in `vg/pass_view.wgsl`.
            let (software, hardware, triangles) = (words[4], words[5], words[8]);
            self.shared
                .clusters
                .store(software + hardware, Ordering::Release);
            self.shared
                .software_clusters
                .store(software, Ordering::Release);
            self.shared.triangles.store(triangles, Ordering::Release);
        }
        self.staging.unmap();
    }

    fn request_map(&self) {
        if !self.shared.copied.swap(false, Ordering::AcqRel) {
            return;
        }
        self.shared.busy.store(true, Ordering::Release);
        let shared = self.shared.clone();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| match result {
                Ok(()) => shared.mapped.store(true, Ordering::Release),
                Err(_) => shared.busy.store(false, Ordering::Release),
            });
    }

    /// Clusters selected in the most recently read-back frame.
    pub fn clusters(&self) -> u32 {
        self.shared.clusters.load(Ordering::Acquire)
    }

    /// Triangles selected in the most recently read-back frame.
    pub fn triangles(&self) -> u32 {
        self.shared.triangles.load(Ordering::Acquire)
    }

    /// Clusters sent to the software rasterizer in the most recently read-back frame.
    pub fn software_clusters(&self) -> u32 {
        self.shared.software_clusters.load(Ordering::Acquire)
    }
}

/// The compute pipelines of culling, shared by every view.
pub struct CullPipelines {
    pub scene_layout: wgpu::BindGroupLayout,
    work_layout: wgpu::BindGroupLayout,
    control_layout: wgpu::BindGroupLayout,
    seed: wgpu::ComputePipeline,
    process: wgpu::ComputePipeline,
    advance: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    begin_second_phase: wgpu::ComputePipeline,
}

impl CullPipelines {
    pub fn new(gpu: &Gpu) -> Self {
        let scene_layout = LayoutBuilder::new()
            .uniform_dynamic()
            .storage_read()
            .storage_read()
            .storage_read()
            .texture_unfilterable()
            .build(gpu, "cull scene");
        let work_layout = LayoutBuilder::new()
            .storage_write()
            .storage_write()
            .storage_write()
            .storage_write()
            .storage_write()
            .build(gpu, "cull work");
        let control_layout = LayoutBuilder::new()
            .storage_write()
            .storage_write()
            .build(gpu, "cull control");

        let cull = shaders::module(&gpu.device, "vg/cull.wgsl");
        let control = shaders::module(&gpu.device, "vg/cull_control.wgsl");
        let cull_layouts = [&scene_layout, &work_layout];
        Self {
            seed: gpu.compute_pipeline("cull seed", &cull, "seed", &cull_layouts),
            process: gpu.compute_pipeline("cull process", &cull, "process", &cull_layouts),
            advance: gpu.compute_pipeline("cull advance", &control, "advance", &[&control_layout]),
            finish: gpu.compute_pipeline("cull finish", &control, "finish", &[&control_layout]),
            begin_second_phase: gpu.compute_pipeline(
                "cull second phase",
                &control,
                "begin_second_phase",
                &[&control_layout],
            ),
            scene_layout,
            work_layout,
            control_layout,
        }
    }

    /// Binds a scene and depth pyramid for culling.
    pub fn scene_bind_group(
        &self,
        gpu: &Gpu,
        slots: &ViewSlots,
        scene: &GpuScene,
        hzb: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.bind_group(
            "cull scene",
            &self.scene_layout,
            &[
                slots.binding(),
                scene.instances.as_entire_binding(),
                scene.nodes.as_entire_binding(),
                scene.clusters.as_entire_binding(),
                wgpu::BindingResource::TextureView(hzb),
            ],
        )
    }

    /// Records one culling run. On return, `work` holds indirect arguments for
    /// rasterizing what this phase selected.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GpuScene,
        scene_group: &wgpu::BindGroup,
        slot: u32,
        work: &CullWork,
        phase: CullPhase,
    ) {
        if phase == CullPhase::First {
            encoder.clear_buffer(&work.state, 0, None);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cull"),
            timestamp_writes: None,
        });
        let work_group = match phase {
            CullPhase::First => &work.first_phase,
            CullPhase::Second => &work.second_phase,
        };
        let offset = [ViewSlots::offset(slot)];

        match phase {
            CullPhase::First => {
                pass.set_pipeline(&self.seed);
                pass.set_bind_group(0, scene_group, &offset);
                pass.set_bind_group(1, work_group, &[]);
                pass.dispatch_workgroups(groups_for(scene.gpu_instance_count, 64), 1, 1);
            }
            CullPhase::Second => {
                pass.set_pipeline(&self.begin_second_phase);
                pass.set_bind_group(0, &work.control, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }

        for _ in 0..scene.max_hierarchy_depth {
            pass.set_pipeline(&self.advance);
            pass.set_bind_group(0, &work.control, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&self.process);
            pass.set_bind_group(0, scene_group, &offset);
            pass.set_bind_group(1, work_group, &[]);
            pass.dispatch_workgroups_indirect(&work.args, ARGS_QUEUE_OFFSET);
        }

        pass.set_pipeline(&self.finish);
        pass.set_bind_group(0, &work.control, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pass_view_lays_out_as_its_shader_twin() {
        crate::shaders::assert_twin!(
            "vg/cull.wgsl",
            "PassView",
            PassView,
            [
                raster_view_proj,
                occlusion_view,
                planes,
                camera_position,
                lod_threshold,
                viewport,
                proj_scale,
                hzb_size,
                near,
                lod_scale,
                is_ortho,
                phase,
                occlusion_enabled,
                allow_software,
                instance_filter,
                _pad,
                _pad + 4,
                _pad + 8,
            ]
        );
    }

    #[test]
    fn orthographic_views_draw_only_what_stands_still() {
        let view = PassView::orthographic(Mat4::IDENTITY, Vec3::ZERO, Vec2::ONE, 1.0, 1.0);
        assert_eq!(view.instance_filter, InstanceFilter::Static as u32);
        let moving = view.drawing(InstanceFilter::Movable);
        assert_eq!(moving.instance_filter, InstanceFilter::Movable as u32);
    }

    #[test]
    fn frustum_planes_accept_points_inside_and_reject_points_outside() {
        let projection = Mat4::perspective_infinite_reverse_rh(1.0, 1.0, 0.1);
        let planes = PassView::frustum_planes(projection);
        let is_inside = |p: Vec3| {
            planes
                .iter()
                .all(|plane| plane.truncate().dot(p) + plane.w >= 0.0)
        };
        assert!(is_inside(Vec3::new(0.0, 0.0, -5.0)));
        assert!(is_inside(Vec3::new(0.0, 0.0, -5000.0)));
        assert!(!is_inside(Vec3::new(0.0, 0.0, 5.0)), "behind the camera");
        assert!(
            !is_inside(Vec3::new(50.0, 0.0, -5.0)),
            "right of the frustum"
        );
        assert!(
            !is_inside(Vec3::new(0.0, 0.0, -0.01)),
            "nearer than the near plane"
        );
    }

    #[test]
    fn orthographic_planes_bound_all_six_sides() {
        let projection = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.0, 10.0);
        let planes = PassView::frustum_planes(projection);
        let is_inside = |p: Vec3| {
            planes
                .iter()
                .all(|plane| plane.truncate().dot(p) + plane.w >= 0.0)
        };
        assert!(is_inside(Vec3::new(0.5, 0.5, -5.0)));
        assert!(!is_inside(Vec3::new(0.0, 0.0, -11.0)));
        assert!(!is_inside(Vec3::new(0.0, 0.0, 1.0)));
        assert!(!is_inside(Vec3::new(1.5, 0.0, -5.0)));
    }
}
