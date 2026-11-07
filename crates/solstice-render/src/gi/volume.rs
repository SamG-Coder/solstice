//! The world-space scene representation that indirect rays trace.

use crate::gpu::{groups_for, Gpu, LayoutBuilder};
use crate::scene::GpuScene;
use crate::shaders;
use crate::vg::{CullPhase, CullPipelines, CullWork, PassView, ViewSlots};
use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec2, Vec3, Vec4};

pub const LEVEL_COUNT: usize = 5;
const RESOLUTION: u32 = 96;
const RESOLUTION_I: i32 = RESOLUTION as i32;
const VOXELS_PER_LEVEL: u64 = (RESOLUTION as u64).pow(3);
const TOTAL_VOXELS: u64 = VOXELS_PER_LEVEL * LEVEL_COUNT as u64;
/// Voxel size of the finest level, in metres. Each further level doubles it.
const BASE_VOXEL_SIZE: f32 = 0.2;
/// A level moves only when the camera has moved this many voxels. Moving costs a rebuild.
const SNAP_VOXELS: i32 = 8;
const ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Jump-flood step lengths. Together they must span half the grid's width.
const FLOOD_STEPS: [i32; 6] = [48, 24, 12, 6, 3, 1];
/// Finer levels get refreshed more often, since they sit closest to the viewer.
const LIGHT_SCHEDULE: [u32; 8] = [0, 1, 0, 2, 0, 3, 0, 4];

/// Twin of `VolumeParams` in `common/environment.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VolumeUniform {
    levels: [Vec4; LEVEL_COUNT],
    resolution: f32,
    inv_resolution: f32,
    _pad: [f32; 2],
}

/// Twin of `Params` in `gi/voxelize.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VoxelizeParams {
    origin: Vec3,
    voxel_size: f32,
    resolution: u32,
    level_offset: u32,
    _pad: [u32; 2],
}

/// Twin of `Params` in `gi/jump_flood.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FloodParams {
    resolution: u32,
    level_offset: u32,
    step: IVec3,
    _pad: [u32; 3],
}

/// Twin of `Params` in `gi/voxel_light.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightParams {
    level: u32,
    frame: u32,
    resolution: u32,
    light_count: u32,
    wrap_origin: IVec3,
    _pad: u32,
}

/// Twin of `Params` in `gi/volume_write.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WriteParams {
    level: u32,
    resolution: u32,
    voxel_size: f32,
    _pad0: u32,
    wrap_origin: IVec3,
    _pad1: u32,
}

/// Per-level uniforms and bind groups.
struct Level {
    voxel_size: f32,
    /// Grid corner, in whole voxels from the world origin.
    origin_voxels: Option<IVec3>,
    /// Lighting updates so far. The checkerboard and ray sequence advance on this, not on
    /// the frame number: a level updated every other frame would otherwise only ever
    /// see even frame numbers, and half of its voxels would never be lit.
    light_updates: u32,
    voxelize_params: wgpu::Buffer,
    light_params: wgpu::Buffer,
    write_params: wgpu::Buffer,
    voxelize_group: wgpu::BindGroup,
    seed_group: wgpu::BindGroup,
    light_group: wgpu::BindGroup,
    write_group: wgpu::BindGroup,
}

/// The GI volume and the passes that maintain it.
pub struct GiVolume {
    pub atlas_view: wgpu::TextureView,
    pub uniform: wgpu::Buffer,
    voxels: wgpu::Buffer,
    seeds: wgpu::Buffer,
    flood_a: wgpu::Buffer,
    levels: Vec<Level>,
    voxelize_pipeline: wgpu::ComputePipeline,
    seed_pipeline: wgpu::ComputePipeline,
    flood_pipeline: wgpu::ComputePipeline,
    light_pipeline: wgpu::ComputePipeline,
    write_pipeline: wgpu::ComputePipeline,
    flood_groups: Vec<wgpu::BindGroup>,
    light_count: u32,
    schedule_index: usize,
}

/// What [`GiVolume::prepare`] decided should happen this frame.
pub struct VolumeWork {
    rebuild: Vec<usize>,
    relight: usize,
}

impl GiVolume {
    /// Creates the volume. `environment_layout` is the shared lighting bind group layout.
    pub fn new(
        gpu: &Gpu,
        scene: &GpuScene,
        work: &CullWork,
        environment_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let voxels =
            gpu.storage_buffer("gi voxels", TOTAL_VOXELS * 12, wgpu::BufferUsages::COPY_DST);
        let seeds = gpu.storage_buffer("gi seeds", TOTAL_VOXELS * 4, wgpu::BufferUsages::COPY_DST);
        let radiance =
            gpu.storage_buffer("gi radiance", TOTAL_VOXELS * 8, wgpu::BufferUsages::empty());
        let flood_a = gpu.storage_buffer(
            "gi flood a",
            VOXELS_PER_LEVEL * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let flood_b = gpu.storage_buffer(
            "gi flood b",
            VOXELS_PER_LEVEL * 4,
            wgpu::BufferUsages::empty(),
        );

        let atlas = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gi atlas"),
            size: wgpu::Extent3d {
                width: RESOLUTION,
                height: RESOLUTION,
                depth_or_array_layers: RESOLUTION * LEVEL_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: ATLAS_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&Default::default());

        let voxelize_layout = LayoutBuilder::new()
            .uniform()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_read()
            .storage_write()
            .texture_array()
            .sampler()
            .build(gpu, "gi voxelize");
        let flood_layout = LayoutBuilder::new()
            .uniform()
            .storage_read()
            .storage_read()
            .storage_write()
            .build(gpu, "gi flood");
        let light_layout = LayoutBuilder::new()
            .uniform()
            .storage_read()
            .storage_write()
            .storage_read()
            .build(gpu, "gi light");
        let write_layout = LayoutBuilder::new()
            .uniform()
            .storage_read()
            .storage_read()
            .storage_texture_3d(ATLAS_FORMAT)
            .build(gpu, "gi write");

        let voxelize_module = shaders::module(&gpu.device, "gi/voxelize.wgsl");
        let flood_module = shaders::module(&gpu.device, "gi/jump_flood.wgsl");
        let light_module = shaders::module(&gpu.device, "gi/voxel_light.wgsl");
        let write_module = shaders::module(&gpu.device, "gi/volume_write.wgsl");

        let uniform_of = |label: &str, size: usize| gpu.uniform_buffer(label, size as u64);

        let levels = (0..LEVEL_COUNT)
            .map(|index| {
                let voxel_size = BASE_VOXEL_SIZE * (1u32 << index) as f32;
                let level_offset = index as u64 * VOXELS_PER_LEVEL;

                let voxelize_params =
                    uniform_of("gi voxelize params", std::mem::size_of::<VoxelizeParams>());
                let light_params =
                    uniform_of("gi light params", std::mem::size_of::<LightParams>());
                let write_params =
                    uniform_of("gi write params", std::mem::size_of::<WriteParams>());

                let seed_params = uniform_of("gi seed params", std::mem::size_of::<FloodParams>());
                gpu.queue.write_buffer(
                    &seed_params,
                    0,
                    bytemuck::bytes_of(&FloodParams {
                        resolution: RESOLUTION,
                        level_offset: level_offset as u32,
                        step: IVec3::ZERO,
                        _pad: [0; 3],
                    }),
                );

                Level {
                    voxel_size,
                    origin_voxels: None,
                    light_updates: 0,
                    voxelize_group: gpu.bind_group(
                        "gi voxelize",
                        &voxelize_layout,
                        &[
                            voxelize_params.as_entire_binding(),
                            scene.instances.as_entire_binding(),
                            scene.clusters.as_entire_binding(),
                            scene.vertices.as_entire_binding(),
                            scene.triangles.as_entire_binding(),
                            work.visible.as_entire_binding(),
                            work.state().as_entire_binding(),
                            scene.materials.as_entire_binding(),
                            voxels.as_entire_binding(),
                            wgpu::BindingResource::TextureView(&scene.color_textures),
                            wgpu::BindingResource::Sampler(&scene.material_sampler),
                        ],
                    ),
                    seed_group: gpu.bind_group(
                        "gi seed",
                        &flood_layout,
                        &[
                            seed_params.as_entire_binding(),
                            voxels.as_entire_binding(),
                            flood_b.as_entire_binding(),
                            flood_a.as_entire_binding(),
                        ],
                    ),
                    light_group: gpu.bind_group(
                        "gi light",
                        &light_layout,
                        &[
                            light_params.as_entire_binding(),
                            voxels.as_entire_binding(),
                            radiance.as_entire_binding(),
                            scene.lights.as_entire_binding(),
                        ],
                    ),
                    write_group: gpu.bind_group(
                        "gi write",
                        &write_layout,
                        &[
                            write_params.as_entire_binding(),
                            seeds.as_entire_binding(),
                            radiance.as_entire_binding(),
                            wgpu::BindingResource::TextureView(&atlas_view),
                        ],
                    ),
                    voxelize_params,
                    light_params,
                    write_params,
                }
            })
            .collect();

        // One pass per axis per step. Passes alternate buffers, starting from A, where the
        // seed pass wrote. The even total means the result ends up back in A.
        let mut flood_groups = Vec::new();
        let axes = [IVec3::X, IVec3::Y, IVec3::Z];
        for (pass, step) in FLOOD_STEPS
            .iter()
            .flat_map(|&s| axes.map(|axis| axis * s))
            .enumerate()
        {
            let params = uniform_of("gi flood params", std::mem::size_of::<FloodParams>());
            gpu.queue.write_buffer(
                &params,
                0,
                bytemuck::bytes_of(&FloodParams {
                    resolution: RESOLUTION,
                    level_offset: 0,
                    step,
                    _pad: [0; 3],
                }),
            );
            let (from, to) = if pass % 2 == 0 {
                (&flood_a, &flood_b)
            } else {
                (&flood_b, &flood_a)
            };
            flood_groups.push(gpu.bind_group(
                "gi flood",
                &flood_layout,
                &[
                    params.as_entire_binding(),
                    voxels.as_entire_binding(),
                    from.as_entire_binding(),
                    to.as_entire_binding(),
                ],
            ));
        }
        assert!(
            flood_groups.len() % 2 == 0,
            "the flood result must finish in buffer A"
        );

        Self {
            atlas_view,
            uniform: uniform_of("gi volume", std::mem::size_of::<VolumeUniform>()),
            voxelize_pipeline: gpu.compute_pipeline(
                "gi voxelize",
                &voxelize_module,
                "main",
                &[&voxelize_layout],
            ),
            seed_pipeline: gpu.compute_pipeline("gi seed", &flood_module, "seed", &[&flood_layout]),
            flood_pipeline: gpu.compute_pipeline(
                "gi flood",
                &flood_module,
                "flood",
                &[&flood_layout],
            ),
            light_pipeline: gpu.compute_pipeline(
                "gi light",
                &light_module,
                "main",
                &[&light_layout, environment_layout],
            ),
            write_pipeline: gpu.compute_pipeline(
                "gi write",
                &write_module,
                "main",
                &[&write_layout],
            ),
            voxels,
            seeds,
            flood_a,
            levels,
            flood_groups,
            light_count: scene.light_count,
            schedule_index: 0,
        }
    }

    /// Follows the camera and decides this frame's work. Writes the views of any levels to
    /// rebuild to view slots starting at `first_slot`.
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        slots: &ViewSlots,
        first_slot: u32,
        camera_position: Vec3,
    ) -> VolumeWork {
        let first_build = self.levels[0].origin_voxels.is_none();
        let mut rebuild = Vec::new();
        for (index, level) in self.levels.iter_mut().enumerate() {
            let wanted = snapped_origin(camera_position, level.voxel_size);
            // Moving is costly, so at most one level moves per frame after the first.
            if level.origin_voxels != Some(wanted) && (first_build || rebuild.is_empty()) {
                level.origin_voxels = Some(wanted);
                rebuild.push(index);
                write_level_view(
                    gpu,
                    slots,
                    first_slot + index as u32,
                    wanted,
                    level.voxel_size,
                );
            }
        }

        let relight = LIGHT_SCHEDULE[self.schedule_index] as usize;
        self.schedule_index = (self.schedule_index + 1) % LIGHT_SCHEDULE.len();
        self.levels[relight].light_updates = self.levels[relight].light_updates.wrapping_add(1);

        let mut uniform = VolumeUniform {
            levels: [Vec4::ZERO; LEVEL_COUNT],
            resolution: RESOLUTION as f32,
            inv_resolution: 1.0 / RESOLUTION as f32,
            _pad: [0.0; 2],
        };
        for (index, level) in self.levels.iter().enumerate() {
            let origin = level.origin_voxels.unwrap_or(IVec3::ZERO);
            uniform.levels[index] = (origin.as_vec3() * level.voxel_size).extend(level.voxel_size);
            gpu.queue.write_buffer(
                &level.voxelize_params,
                0,
                bytemuck::bytes_of(&VoxelizeParams {
                    origin: origin.as_vec3() * level.voxel_size,
                    voxel_size: level.voxel_size,
                    resolution: RESOLUTION,
                    level_offset: (index as u64 * VOXELS_PER_LEVEL) as u32,
                    _pad: [0; 2],
                }),
            );
            gpu.queue.write_buffer(
                &level.light_params,
                0,
                bytemuck::bytes_of(&LightParams {
                    level: index as u32,
                    frame: level.light_updates,
                    resolution: RESOLUTION,
                    light_count: self.light_count,
                    wrap_origin: origin,
                    _pad: 0,
                }),
            );
            gpu.queue.write_buffer(
                &level.write_params,
                0,
                bytemuck::bytes_of(&WriteParams {
                    level: index as u32,
                    resolution: RESOLUTION,
                    voxel_size: level.voxel_size,
                    _pad0: 0,
                    wrap_origin: origin,
                    _pad1: 0,
                }),
            );
        }
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));

        VolumeWork { rebuild, relight }
    }

    /// Records rebuilds of moved levels, then a lighting update.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cull: &CullPipelines,
        scene: &GpuScene,
        cull_scene: &wgpu::BindGroup,
        cull_work: &CullWork,
        environment: &wgpu::BindGroup,
        first_slot: u32,
        work: &VolumeWork,
    ) {
        for &index in &work.rebuild {
            cull.record(
                encoder,
                scene,
                cull_scene,
                first_slot + index as u32,
                cull_work,
                CullPhase::First,
            );
            self.record_rebuild(encoder, index, cull_work);
            self.record_write(encoder, index);
        }

        let groups = groups_for(RESOLUTION, 4);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi light"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.light_pipeline);
            pass.set_bind_group(0, &self.levels[work.relight].light_group, &[]);
            pass.set_bind_group(1, environment, &[]);
            pass.dispatch_workgroups(groups, groups, groups);
        }
        self.record_write(encoder, work.relight);
    }

    fn record_rebuild(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        index: usize,
        cull_work: &CullWork,
    ) {
        let level_offset = index as u64 * VOXELS_PER_LEVEL;
        encoder.clear_buffer(&self.voxels, level_offset * 12, Some(VOXELS_PER_LEVEL * 12));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi voxelize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.voxelize_pipeline);
            pass.set_bind_group(0, &self.levels[index].voxelize_group, &[]);
            pass.dispatch_workgroups_indirect(cull_work.args(), CullWork::software_args_offset());
        }
        {
            let groups = groups_for(RESOLUTION, 4);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi jump flood"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.seed_pipeline);
            pass.set_bind_group(0, &self.levels[index].seed_group, &[]);
            pass.dispatch_workgroups(groups, groups, groups);

            pass.set_pipeline(&self.flood_pipeline);
            for group in &self.flood_groups {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(groups, groups, groups);
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.flood_a,
            0,
            &self.seeds,
            level_offset * 4,
            VOXELS_PER_LEVEL * 4,
        );
    }

    fn record_write(&self, encoder: &mut wgpu::CommandEncoder, index: usize) {
        let groups = groups_for(RESOLUTION, 4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gi volume write"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.write_pipeline);
        pass.set_bind_group(0, &self.levels[index].write_group, &[]);
        pass.dispatch_workgroups(groups, groups, groups);
    }
}

/// The grid corner for a level centred near `position`, in whole voxels.
fn snapped_origin(position: Vec3, voxel_size: f32) -> IVec3 {
    let center = (position / voxel_size).floor().as_ivec3();
    let snapped = IVec3::new(
        center.x.div_euclid(SNAP_VOXELS),
        center.y.div_euclid(SNAP_VOXELS),
        center.z.div_euclid(SNAP_VOXELS),
    ) * SNAP_VOXELS;
    snapped - IVec3::splat(RESOLUTION_I / 2)
}

/// Writes the orthographic view that selects a level's clusters.
fn write_level_view(
    gpu: &Gpu,
    slots: &ViewSlots,
    slot: u32,
    origin_voxels: IVec3,
    voxel_size: f32,
) {
    let half = voxel_size * RESOLUTION as f32 * 0.5;
    let center = (origin_voxels.as_vec3() + Vec3::splat(RESOLUTION as f32 * 0.5)) * voxel_size;
    let view = Mat4::look_to_rh(center + Vec3::Z * half, Vec3::NEG_Z, Vec3::Y);
    let projection = Mat4::orthographic_rh(-half, half, -half, half, 0.0, half * 2.0);
    let mut pass_view = PassView::orthographic(
        projection * view,
        center,
        Vec2::splat(RESOLUTION as f32),
        1.0 / voxel_size,
        // Half a voxel of geometric error is invisible at this resolution.
        0.5,
    );
    pass_view.allow_software = 2;
    slots.write(gpu, slot, &pass_view);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_keeps_the_camera_near_the_centre_of_the_grid() {
        for position in [
            Vec3::ZERO,
            Vec3::new(13.7, -4.2, 100.1),
            Vec3::new(-99.0, 7.0, -0.1),
        ] {
            let origin = snapped_origin(position, BASE_VOXEL_SIZE);
            let local = position / BASE_VOXEL_SIZE - origin.as_vec3();
            let middle = RESOLUTION as f32 * 0.5;
            assert!((local - Vec3::splat(middle)).abs().max_element() <= SNAP_VOXELS as f32 + 1.0);
        }
    }

    #[test]
    fn origin_stays_put_for_small_moves() {
        let origin = snapped_origin(Vec3::new(0.5, 0.5, 0.5), BASE_VOXEL_SIZE);
        assert_eq!(
            origin,
            snapped_origin(Vec3::new(0.51, 0.52, 0.53), BASE_VOXEL_SIZE)
        );
    }

    #[test]
    fn flood_steps_span_half_the_grid() {
        assert!(FLOOD_STEPS.iter().sum::<i32>() >= RESOLUTION_I / 2);
        assert_eq!(FLOOD_STEPS[FLOOD_STEPS.len() - 1], 1);
    }

    #[test]
    fn every_level_is_scheduled_for_lighting() {
        for level in 0..LEVEL_COUNT as u32 {
            assert!(LIGHT_SCHEDULE.contains(&level));
        }
    }
}
