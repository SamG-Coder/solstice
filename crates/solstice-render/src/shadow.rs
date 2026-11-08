//! Cascaded sun shadow maps, drawn through the same cluster culling as the main view.
//!
//! Shadow maps are the most expensive pass on small GPUs, and most of a scene stands still, so
//! a cascade holds only static instances and is only redrawn when its view changes. Views move
//! in coarse steps to make that rare: each cascade covers slightly more than it must, so it can
//! stay put while the camera moves. When several cascades are stale, the far ones take turns.
//!
//! Movable instances are drawn every frame into a map of their own, through the same view as
//! the cascade they fall in, and shading multiplies the light the two let through. The four
//! cascades share
//! one more layer of the array for this, a quarter each, at half their resolution: a moving
//! thing's shadow is a little softer, but it costs one layer rather than four, and no new
//! binding. Shaders that sample the shadows are already at WebGPU's limit of 16 textures. A
//! cascade with nothing movable in it draws nothing, and is not sampled twice.

use crate::frame::Camera;
use crate::gpu::{Gpu, LayoutBuilder};
use crate::scene::GpuScene;
use crate::vg::{
    CullPhase, CullPipelines, CullWork, InstanceFilter, PassView, RasterBindings, Rasterizer,
    ViewSlots,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

pub const CASCADE_COUNT: usize = 4;
const RESOLUTION: u32 = 2048;
/// Each cascade's quarter of the layer for movable instances, the last layer of the array.
const MOVING_RESOLUTION: u32 = RESOLUTION / 2;
const MOVING_LAYER: u32 = CASCADE_COUNT as u32;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Far end of each cascade, in metres from the camera.
const SPLITS: [f32; CASCADE_COUNT] = [12.0, 38.0, 120.0, 420.0];
/// How far casters may lie towards the sun beyond a cascade's bounds.
const DEPTH_REACH: f32 = 900.0;
/// Cascades move in steps of this many texels. A whole number keeps shadow edges from
/// crawling as they move, and a large one means they rarely need to move at all.
const SNAP_TEXELS: f32 = 64.0;
/// Extra radius that covers the furthest a slice can be from its snapped centre, which is
/// half a snap step on each of three axes. The bounds test in this module checks that it is enough.
const SNAP_PADDING: f32 = 1.07;

/// Twin of `Shadows` in `lighting/deferred.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowUniform {
    view_proj: [Mat4; CASCADE_COUNT],
    splits: Vec4,
    texel_size: Vec4,
    /// 1 for each cascade whose movable instances were drawn this frame, else 0.
    moving: Vec4,
}

/// What [`ShadowMaps::record`] draws this frame.
pub struct ShadowWork {
    /// Cascades whose static contents are redrawn.
    pub due: [bool; CASCADE_COUNT],
    /// Cascades with movable instances to draw.
    pub moving: [bool; CASCADE_COUNT],
}

/// Shadow map resources and passes.
pub struct ShadowMaps {
    pub array_view: wgpu::TextureView,
    pub comparison_sampler: wgpu::Sampler,
    pub uniform: wgpu::Buffer,
    layer_views: Vec<wgpu::TextureView>,
    /// The layer movable instances are drawn into.
    moving_view: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    data: ShadowUniform,
    /// What each cascade last drew, to tell when it has gone stale.
    drawn: [Option<(Mat4, f32)>; CASCADE_COUNT],
    /// The centre and radius of each cascade's view as last drawn.
    extents: [(Vec3, f32); CASCADE_COUNT],
    next_far_cascade: usize,
    /// Cascades that must redraw because the geometry on the GPU has improved.
    outdated: [bool; CASCADE_COUNT],
    /// Redraw every stale cascade next frame, rather than the far ones taking turns.
    redraw_all: bool,
}

impl ShadowMaps {
    pub fn new(gpu: &Gpu, rasterizer: &Rasterizer) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: RESOLUTION,
                height: RESOLUTION,
                depth_or_array_layers: CASCADE_COUNT as u32 + 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let layer_view = |layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let layer_views = (0..CASCADE_COUNT as u32).map(layer_view).collect();
        let moving_view = layer_view(MOVING_LAYER);
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[Some(&rasterizer.raster_layout)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &rasterizer.hardware_module,
                    entry_point: Some("vs_depth"),
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
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 0,
                        slope_scale: 1.5,
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            });

        Self {
            array_view,
            comparison_sampler: gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("shadow comparison"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                compare: Some(wgpu::CompareFunction::LessEqual),
                ..Default::default()
            }),
            uniform: gpu.uniform_buffer("shadow", std::mem::size_of::<ShadowUniform>() as u64),
            layer_views,
            moving_view,
            pipeline,
            data: ShadowUniform {
                view_proj: [Mat4::IDENTITY; CASCADE_COUNT],
                splits: Vec4::from_array(SPLITS),
                texel_size: Vec4::ONE,
                moving: Vec4::ZERO,
            },
            drawn: [None; CASCADE_COUNT],
            extents: [(Vec3::ZERO, 1.0); CASCADE_COUNT],
            next_far_cascade: 1,
            outdated: [false; CASCADE_COUNT],
            redraw_all: false,
        }
    }

    /// Makes every cascade redraw in turn, as finer geometry has arrived.
    pub fn mark_stale(&mut self) {
        self.outdated = [true; CASCADE_COUNT];
    }

    /// Makes every cascade redraw next frame, as instances in them have changed: shown or
    /// hidden, or made movable or static. Taking turns would leave the far cascades showing the
    /// change late, as a shadow left behind or a missing one.
    pub fn mark_stale_now(&mut self) {
        self.mark_stale();
        self.redraw_all = true;
    }

    /// Finds the stale cascades and writes the views of those to redraw to slots from
    /// `first_slot`. Then finds the cascades that `moving`, the bounding spheres of movable
    /// instances, fall in, and writes their views to slots from `first_moving_slot`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        slots: &ViewSlots,
        first_slot: u32,
        first_moving_slot: u32,
        camera: &Camera,
        aspect: f32,
        sun_direction: Vec3,
        lod_threshold: f32,
        moving: &[(Vec3, f32)],
    ) -> ShadowWork {
        let fits: [(Mat4, Vec3, f32); CASCADE_COUNT] = std::array::from_fn(|cascade| {
            let near = if cascade == 0 {
                camera.near
            } else {
                SPLITS[cascade - 1]
            };
            fit_cascade(camera, aspect, sun_direction, near, SPLITS[cascade])
        });
        let stale: [bool; CASCADE_COUNT] = std::array::from_fn(|cascade| {
            self.outdated[cascade] || self.drawn[cascade] != Some((fits[cascade].0, lod_threshold))
        });
        let never_drawn: [bool; CASCADE_COUNT] =
            std::array::from_fn(|cascade| self.drawn[cascade].is_none());
        let mut due = choose_due(stale, never_drawn, &mut self.next_far_cascade);
        if std::mem::take(&mut self.redraw_all) {
            due = stale;
        }

        for cascade in (0..CASCADE_COUNT).filter(|&c| due[c]) {
            self.outdated[cascade] = false;
            let (view_proj, center, radius) = fits[cascade];
            self.drawn[cascade] = Some((view_proj, lod_threshold));
            self.extents[cascade] = (center, radius);
            self.data.view_proj[cascade] = view_proj;
            self.data.texel_size[cascade] = 2.0 * radius / RESOLUTION as f32;
            let view = PassView::orthographic(
                view_proj,
                center,
                Vec2::splat(RESOLUTION as f32),
                RESOLUTION as f32 / (2.0 * radius),
                lod_threshold,
            );
            slots.write(gpu, first_slot + cascade as u32, &view);
        }

        // Movable instances go through each cascade's view as it stands, drawn this frame or
        // before, so that the two maps line up.
        let moving_due: [bool; CASCADE_COUNT] = std::array::from_fn(|cascade| {
            let (center, radius) = self.extents[cascade];
            let view_proj = self.data.view_proj[cascade];
            let touched = self.drawn[cascade].is_some()
                && moving
                    .iter()
                    .any(|&sphere| cascade_touches(view_proj, radius, sphere));
            if touched {
                let view = PassView::orthographic(
                    view_proj,
                    center,
                    Vec2::splat(MOVING_RESOLUTION as f32),
                    MOVING_RESOLUTION as f32 / (2.0 * radius),
                    lod_threshold,
                )
                .drawing(InstanceFilter::Movable);
                slots.write(gpu, first_moving_slot + cascade as u32, &view);
            }
            touched
        });
        self.data.moving = Vec4::from_array(moving_due.map(|m| f32::from(u8::from(m))));

        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.data));
        ShadowWork {
            due,
            moving: moving_due,
        }
    }

    /// Culls and draws what `shadow_work` says is due: stale cascades, then movable instances.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        cull: &CullPipelines,
        scene: &GpuScene,
        cull_scene: &wgpu::BindGroup,
        work: &CullWork,
        raster: &RasterBindings,
        first_slot: u32,
        first_moving_slot: u32,
        shadow_work: &ShadowWork,
    ) {
        for cascade in (0..CASCADE_COUNT).filter(|&c| shadow_work.due[c]) {
            let slot = first_slot + cascade as u32;
            cull.record(encoder, scene, cull_scene, slot, work, CullPhase::First);

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.layer_views[cascade],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &raster.group, &[ViewSlots::offset(slot)]);
            pass.draw_indirect(work.args(), CullWork::draw_args_offset());
        }

        let mut cleared = false;
        for cascade in (0..CASCADE_COUNT).filter(|&c| shadow_work.moving[c]) {
            let slot = first_moving_slot + cascade as u32;
            cull.record(encoder, scene, cull_scene, slot, work, CullPhase::First);

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("moving shadows"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.moving_view,
                    depth_ops: Some(wgpu::Operations {
                        // The whole layer is cleared once a frame, before its first quarter.
                        load: if cleared {
                            wgpu::LoadOp::Load
                        } else {
                            wgpu::LoadOp::Clear(1.0)
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            cleared = true;
            let [x, y] = moving_quarter(cascade);
            let size = MOVING_RESOLUTION as f32;
            pass.set_viewport(x as f32 * size, y as f32 * size, size, size, 0.0, 1.0);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &raster.group, &[ViewSlots::offset(slot)]);
            pass.draw_indirect(work.args(), CullWork::draw_args_offset());
        }
    }

    /// Layout entries for sampling the maps: uniform, depth array, comparison sampler.
    pub fn add_to_layout(builder: LayoutBuilder) -> LayoutBuilder {
        builder.uniform().depth_texture_array().sampler_comparison()
    }
}

/// Which quarter of the movable layer a cascade has, as column and row. Twin of
/// `moving_quarter` in `lighting/sun_shadow.wgsl`.
fn moving_quarter(cascade: usize) -> [u32; 2] {
    [cascade as u32 % 2, cascade as u32 / 2]
}

/// Whether a sphere could cast a shadow into a cascade drawn through `view_proj`, an
/// orthographic view `radius` across each way from its centre, with depth reaching back
/// [`DEPTH_REACH`] towards the sun.
fn cascade_touches(view_proj: Mat4, radius: f32, (center, sphere_radius): (Vec3, f32)) -> bool {
    let clip = view_proj.project_point3(center);
    let across = sphere_radius / radius;
    let deep = sphere_radius / (DEPTH_REACH + radius * 2.0);
    clip.x.abs() <= 1.0 + across
        && clip.y.abs() <= 1.0 + across
        && clip.z >= -deep
        && clip.z <= 1.0 + deep
}

/// Picks what to redraw. Anything never drawn goes now, since it has no usable contents.
/// Otherwise the nearest cascade goes whenever stale, as errors there are the most visible,
/// and at most one stale far cascade joins it, which bounds the cost of a frame.
fn choose_due(
    stale: [bool; CASCADE_COUNT],
    never_drawn: [bool; CASCADE_COUNT],
    next_far: &mut usize,
) -> [bool; CASCADE_COUNT] {
    let mut due = never_drawn;
    due[0] |= stale[0];
    let far_count = CASCADE_COUNT - 1;
    for offset in 0..far_count {
        let cascade = (*next_far - 1 + offset) % far_count + 1;
        if stale[cascade] {
            due[cascade] = true;
            *next_far = cascade % far_count + 1;
            break;
        }
    }
    due
}

/// Fits an orthographic view around a slice of the camera frustum.
///
/// Bounding the slice with a sphere keeps the projection's size constant as the camera
/// rotates. Snapping its centre to whole shadow texels then keeps shadow edges from
/// crawling as the camera translates.
fn fit_cascade(camera: &Camera, aspect: f32, sun: Vec3, near: f32, far: f32) -> (Mat4, Vec3, f32) {
    let tan_y = (camera.fov_y * 0.5).tan();
    let k = (1.0 + aspect * aspect) * tan_y * tan_y;
    // Centre and radius of the sphere through the slice's corners, along the view axis.
    let center_distance = ((far + near) * 0.5 * (1.0 + k)).min(far);
    let radius = (((far - center_distance).powi(2) + far * far * k)
        .max((center_distance - near).powi(2) + near * near * k))
    .sqrt()
        * SNAP_PADDING;
    let center = camera.position + camera.forward() * center_distance;

    let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let light_view = Mat4::look_to_rh(Vec3::ZERO, -sun, up);

    let step = 2.0 * radius / RESOLUTION as f32 * SNAP_TEXELS;
    let in_light = light_view.transform_point3(center);
    let snapped = Vec3::new(
        (in_light.x / step).round() * step,
        (in_light.y / step).round() * step,
        // Along the light too: otherwise any camera motion changes the view, and every
        // frame counts as stale even though the map's contents would be the same.
        (in_light.z / step).round() * step,
    );
    let snapped_center = light_view.inverse().transform_point3(snapped);

    let eye = snapped_center + sun * DEPTH_REACH;
    let view = Mat4::look_to_rh(eye, -sun, up);
    let projection = Mat4::orthographic_rh(
        -radius,
        radius,
        -radius,
        radius,
        0.0,
        DEPTH_REACH + radius * 2.0,
    );
    (projection * view, snapped_center, radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> Camera {
        Camera {
            position: Vec3::new(3.0, 2.0, 1.0),
            yaw: 0.8,
            pitch: -0.2,
            ..Default::default()
        }
    }

    fn sun() -> Vec3 {
        Vec3::new(0.6, 0.4, 0.5).normalize()
    }

    #[test]
    fn cascade_contains_the_frustum_slice() {
        let camera = test_camera();
        let aspect = 16.0 / 9.0;
        let (view_proj, _, _) = fit_cascade(&camera, aspect, sun(), 5.0, 30.0);
        let tan_y = (camera.fov_y * 0.5).tan();
        let right = camera.right();
        let up = right.cross(camera.forward());
        for distance in [5.0_f32, 30.0] {
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                let corner = camera.position
                    + camera.forward() * distance
                    + right * sx * distance * tan_y * aspect
                    + up * sy * distance * tan_y;
                let clip = view_proj.project_point3(corner);
                assert!(
                    clip.x.abs() <= 1.0 && clip.y.abs() <= 1.0,
                    "corner outside: {clip}"
                );
                assert!(
                    (0.0..=1.0).contains(&clip.z),
                    "corner outside depth range: {clip}"
                );
            }
        }
    }

    #[test]
    fn small_moves_leave_the_cascade_where_it_is() {
        let aspect = 1.5;
        let mut camera = test_camera();
        let (first, _, _) = fit_cascade(&camera, aspect, sun(), 5.0, 30.0);
        let mut unchanged = 0;
        for _ in 0..50 {
            camera.position.x += 0.001;
            let (next, _, _) = fit_cascade(&camera, aspect, sun(), 5.0, 30.0);
            unchanged += u32::from(next == first);
        }
        assert!(
            unchanged >= 40,
            "only {unchanged} of 50 small moves kept the cascade"
        );
    }

    #[test]
    fn a_cascade_is_touched_by_what_stands_in_it_and_not_by_what_stands_aside() {
        let camera = test_camera();
        let (view_proj, center, radius) = fit_cascade(&camera, 1.5, sun(), 5.0, 30.0);
        assert!(cascade_touches(view_proj, radius, (center, 0.5)));
        // The light's own sideways axis, which runs along the cascade's x.
        let across = (-sun()).cross(Vec3::Y).normalize();
        // Out to one side, beyond the cascade's edge by more than its own radius.
        let aside = center + across * radius * 1.5;
        assert!(!cascade_touches(view_proj, radius, (aside, 1.0)));
        // As far out, but large enough to reach in.
        assert!(cascade_touches(view_proj, radius, (aside, radius * 0.6)));
        // Up towards the sun, a caster can be far beyond the cascade and still shade it.
        assert!(cascade_touches(
            view_proj,
            radius,
            (center + sun() * 500.0, 1.0)
        ));
        // Behind the cascade, away from the sun, it can't.
        assert!(!cascade_touches(
            view_proj,
            radius,
            (center - sun() * (radius * 2.0 + 5.0), 1.0)
        ));
    }

    #[test]
    fn the_uniform_lays_out_as_its_shader_twin() {
        crate::shaders::assert_twin!(
            "lighting/deferred.wgsl",
            "Shadows",
            ShadowUniform,
            [view_proj, splits, texel_size, moving]
        );
    }

    #[test]
    fn each_cascade_has_its_own_quarter() {
        let quarters: Vec<_> = (0..CASCADE_COUNT).map(moving_quarter).collect();
        assert_eq!(quarters, vec![[0, 0], [1, 0], [0, 1], [1, 1]]);
    }

    #[test]
    fn nothing_is_redrawn_when_nothing_is_stale() {
        let mut next = 1;
        assert_eq!(choose_due([false; 4], [false; 4], &mut next), [false; 4]);
    }

    #[test]
    fn never_drawn_cascades_all_go_at_once() {
        let mut next = 1;
        assert_eq!(choose_due([true; 4], [true; 4], &mut next), [true; 4]);
    }

    #[test]
    fn stale_far_cascades_take_turns() {
        let mut next = 1;
        let never = [false; 4];
        assert_eq!(
            choose_due([true; 4], never, &mut next),
            [true, true, false, false]
        );
        assert_eq!(
            choose_due([true; 4], never, &mut next),
            [true, false, true, false]
        );
        assert_eq!(
            choose_due([true; 4], never, &mut next),
            [true, false, false, true]
        );
        assert_eq!(
            choose_due([true; 4], never, &mut next),
            [true, true, false, false]
        );
    }

    #[test]
    fn a_single_stale_far_cascade_is_found_whatever_the_turn() {
        let mut next = 2;
        let stale = [false, true, false, false];
        assert_eq!(choose_due(stale, [false; 4], &mut next), stale);
    }

    #[test]
    fn cascade_size_does_not_change_with_rotation() {
        let aspect = 1.5;
        let mut camera = test_camera();
        let (_, _, a) = fit_cascade(&camera, aspect, sun(), 5.0, 30.0);
        camera.yaw += 1.3;
        camera.pitch += 0.4;
        let (_, _, b) = fit_cascade(&camera, aspect, sun(), 5.0, 30.0);
        assert!((a - b).abs() < 1.0e-3);
    }

    #[test]
    fn small_camera_moves_shift_shadows_by_whole_texels() {
        let aspect = 1.5;
        let mut camera = test_camera();
        let (first, _, _) = fit_cascade(&camera, aspect, sun(), 5.0, 30.0);
        camera.position += Vec3::new(0.013, 0.0, 0.0);
        let (second, _, _) = fit_cascade(&camera, aspect, sun(), 5.0, 30.0);
        let point = Vec3::new(10.0, 3.0, -4.0);
        let shift =
            (second.project_point3(point) - first.project_point3(point)) * 0.5 * RESOLUTION as f32;
        for value in [shift.x, shift.y] {
            assert!(
                (value - value.round()).abs() < 0.02,
                "shift of {value} texels"
            );
        }
    }
}
