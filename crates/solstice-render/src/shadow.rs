//! Cascaded sun shadow maps, drawn through the same cluster culling as the main view.
//!
//! Shadow maps are the most expensive pass on small GPUs, and the scene is static, so a
//! cascade is only redrawn when its view changes. Views move in coarse steps to make that
//! rare: each cascade covers slightly more than it must, so it can stay put while the
//! camera moves. When several cascades are stale, the far ones take turns.

use crate::frame::Camera;
use crate::gpu::{Gpu, LayoutBuilder};
use crate::scene::GpuScene;
use crate::vg::{
    CullPhase, CullPipelines, CullWork, PassView, RasterBindings, Rasterizer, ViewSlots,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

pub const CASCADE_COUNT: usize = 4;
const RESOLUTION: u32 = 2048;
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
/// Where along a cascade receivers begin to dither into the next one, as a fraction of its
/// span. Twin of the blend in `lighting/sun_shadow.wgsl`.
const BLEND_START: f32 = 0.85;
/// How far the camera may turn, on each side, before a cascade's receiver planes are redrawn:
/// a little more than the turn that moves its fitted sphere a snap step and refits it anyway,
/// so that turning seldom costs a redraw of its own.
const ROTATION_MARGIN: f32 = 0.1;

/// How far the shadow lookup reaches from the receiver it is for, in metres, for a cascade
/// with texels `texel` across, as `lighting/sun_shadow.wgsl` samples: an offset along the
/// normal of 1.2 + 0.8 × a slope of up to 8 texels, then a filter disc 1.5 texels across, and
/// a texel of bilinear comparison. The disc is held to 1.8 cm at least, which in the nearest
/// cascade is more than a texel.
fn lookup_reach(texel: f32) -> f32 {
    texel * (7.6 + 1.5 * (0.018 / texel).max(1.0) + 1.0)
}

/// What a cascade was drawn with, to tell when it has gone stale.
#[derive(Clone, Copy, PartialEq)]
struct Drawn {
    view_proj: Mat4,
    lod_threshold: f32,
    eye_lod_scale: f32,
    /// The receiver planes its casters were culled with, `receiver_count` of them in use.
    receivers: [Vec4; 6],
    receiver_count: usize,
    /// How far receivers may lie outside those planes and still find their casters.
    slack: f32,
}

impl Drawn {
    /// Whether the receivers in the frustum slice with these `corners` all have their casters
    /// in this cascade.
    fn covers(&self, corners: &[Vec3; 8]) -> bool {
        self.receivers[..self.receiver_count].iter().all(|plane| {
            corners
                .iter()
                .all(|&corner| plane.truncate().dot(corner) + plane.w - self.slack >= 0.0)
        })
    }
}

/// Twin of `Shadows` in `lighting/deferred.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowUniform {
    view_proj: [Mat4; CASCADE_COUNT],
    splits: Vec4,
    texel_size: Vec4,
}

/// Shadow map resources and passes.
pub struct ShadowMaps {
    pub array_view: wgpu::TextureView,
    pub comparison_sampler: wgpu::Sampler,
    pub uniform: wgpu::Buffer,
    layer_views: Vec<wgpu::TextureView>,
    pipeline: wgpu::RenderPipeline,
    data: ShadowUniform,
    /// What each cascade last drew, to tell when it has gone stale.
    drawn: [Option<Drawn>; CASCADE_COUNT],
    next_far_cascade: usize,
    /// Cascades that must redraw because the geometry on the GPU has improved.
    outdated: [bool; CASCADE_COUNT],
}

impl ShadowMaps {
    pub fn new(gpu: &Gpu, rasterizer: &Rasterizer) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: RESOLUTION,
                height: RESOLUTION,
                depth_or_array_layers: CASCADE_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let layer_views = (0..CASCADE_COUNT as u32)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
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
            pipeline,
            data: ShadowUniform {
                view_proj: [Mat4::IDENTITY; CASCADE_COUNT],
                splits: Vec4::from_array(SPLITS),
                texel_size: Vec4::ONE,
            },
            drawn: [None; CASCADE_COUNT],
            next_far_cascade: 1,
            outdated: [false; CASCADE_COUNT],
        }
    }

    /// Makes every cascade redraw in turn, as finer geometry has arrived.
    pub fn mark_stale(&mut self) {
        self.outdated = [true; CASCADE_COUNT];
    }

    /// Finds the stale cascades and writes the views of those to redraw to slots from
    /// `first_slot`. Returns which cascades are due.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        slots: &ViewSlots,
        first_slot: u32,
        camera: &Camera,
        aspect: f32,
        sun_direction: Vec3,
        lod_threshold: f32,
        eye_lod_scale: f32,
    ) -> [bool; CASCADE_COUNT] {
        let fits: [(Mat4, Vec3, f32); CASCADE_COUNT] = std::array::from_fn(|cascade| {
            let near = if cascade == 0 {
                camera.near
            } else {
                SPLITS[cascade - 1]
            };
            fit_cascade(camera, aspect, sun_direction, near, SPLITS[cascade])
        });
        // Where each cascade's receivers begin: where the previous cascade's start to dither
        // into it, since the lookup blends across the last of each cascade.
        let nearest: [f32; CASCADE_COUNT] = std::array::from_fn(|cascade| match cascade {
            0 => camera.near,
            1 => SPLITS[0] * BLEND_START,
            _ => SPLITS[cascade - 2] + (SPLITS[cascade - 1] - SPLITS[cascade - 2]) * BLEND_START,
        });
        let stale: [bool; CASCADE_COUNT] = std::array::from_fn(|cascade| {
            self.outdated[cascade]
                || !self.drawn[cascade].is_some_and(|drawn| {
                    drawn.view_proj == fits[cascade].0
                        && drawn.lod_threshold == lod_threshold
                        && drawn.eye_lod_scale == eye_lod_scale
                        && drawn.covers(&slice_corners(
                            camera,
                            aspect,
                            nearest[cascade],
                            SPLITS[cascade],
                        ))
                })
        });
        let never_drawn: [bool; CASCADE_COUNT] =
            std::array::from_fn(|cascade| self.drawn[cascade].is_none());
        let due = choose_due(stale, never_drawn, &mut self.next_far_cascade);

        for cascade in (0..CASCADE_COUNT).filter(|&c| due[c]) {
            self.outdated[cascade] = false;
            let (view_proj, center, radius) = fits[cascade];
            let texel = 2.0 * radius / RESOLUTION as f32;
            self.data.view_proj[cascade] = view_proj;
            self.data.texel_size[cascade] = texel;
            // Only casters whose shadow can fall on this cascade's slice of the view are drawn:
            // the sphere the cascade is fitted to covers far more than the slice, especially
            // beside and behind the camera. The slice is widened by the turn and the step the
            // camera can make before the cascade is refitted, so that the cached map serves
            // every pose it is kept for, and by the reach of the shadow lookup.
            let slack = lookup_reach(texel);
            let planes = receiver_planes(
                camera,
                aspect,
                nearest[cascade],
                SPLITS[cascade],
                texel * SNAP_TEXELS + slack,
            );
            let view = PassView::orthographic(
                view_proj,
                center,
                Vec2::splat(RESOLUTION as f32),
                RESOLUTION as f32 / (2.0 * radius),
                lod_threshold,
            )
            // Detail the camera could not see from where the shadows land is left out.
            .seen_from(
                camera.position,
                sun_direction,
                eye_lod_scale,
                nearest[cascade],
            )
            .shadowing(&planes, sun_direction);
            self.drawn[cascade] = Some(Drawn {
                view_proj,
                lod_threshold,
                eye_lod_scale,
                receivers: view.receiver_planes,
                receiver_count: view.receiver_plane_count as usize,
                slack,
            });
            slots.write(gpu, first_slot + cascade as u32, &view);
        }
        gpu.queue
            .write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.data));
        due
    }

    /// Culls and draws the cascades marked in `due`.
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
        due: &[bool; CASCADE_COUNT],
    ) {
        for cascade in (0..CASCADE_COUNT).filter(|&c| due[c]) {
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
    }

    /// Layout entries for sampling the maps: uniform, depth array, comparison sampler.
    pub fn add_to_layout(builder: LayoutBuilder) -> LayoutBuilder {
        builder.uniform().depth_texture_array().sampler_comparison()
    }
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

/// The camera with its field of view widened by `margin` radians on every side.
fn widened(camera: &Camera, margin: f32) -> Camera {
    Camera {
        fov_y: camera.fov_y + 2.0 * margin,
        ..*camera
    }
}

/// The aspect ratio that widens the horizontal field of view by `margin` on each side too.
fn widened_aspect(camera: &Camera, aspect: f32, margin: f32) -> f32 {
    let half_width = (aspect * (camera.fov_y * 0.5).tan()).atan() + margin;
    half_width.tan() / (camera.fov_y * 0.5 + margin).tan()
}

/// How far along the camera's axis the nearest point at `distance` from it lies: the frustum's
/// corners are the furthest off the axis.
fn corner_depth(camera: &Camera, aspect: f32, distance: f32) -> f32 {
    let tan_y = (camera.fov_y * 0.5).tan();
    distance / (1.0 + (1.0 + aspect * aspect) * tan_y * tan_y).sqrt()
}

/// The planes of the camera's frustum holding every point between `near` and `far` from the
/// camera, facing inwards, each moved outwards by `padding` metres. Cascades are chosen by
/// distance from the camera, not depth along its axis, so the near plane is drawn back to
/// where the nearest point at that distance, in the frustum's corner, lies in depth.
fn slice_planes(camera: &Camera, aspect: f32, near: f32, far: f32, padding: f32) -> [Vec4; 6] {
    // A finite, forward-depth projection: `frustum_planes` reads a far plane from it too.
    let projection = Mat4::perspective_rh(
        camera.fov_y,
        aspect,
        corner_depth(camera, aspect, near),
        far,
    );
    PassView::frustum_planes(projection * camera.view()).map(|plane| plane + Vec4::W * padding)
}

/// The planes of the slice of the frustum between `near` and `far`, widened to hold the slice
/// for every turn of the camera up to `ROTATION_MARGIN` on each side, and moved out by
/// `padding` metres besides. Turning widens the slice's sides and swings its flat far face
/// about the camera, carrying its corners further along the axis by their distance from the
/// axis times the turn. The near plane needs no such allowance: widening the frustum draws
/// it back to the wider frustum's corner depth, nearer than any corner the turn can bring.
fn receiver_planes(camera: &Camera, aspect: f32, near: f32, far: f32, padding: f32) -> [Vec4; 6] {
    let mut planes = slice_planes(
        &widened(camera, ROTATION_MARGIN),
        widened_aspect(camera, aspect, ROTATION_MARGIN),
        near,
        far,
        padding,
    );
    let tan_y = (camera.fov_y * 0.5).tan();
    let off_axis = tan_y * (1.0 + aspect * aspect).sqrt();
    // `frustum_planes` puts the far plane last.
    planes[5].w += far * (off_axis * ROTATION_MARGIN.sin() + 1.0 - ROTATION_MARGIN.cos());
    planes
}

/// The eight corners of that same slice of the frustum.
fn slice_corners(camera: &Camera, aspect: f32, near: f32, far: f32) -> [Vec3; 8] {
    let tan_y = (camera.fov_y * 0.5).tan();
    let forward = camera.forward();
    let right = camera.right();
    let up = right.cross(forward);
    let depths = [corner_depth(camera, aspect, near), far];
    std::array::from_fn(|i| {
        let depth = depths[i / 4];
        let sx = if i % 2 == 0 { -1.0 } else { 1.0 };
        let sy = if (i / 2) % 2 == 0 { -1.0 } else { 1.0 };
        camera.position
            + forward * depth
            + right * (sx * depth * tan_y * aspect)
            + up * (sy * depth * tan_y)
    })
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

    /// Points at `distance` from the camera, across its field of view: the centre, the four
    /// corners, and a few between.
    fn points_at(camera: &Camera, aspect: f32, distance: f32) -> Vec<Vec3> {
        let tan_y = (camera.fov_y * 0.5).tan();
        let right = camera.right();
        let up = right.cross(camera.forward());
        let mut points = Vec::new();
        for (sx, sy) in [
            (0.0, 0.0),
            (-1.0, -1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
            (0.5, -0.3),
            (-0.7, 0.9),
        ] {
            let direction =
                (camera.forward() + right * sx * tan_y * aspect + up * sy * tan_y).normalize();
            points.push(camera.position + direction * distance);
        }
        points
    }

    fn inside(planes: &[Vec4], point: Vec3) -> bool {
        planes
            .iter()
            .all(|plane| plane.truncate().dot(point) + plane.w >= -1.0e-3)
    }

    #[test]
    fn a_slice_holds_every_point_between_its_distances() {
        let camera = test_camera();
        let aspect = 16.0 / 9.0;
        let planes = slice_planes(&camera, aspect, 38.0, 120.0, 0.0);
        for distance in [38.0, 50.0, 100.0, 119.9] {
            for point in points_at(&camera, aspect, distance) {
                assert!(inside(&planes, point), "{point} at {distance} is outside");
            }
        }
        // Nearer than the slice, in its corner, is outside; so is beyond it.
        let corner = points_at(&camera, aspect, 20.0)[1];
        assert!(!inside(&planes, corner), "{corner} at 20 m is inside");
        let beyond = points_at(&camera, aspect, 130.0)[0];
        assert!(!inside(&planes, beyond), "{beyond} at 130 m is inside");
    }

    #[test]
    fn padding_moves_every_plane_outwards() {
        let camera = test_camera();
        let planes = slice_planes(&camera, 1.5, 38.0, 120.0, 0.0);
        let padded = slice_planes(&camera, 1.5, 38.0, 120.0, 3.0);
        for (plane, wide) in planes.iter().zip(&padded) {
            assert_eq!(plane.truncate(), wide.truncate());
            assert!((wide.w - plane.w - 3.0).abs() < 1.0e-4);
        }
        // Just beyond the far end is now inside.
        let beyond = points_at(&camera, 1.5, 122.0)[0];
        assert!(!inside(&planes, beyond) && inside(&padded, beyond));
    }

    #[test]
    fn every_caster_that_can_shadow_the_slice_passes_the_receiver_planes() {
        let camera = test_camera();
        let aspect = 1.5;
        let planes = slice_planes(&camera, aspect, 38.0, 120.0, 0.0);
        for sun in [sun(), Vec3::new(-0.7, 0.3, 0.1).normalize(), Vec3::Y] {
            let view = PassView::orthographic(Mat4::IDENTITY, Vec3::ZERO, Vec2::ONE, 1.0, 1.0)
                .shadowing(&planes, sun);
            let kept = &view.receiver_planes[..view.receiver_plane_count as usize];
            assert!(
                !kept.is_empty() && kept.len() < 6,
                "{} planes kept",
                kept.len()
            );
            // A caster anywhere between a receiver in the slice and the sun passes.
            for distance in [38.0, 80.0, 119.0] {
                for receiver in points_at(&camera, aspect, distance) {
                    for t in [0.0, 1.0, 50.0, 900.0] {
                        let caster = receiver + sun * t;
                        assert!(
                            inside(kept, caster),
                            "{caster} fails, {t} m towards the sun"
                        );
                    }
                }
            }
            // One well beyond the slice, away from the sun, cannot shadow it and does not pass.
            let shadowed_side = points_at(&camera, aspect, 80.0)[0] - sun * 400.0;
            assert!(
                !inside(kept, shadowed_side),
                "{shadowed_side} passes for sun {sun}"
            );
        }
    }

    /// What a far cascade would be drawn with for this camera and light, as `prepare` builds it.
    fn drawn_for(camera: &Camera, aspect: f32, light: Vec3) -> Drawn {
        let (near, far) = (SPLITS[1] + (SPLITS[2] - SPLITS[1]) * BLEND_START, SPLITS[3]);
        let (view_proj, _, radius) = fit_cascade(camera, aspect, light, SPLITS[2], far);
        let texel = 2.0 * radius / RESOLUTION as f32;
        let slack = lookup_reach(texel);
        let planes = receiver_planes(camera, aspect, near, far, texel * SNAP_TEXELS + slack);
        let view = PassView::orthographic(view_proj, Vec3::ZERO, Vec2::ONE, 1.0, 1.0)
            .shadowing(&planes, light);
        Drawn {
            view_proj,
            lod_threshold: 1.0,
            eye_lod_scale: 1.0,
            receivers: view.receiver_planes,
            receiver_count: view.receiver_plane_count as usize,
            slack,
        }
    }

    fn covers_now(drawn: &Drawn, camera: &Camera, aspect: f32) -> bool {
        let near = SPLITS[1] + (SPLITS[2] - SPLITS[1]) * BLEND_START;
        drawn.covers(&slice_corners(camera, aspect, near, SPLITS[3]))
    }

    #[test]
    fn a_cascade_covers_the_camera_turning_within_its_margin_and_not_beyond() {
        let aspect = 16.0 / 9.0;
        // Under the test sun only the far plane faces the light, so turning never leaves the
        // cascade's volume; a level camera under a light from overhead keeps the side, far
        // and bottom planes, so turning too far shows.
        let level = Camera {
            pitch: 0.0,
            ..test_camera()
        };
        for (camera, light) in [(test_camera(), sun()), (level, Vec3::Y)] {
            let drawn = drawn_for(&camera, aspect, light);
            assert!(
                covers_now(&drawn, &camera, aspect),
                "not even where it was drawn"
            );
            for (yaw, pitch) in [(0.095, 0.0), (-0.095, 0.0), (0.0, 0.095), (0.06, -0.06)] {
                let turned = Camera {
                    yaw: camera.yaw + yaw,
                    pitch: camera.pitch + pitch,
                    ..camera
                };
                assert!(
                    covers_now(&drawn, &turned, aspect),
                    "turned by {yaw}, {pitch}"
                );
            }
        }
        let drawn = drawn_for(&level, aspect, Vec3::Y);
        for (yaw, pitch) in [(0.3, 0.0), (-0.3, 0.0), (0.0, -0.3)] {
            let turned = Camera {
                yaw: level.yaw + yaw,
                pitch: level.pitch + pitch,
                ..level
            };
            assert!(
                !covers_now(&drawn, &turned, aspect),
                "still covered after {yaw}, {pitch}"
            );
        }
    }

    #[test]
    fn the_widened_planes_alone_hold_the_slice_through_a_turn_of_the_margin() {
        let aspect = 16.0 / 9.0;
        let camera = test_camera();
        let near = SPLITS[1] + (SPLITS[2] - SPLITS[1]) * BLEND_START;
        // No padding: every plane, near and far included, must hold the turned slice by the
        // widening alone.
        let drawn = Drawn {
            view_proj: Mat4::IDENTITY,
            lod_threshold: 1.0,
            eye_lod_scale: 1.0,
            receivers: receiver_planes(&camera, aspect, near, SPLITS[3], 0.0),
            receiver_count: 6,
            slack: 0.0,
        };
        for (yaw, pitch) in [
            (0.0, 0.0),
            (0.095, 0.0),
            (-0.095, 0.0),
            (0.0, 0.095),
            (0.0, -0.095),
        ] {
            let turned = Camera {
                yaw: camera.yaw + yaw,
                pitch: camera.pitch + pitch,
                ..camera
            };
            assert!(
                drawn.covers(&slice_corners(&turned, aspect, near, SPLITS[3])),
                "turned by {yaw}, {pitch}"
            );
        }
    }

    #[test]
    fn a_cascade_covers_the_camera_moving_less_than_a_snap_step_and_not_further() {
        let aspect = 1.5;
        // A level camera under a light from straight overhead: then the side, far and bottom
        // planes all face the light and are kept, so moving past any of them shows.
        let camera = Camera {
            pitch: 0.0,
            ..test_camera()
        };
        let drawn = drawn_for(&camera, aspect, Vec3::Y);
        let (_, _, radius) = fit_cascade(&camera, aspect, Vec3::Y, SPLITS[2], SPLITS[3]);
        let step = 2.0 * radius / RESOLUTION as f32 * SNAP_TEXELS;
        for direction in [
            camera.right(),
            -camera.right(),
            camera.forward(),
            Vec3::NEG_Y,
        ] {
            let moved = Camera {
                position: camera.position + direction * step * 0.9,
                ..camera
            };
            assert!(
                covers_now(&drawn, &moved, aspect),
                "moved {direction} by a step"
            );
            let far = Camera {
                position: camera.position + direction * step * 6.0,
                ..camera
            };
            assert!(
                !covers_now(&drawn, &far, aspect),
                "still covered {direction} by six"
            );
        }
    }

    #[test]
    fn the_lookup_reach_grows_with_the_texel_but_never_falls_below_the_filter_floor() {
        // Far cascades: about ten texels. The nearest, with 1.5 cm texels, keeps the 1.8 cm
        // disc, so more than ten.
        assert!((lookup_reach(0.5) / 0.5 - 10.1).abs() < 1.0e-4);
        assert!(lookup_reach(0.015) / 0.015 > 10.3);
        assert!(lookup_reach(0.2) > lookup_reach(0.1));
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
