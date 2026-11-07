//! Cascaded sun shadow maps, drawn through the same cluster culling as the main view.
//!
//! Only the nearest cascade updates every frame. The rest take turns, which is invisible
//! because each cascade's position is snapped to whole texels and so does not shimmer.

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
    next_far_cascade: usize,
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
                    cull_mode: Some(wgpu::Face::Back),
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
            next_far_cascade: 1,
        }
    }

    /// Picks the cascades to refresh and writes their views to slots from `first_slot`.
    /// Returns which cascades are due. On the first frame, that is all of them.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        gpu: &Gpu,
        slots: &ViewSlots,
        first_slot: u32,
        camera: &Camera,
        aspect: f32,
        sun_direction: Vec3,
        first_frame: bool,
    ) -> [bool; CASCADE_COUNT] {
        let mut due = [first_frame; CASCADE_COUNT];
        due[0] = true;
        if !first_frame {
            due[self.next_far_cascade] = true;
            self.next_far_cascade = self.next_far_cascade % (CASCADE_COUNT - 1) + 1;
        }

        for cascade in 0..CASCADE_COUNT {
            if !due[cascade] {
                continue;
            }
            let near = if cascade == 0 {
                camera.near
            } else {
                SPLITS[cascade - 1]
            };
            let far = SPLITS[cascade];
            let (view_proj, center, radius) = fit_cascade(camera, aspect, sun_direction, near, far);
            self.data.view_proj[cascade] = view_proj;
            self.data.texel_size[cascade] = 2.0 * radius / RESOLUTION as f32;

            let view = PassView::orthographic(
                view_proj,
                center,
                Vec2::splat(RESOLUTION as f32),
                RESOLUTION as f32 / (2.0 * radius),
                1.0,
            );
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
        * 1.02;
    let center = camera.position + camera.forward() * center_distance;

    let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
    let light_view = Mat4::look_to_rh(Vec3::ZERO, -sun, up);

    let texel = 2.0 * radius / RESOLUTION as f32;
    let in_light = light_view.transform_point3(center);
    let snapped = Vec3::new(
        (in_light.x / texel).round() * texel,
        (in_light.y / texel).round() * texel,
        in_light.z,
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
