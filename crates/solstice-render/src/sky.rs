//! Atmosphere lookup tables. See `shaders/sky/luts.wgsl`.

use crate::gpu::{groups_for, Gpu, LayoutBuilder, Target};
use crate::shaders;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
const MULTI_SCATTER_SIZE: (u32, u32) = (32, 32);
const SKY_VIEW_SIZE: (u32, u32) = (256, 144);

/// Height of the viewer in kilometres above the planet centre at sea level.
const GROUND_RADIUS_KM: f32 = 6360.0;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkyParams {
    sun_direction: Vec3,
    viewer_radius: f32,
}

/// The sky's precomputed data and the passes that fill it.
pub struct Sky {
    pub transmittance: Target,
    pub sky_view: Target,
    params: wgpu::Buffer,
    build_transmittance: (wgpu::ComputePipeline, wgpu::BindGroup),
    build_multi_scatter: (wgpu::ComputePipeline, wgpu::BindGroup),
    build_sky_view: (wgpu::ComputePipeline, wgpu::BindGroup),
    static_tables_built: bool,
}

impl Sky {
    pub fn new(gpu: &Gpu) -> Self {
        let layout = LayoutBuilder::new()
            .uniform()
            .texture()
            .texture()
            .sampler()
            .storage_texture(FORMAT)
            .build(gpu, "sky");
        let module = shaders::module(&gpu.device, "sky/luts.wgsl");
        let params = gpu.uniform_buffer("sky params", std::mem::size_of::<SkyParams>() as u64);
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let none = wgpu::TextureUsages::empty();
        let transmittance = Target::new(gpu, "sky transmittance", TRANSMITTANCE_SIZE, FORMAT, none);
        let multi_scatter = Target::new(gpu, "sky multi scatter", MULTI_SCATTER_SIZE, FORMAT, none);
        let sky_view = Target::new(gpu, "sky view", SKY_VIEW_SIZE, FORMAT, none);
        // Bind groups must be complete, but no entry point reads the table it is writing.
        let placeholder = Target::new(gpu, "sky placeholder", (1, 1), FORMAT, none);

        let pipeline = |entry: &str, input_a: &Target, input_b: &Target, output: &Target| {
            let group = gpu.bind_group(
                entry,
                &layout,
                &[
                    params.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&input_a.view),
                    wgpu::BindingResource::TextureView(&input_b.view),
                    wgpu::BindingResource::Sampler(&sampler),
                    wgpu::BindingResource::TextureView(&output.view),
                ],
            );
            (
                gpu.compute_pipeline(entry, &module, entry, &[&layout]),
                group,
            )
        };

        Self {
            build_transmittance: pipeline(
                "build_transmittance",
                &placeholder,
                &placeholder,
                &transmittance,
            ),
            build_multi_scatter: pipeline(
                "build_multi_scatter",
                &transmittance,
                &placeholder,
                &multi_scatter,
            ),
            build_sky_view: pipeline("build_sky_view", &transmittance, &multi_scatter, &sky_view),
            transmittance,
            sky_view,
            params,
            static_tables_built: false,
        }
    }

    /// Updates the tables for the current sun and viewer height.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        sun_direction: Vec3,
        viewer_height_m: f32,
    ) {
        let params = SkyParams {
            sun_direction,
            viewer_radius: GROUND_RADIUS_KM + viewer_height_m.max(1.0) * 0.001,
        };
        gpu.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sky"),
            timestamp_writes: None,
        });
        let mut run = |(pipeline, group): &(wgpu::ComputePipeline, wgpu::BindGroup),
                       size: (u32, u32)| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups_for(size.0, 8), groups_for(size.1, 8), 1);
        };
        if !self.static_tables_built {
            run(&self.build_transmittance, TRANSMITTANCE_SIZE);
            run(&self.build_multi_scatter, MULTI_SCATTER_SIZE);
            self.static_tables_built = true;
        }
        run(&self.build_sky_view, SKY_VIEW_SIZE);
    }
}
