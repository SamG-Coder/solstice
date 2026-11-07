//! The deferred lighting pass and the bind group shared by all lighting passes.

use crate::clouds::{CloudBindings, CloudSky};
use crate::gi::{GiVolume, ScreenGiBindings};
use crate::gpu::{groups_for, Gpu, LayoutBuilder};
use crate::scene::GpuScene;
use crate::shaders;
use crate::shadow::ShadowMaps;
use crate::sky::Sky;
use crate::vg::GBuffer;

const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Creates the layout of the shared lighting bind group. See `common/environment.wgsl`.
pub fn environment_layout(gpu: &Gpu) -> wgpu::BindGroupLayout {
    LayoutBuilder::new()
        .uniform()
        .uniform()
        .texture_3d()
        .sampler()
        .texture()
        .texture()
        .sampler()
        .texture()
        .texture()
        .uniform()
        .build(gpu, "environment")
}

/// Creates the shared lighting bind group, seeing the clouds through `clouds`.
pub fn environment_group(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    frame: &wgpu::Buffer,
    volume: &GiVolume,
    sky: &Sky,
    clouds: CloudSky,
) -> wgpu::BindGroup {
    let clamp_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("environment clamp"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let repeat_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("environment repeat"),
        address_mode_u: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    gpu.bind_group(
        "environment",
        layout,
        &[
            frame.as_entire_binding(),
            volume.uniform.as_entire_binding(),
            wgpu::BindingResource::TextureView(&volume.atlas_view),
            wgpu::BindingResource::Sampler(&clamp_sampler),
            wgpu::BindingResource::TextureView(&sky.transmittance.view),
            wgpu::BindingResource::TextureView(&sky.sky_view.view),
            wgpu::BindingResource::Sampler(&repeat_sampler),
            wgpu::BindingResource::TextureView(clouds.sky),
            wgpu::BindingResource::TextureView(clouds.shadow),
            clouds.params.as_entire_binding(),
        ],
    )
}

/// The pass that combines all lighting into the HDR scene colour.
pub struct DeferredLighting {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl DeferredLighting {
    pub fn new(gpu: &Gpu, environment: &wgpu::BindGroupLayout) -> Self {
        let builder = LayoutBuilder::new()
            .texture()
            .texture()
            .texture()
            .texture_unfilterable()
            .texture()
            .texture()
            .storage_texture(HDR_FORMAT);
        let layout = ShadowMaps::add_to_layout(builder)
            .storage_read()
            .texture()
            .build(gpu, "deferred");
        let module = shaders::module(&gpu.device, "lighting/deferred.wgsl");
        Self {
            pipeline: gpu.compute_pipeline("deferred", &module, "main", &[&layout, environment]),
            layout,
        }
    }

    /// One bind group per frame parity, reading that frame's GI output and clouds.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_groups(
        &self,
        gpu: &Gpu,
        gbuffer: &GBuffer,
        gi: &ScreenGiBindings,
        hdr: &wgpu::TextureView,
        shadows: &ShadowMaps,
        scene: &GpuScene,
        clouds: &CloudBindings,
    ) -> [wgpu::BindGroup; 2] {
        let tex = |view| wgpu::BindingResource::TextureView(view);
        let group = |parity: usize| {
            gpu.bind_group(
                "deferred",
                &self.layout,
                &[
                    tex(&gbuffer.albedo.view),
                    tex(&gbuffer.normal.view),
                    tex(&gbuffer.emissive.view),
                    tex(&gbuffer.depth.view),
                    tex(&gi.diffuse[parity].view),
                    tex(&gi.reflections[parity].view),
                    tex(hdr),
                    shadows.uniform.as_entire_binding(),
                    tex(&shadows.array_view),
                    wgpu::BindingResource::Sampler(&shadows.comparison_sampler),
                    scene.lights.as_entire_binding(),
                    tex(&clouds.screen[parity].view),
                ],
            )
        };
        [group(0), group(1)]
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        environment: &wgpu::BindGroup,
        size: (u32, u32),
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("deferred"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.set_bind_group(1, environment, &[]);
        pass.dispatch_workgroups(groups_for(size.0, 8), groups_for(size.1, 8), 1);
    }
}
