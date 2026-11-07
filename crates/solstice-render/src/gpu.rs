//! Device acquisition and small helpers that remove wgpu boilerplate.

use std::fmt;

/// Why the GPU could not be initialised.
#[derive(Debug)]
pub enum GpuError {
    NoAdapter(String),
    Device(wgpu::RequestDeviceError),
    /// The adapter lacks something this renderer cannot work without.
    Unsupported(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(reason) => write!(f, "no suitable GPU adapter: {reason}"),
            Self::Device(error) => write!(f, "could not create a GPU device: {error}"),
            Self::Unsupported(what) => write!(f, "this GPU is not supported: {what}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// Storage buffers the heaviest pass binds in a single shader stage.
const REQUIRED_STORAGE_BUFFERS: u32 = 8;
/// Storage textures the G-buffer resolve binds in a single shader stage.
const REQUIRED_STORAGE_TEXTURES: u32 = 5;

/// The device and queue, plus what was learned about them.
pub struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
    /// Whether scene textures may stay in BC formats. Without this they are unpacked on load.
    pub block_compressed_textures: bool,
}

impl Gpu {
    /// Requests a device, asking for the adapter's full limits.
    ///
    /// The WebGPU default limits are far below what browsers actually expose on
    /// desktop GPUs, and this renderer needs large storage buffers.
    pub async fn request(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, GpuError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: surface,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|error| GpuError::NoAdapter(error.to_string()))?;

        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < REQUIRED_STORAGE_BUFFERS {
            return Err(GpuError::Unsupported(format!(
                "needs {REQUIRED_STORAGE_BUFFERS} storage buffers per stage, has {}",
                limits.max_storage_buffers_per_shader_stage
            )));
        }
        if limits.max_storage_textures_per_shader_stage < REQUIRED_STORAGE_TEXTURES {
            return Err(GpuError::Unsupported(format!(
                "needs {REQUIRED_STORAGE_TEXTURES} storage textures per stage, has {}",
                limits.max_storage_textures_per_shader_stage
            )));
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("solstice"),
                // All optional. Timestamps let the profiler time passes where the platform
                // allows it, and BC textures save most of the memory that scenes' textures use.
                required_features: adapter.features()
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                        | wgpu::Features::TEXTURE_COMPRESSION_BC),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(GpuError::Device)?;

        let block_compressed_textures = device
            .features()
            .contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        let adapter_info = adapter.get_info();
        log::info!("using {} ({:?})", adapter_info.name, adapter_info.backend);
        Ok(Self {
            adapter,
            device,
            queue,
            adapter_info,
            limits,
            block_compressed_textures,
        })
    }

    /// Creates a compute pipeline.
    pub fn compute_pipeline(
        &self,
        label: &str,
        module: &wgpu::ShaderModule,
        entry_point: &str,
        layouts: &[&wgpu::BindGroupLayout],
    ) -> wgpu::ComputePipeline {
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
            layouts.iter().map(|l| Some(*l)).collect();
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &layouts,
                immediate_size: 0,
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Creates a storage buffer of `size` bytes with no initial contents.
    pub fn storage_buffer(
        &self,
        label: &str,
        size: u64,
        extra: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE | extra,
            mapped_at_creation: false,
        })
    }

    /// Creates a storage buffer holding `data`. Bindings cannot be empty and must fit one
    /// element, so an empty slice gets a zeroed buffer the size of one element.
    pub fn storage_buffer_init<T: bytemuck::Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        let placeholder = vec![0u8; std::mem::size_of::<T>().max(16)];
        let bytes: &[u8] = if data.is_empty() {
            &placeholder
        } else {
            bytemuck::cast_slice(data)
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .expect("a buffer mapped at creation is writable")
            .copy_from_slice(bytes);
        buffer.unmap();
        buffer
    }

    /// Creates a uniform buffer of `size` bytes, updated with `Queue::write_buffer`.
    pub fn uniform_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Creates a bind group, numbering `resources` from binding 0 in order.
    pub fn bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        resources: &[wgpu::BindingResource<'_>],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
            .iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: resource.clone(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &entries,
        })
    }
}

/// Builds a bind group layout, numbering entries from binding 0 in call order.
///
/// Calls read like the shader's binding list, which makes the two easy to compare.
pub struct LayoutBuilder {
    entries: Vec<wgpu::BindGroupLayoutEntry>,
    visibility: wgpu::ShaderStages,
}

impl LayoutBuilder {
    /// Entries visible to compute shaders until [`Self::visibility`] says otherwise.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            visibility: wgpu::ShaderStages::COMPUTE,
        }
    }

    /// Sets the stages for entries added after this call.
    pub fn visibility(mut self, visibility: wgpu::ShaderStages) -> Self {
        self.visibility = visibility;
        self
    }

    fn push(mut self, ty: wgpu::BindingType) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.entries.len() as u32,
            visibility: self.visibility,
            ty,
            count: None,
        });
        self
    }

    fn buffer(self, ty: wgpu::BufferBindingType, dynamic: bool) -> Self {
        self.push(wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        })
    }

    pub fn uniform(self) -> Self {
        self.buffer(wgpu::BufferBindingType::Uniform, false)
    }

    pub fn uniform_dynamic(self) -> Self {
        self.buffer(wgpu::BufferBindingType::Uniform, true)
    }

    pub fn storage_read(self) -> Self {
        self.buffer(wgpu::BufferBindingType::Storage { read_only: true }, false)
    }

    pub fn storage_write(self) -> Self {
        self.buffer(wgpu::BufferBindingType::Storage { read_only: false }, false)
    }

    fn texture_of(
        self,
        sample_type: wgpu::TextureSampleType,
        dimension: wgpu::TextureViewDimension,
    ) -> Self {
        self.push(wgpu::BindingType::Texture {
            sample_type,
            view_dimension: dimension,
            multisampled: false,
        })
    }

    /// A filterable floating-point 2D texture.
    pub fn texture(self) -> Self {
        self.texture_of(
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
        )
    }

    /// A non-filterable floating-point 2D texture, such as `r32float`.
    pub fn texture_unfilterable(self) -> Self {
        self.texture_of(
            wgpu::TextureSampleType::Float { filterable: false },
            wgpu::TextureViewDimension::D2,
        )
    }

    pub fn texture_array(self) -> Self {
        self.texture_of(
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2Array,
        )
    }

    pub fn texture_3d(self) -> Self {
        self.texture_of(
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D3,
        )
    }

    pub fn depth_texture_array(self) -> Self {
        self.texture_of(
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2Array,
        )
    }

    pub fn storage_texture(self, format: wgpu::TextureFormat) -> Self {
        self.push(wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        })
    }

    pub fn storage_texture_3d(self, format: wgpu::TextureFormat) -> Self {
        self.push(wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D3,
        })
    }

    pub fn sampler(self) -> Self {
        self.push(wgpu::BindingType::Sampler(
            wgpu::SamplerBindingType::Filtering,
        ))
    }

    pub fn sampler_comparison(self) -> Self {
        self.push(wgpu::BindingType::Sampler(
            wgpu::SamplerBindingType::Comparison,
        ))
    }

    pub fn build(self, gpu: &Gpu, label: &str) -> wgpu::BindGroupLayout {
        gpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &self.entries,
            })
    }
}

impl Default for LayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A sampled, storage-writable 2D render target.
pub struct Target {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl Target {
    /// Creates a target usable as storage and as a texture. `extra` adds usages.
    pub fn new(
        gpu: &Gpu,
        label: &str,
        size: (u32, u32),
        format: wgpu::TextureFormat,
        extra: wgpu::TextureUsages,
    ) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | extra,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }
}

/// Number of workgroups needed to cover `count` items.
pub fn groups_for(count: u32, group_size: u32) -> u32 {
    count.div_ceil(group_size)
}
