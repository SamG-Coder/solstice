//! A WebGPU renderer built around virtualized geometry and dynamic global illumination.
//!
//! Start with [`Renderer`]: give it a [`solstice_asset::LoadedScene`], and each frame a
//! [`Camera`], an [`Environment`] and a texture to draw to. [`Renderer::render`] is written
//! to read top to bottom as the frame's recipe; each pass lives in its own module.
//!
//! * [`vg`]: cluster LOD selection, culling and rasterization.
//! * [`streaming`]: moving pages of geometry onto and off the GPU.
//! * [`shadow`]: cascaded sun shadows.
//! * [`sky`]: the physically based atmosphere.
//! * [`gi`]: indirect lighting and reflections.
//! * [`lighting`]: combining everything into HDR colour.
//! * [`post`]: temporal resolve, exposure, bloom and tone mapping.

pub mod frame;
pub mod gi;
pub mod gpu;
pub mod lighting;
pub mod post;
pub mod profiler;
pub mod renderer;
pub mod scene;
pub mod shaders;
pub mod shadow;
pub mod sky;
pub mod streaming;
pub mod vg;

pub use frame::{Camera, DebugView, Environment};
pub use gpu::{Gpu, GpuError};
pub use renderer::{FrameStats, Quality, Renderer, RendererSettings};
