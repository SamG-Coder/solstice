//! Data shared between the offline builder and the runtime renderer.
//!
//! * [`gpu`] defines the exact memory layout of everything the shaders read.
//! * [`scene`] defines the CPU-side description of a scene.
//! * [`file`] serialises a scene to a single flat binary.

pub mod file;
pub mod gpu;
pub mod scene;

pub use gpu::{PackedCluster, PackedNode, PackedVertex};
pub use scene::{
    Instance, Light, Material, SceneAsset, TextureEncoding, TextureLayer, Viewpoint, VirtualMesh,
};
