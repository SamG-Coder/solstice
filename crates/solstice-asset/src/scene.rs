//! CPU-side description of a scene.

use crate::gpu::{PackedCluster, PackedNode, PackedVertex};
use crate::paged::ClusterSet;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// A mesh preprocessed into a cluster LOD DAG, ready for upload.
#[derive(Clone, Debug, Default)]
pub struct VirtualMesh {
    pub clusters: Vec<PackedCluster>,
    pub nodes: Vec<PackedNode>,
    pub vertices: Vec<PackedVertex>,
    pub triangles: Vec<u32>,
    /// Clusters grouped by shared parent. Each hierarchy leaf is one set.
    pub sets: Vec<ClusterSet>,
    /// See [`ClusterSet::first_generated`].
    pub generated: Vec<u32>,
    /// Index of the hierarchy root in `nodes`.
    pub root_node: u32,
    /// Longest root-to-leaf path in the hierarchy, in nodes.
    pub hierarchy_depth: u32,
    /// Object-space position of the 16-bit grid's origin.
    pub quant_origin: Vec3,
    /// Object-space size of one grid step.
    pub quant_step: f32,
    /// UV decoding: `uv = uv_origin + packed * uv_step`.
    pub uv_origin: [f32; 2],
    pub uv_step: [f32; 2],
    /// Object-space bounding sphere of the whole mesh.
    pub bounds_center: Vec3,
    pub bounds_radius: f32,
    /// Triangle count of the source mesh (finest LOD only).
    pub source_triangles: u64,
    /// For a skinned mesh, each cluster's bones. Empty for one that isn't. In a skinned mesh,
    /// each cluster's vertices in `vertices` are followed by their skins.
    pub cluster_bones: Vec<crate::gpu::ClusterBones>,
}

impl VirtualMesh {
    pub fn skinned(&self) -> bool {
        !self.cluster_bones.is_empty()
    }
}

/// A physically based surface description.
///
/// `#[repr(C)]` so the runtime can upload this slice directly; mirrored by
/// `Material` in `common/scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Material {
    pub base_color: [f32; 4],
    /// RGB radiance.
    pub emissive: [f32; 3],
    /// [`Material::TWO_SIDED`], [`Material::FOLIAGE`] and [`Material::MASKED`]. Older files hold
    /// zero here.
    pub flags: u32,
    pub roughness: f32,
    pub metallic: f32,
    /// Texture repeats per world unit of UV.
    pub uv_scale: f32,
    /// Strength of the normal map; 0 disables it.
    pub normal_strength: f32,
    /// Index into the texture array, or -1 for an untextured material.
    pub texture_layer: i32,
    /// How strongly to hide repetition, for textures that tile. Leave at 0 for textures made
    /// for one object, such as scans, whose colour would otherwise be stained.
    pub macro_variation: f32,
    /// Which texture array holds the layer right now. Set by the renderer, zero on disk.
    pub texture_tier: u32,
    /// Where in that array the layer is. Set by the renderer, zero on disk.
    pub texture_slot: u32,
}

impl Material {
    /// Drawn from both sides, with the normal turned towards the viewer.
    pub const TWO_SIDED: u32 = 1;
    /// Thin, translucent vegetation: light from behind shines through, tinted by its colour, as
    /// Unreal's two-sided foliage shading model does. Foliage should also be two-sided.
    pub const FOLIAGE: u32 = 2;
    /// Cut out: alpha in its colour map is coverage instead of cavity, and a texel is solid when
    /// that is at least 0.5. The cut-off is built into the map (`docs/cutouts.md`).
    pub const MASKED: u32 = 4;
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: [0.8, 0.8, 0.8, 1.0],
            emissive: [0.0; 3],
            flags: 0,
            roughness: 0.7,
            metallic: 0.0,
            uv_scale: 1.0,
            normal_strength: 1.0,
            texture_layer: -1,
            macro_variation: 0.0,
            texture_tier: 0,
            texture_slot: 0,
        }
    }
}

/// An omnidirectional local light. Mirrored by `Light` in `common/scene.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Light {
    pub position: [f32; 3],
    /// Distance at which the light fades to nothing.
    pub range: f32,
    /// Linear RGB intensity, in candela.
    pub intensity: [f32; 3],
    pub _pad: u32,
}

/// One texture set, with a full mip chain for each map. Their byte layout depends on the
/// scene's [`TextureEncoding`].
///
/// * `color`: RGB base colour (sRGB), A cavity, or coverage for a [`Material::MASKED`] material.
/// * `normal`: RG tangent-space normal XY.
/// * `material`: R roughness multiplier (0.5 is neutral), G metallic multiplier.
#[derive(Clone, Debug, Default)]
pub struct TextureLayer {
    pub color_mips: Vec<Vec<u8>>,
    pub normal_mips: Vec<Vec<u8>>,
    pub material_mips: Vec<Vec<u8>>,
}

/// How a scene's texture maps are stored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextureEncoding {
    /// Colour is RGBA8, normal and material are RG8.
    #[default]
    Raw,
    /// Colour is BC7, normal and material are BC5.
    Block,
}

impl TextureEncoding {
    /// Bytes in a mip level of a colour map, whose edge is `edge` texels.
    pub fn color_mip_bytes(self, edge: u32) -> usize {
        self.mip_bytes(edge, 4)
    }

    /// Bytes in a mip level of a normal or material map, whose edge is `edge` texels.
    pub fn data_mip_bytes(self, edge: u32) -> usize {
        self.mip_bytes(edge, 2)
    }

    fn mip_bytes(self, edge: u32, raw_texel_bytes: usize) -> usize {
        let edge = edge as usize;
        match self {
            Self::Raw => edge * edge * raw_texel_bytes,
            // Both block formats take 16 bytes per 4x4 block, and a small mip still fills one.
            Self::Block => edge.div_ceil(4).pow(2) * 16,
        }
    }
}

/// One placement of a mesh in the world.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub mesh: u32,
    pub material: u32,
    pub transform: Mat4,
}

/// Where a viewer should start.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Viewpoint {
    pub position: [f32; 3],
    /// Rotation around +Y, in radians. Zero looks along -Z.
    pub yaw: f32,
    /// Rotation above the horizon, in radians.
    pub pitch: f32,
}

/// An open sea around a scene: where its surface lies, and what its water is like.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sea {
    /// Height of the calm surface, in metres.
    pub level: f32,
    /// The name of a built-in ocean, such as `coastal`, or a whole ocean in its text form, as
    /// `solstice_render::Ocean::parse` reads it, so a scene can carry an ocean of its own.
    pub ocean: String,
}

/// One text file of a scene's gameplay: an actor file, a script, the input actions, or the list
/// of placed actors. Carried as it was written, by the name it was written under, so the scene
/// file needs no knowledge of what is in it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GameFile {
    /// Its path relative to the scene's text file, with `/` between folders.
    pub name: String,
    pub text: String,
}

/// A complete, self-contained scene.
#[derive(Clone, Debug, Default)]
pub struct SceneAsset {
    pub meshes: Vec<VirtualMesh>,
    pub materials: Vec<Material>,
    pub instances: Vec<Instance>,
    pub lights: Vec<Light>,
    /// Edge length of the square textures in `textures`.
    pub texture_size: u32,
    pub texture_encoding: TextureEncoding,
    pub textures: Vec<TextureLayer>,
    pub viewpoint: Option<Viewpoint>,
    pub sea: Option<Sea>,
    /// The scene's gameplay, if it has any. Empty for a scene that is only looked at.
    pub gameplay: Vec<GameFile>,
    /// How each instance collides, where the scene says. Instances past its end are
    /// [`Collide::Auto`](crate::collision::Collide::Auto).
    pub collide: Vec<crate::collision::Collide>,
    /// The ground as a grid of heights, for collision, if the scene has terrain.
    pub ground: Option<crate::collision::Heightfield>,
    /// Every skeleton in the scene.
    pub skeletons: Vec<crate::skeleton::Skeleton>,
    /// The poses a scene can hold at once, each the index of its skeleton: one for each placed
    /// character, however many meshes it is made of.
    pub poses: Vec<u32>,
    /// Which pose each instance follows, or [`crate::skeleton::NO_POSE`]. Instances past its end
    /// follow none.
    pub instance_poses: Vec<u32>,
    /// Every animation in the scene.
    pub clips: Vec<crate::skeleton::Clip>,
}

impl SceneAsset {
    /// Triangles drawn if every instance rendered its finest LOD.
    pub fn total_source_triangles(&self) -> u64 {
        self.instances
            .iter()
            .map(|i| self.meshes[i.mesh as usize].source_triangles)
            .sum()
    }

    /// Triangles stored once per mesh, regardless of instancing.
    pub fn unique_source_triangles(&self) -> u64 {
        self.meshes.iter().map(|m| m.source_triangles).sum()
    }
}
