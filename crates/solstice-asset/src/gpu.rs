//! Memory layouts read directly by the shaders.
//!
//! Every struct here has a twin in `solstice-render/shaders/common/scene.wgsl`.
//! The two must be changed together; the size assertions at the bottom of this
//! file catch accidental drift on the Rust side.

use bytemuck::{Pod, Zeroable};

/// Maximum triangles per cluster. The visibility buffer reserves 7 bits for the
/// triangle index, so this cannot grow without changing the ID encoding.
pub const MAX_CLUSTER_TRIANGLES: usize = 128;

/// Maximum vertices per cluster, limited by the 8-bit indices in a packed triangle.
pub const MAX_CLUSTER_VERTICES: usize = 255;

/// Error (world units) given to clusters with no coarser parent, so that they are
/// always "too coarse to be replaced" and terminate the LOD cut. Finite because
/// infinities are implementation-defined in WGSL.
pub const ROOT_PARENT_ERROR: f32 = 1.0e30;

/// A vertex compressed to 16 bytes.
///
/// Positions sit on a single mesh-wide 16-bit grid. Using one grid rather than
/// one per cluster is what keeps neighbouring clusters crack-free: a shared
/// vertex always decodes to the identical position.
///
/// Layout, as little-endian `u16`s: `x y z nx ny u v _`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct PackedVertex {
    pub data: [u16; 8],
}

/// One leaf of the LOD DAG: up to [`MAX_CLUSTER_TRIANGLES`] triangles.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PackedCluster {
    /// Object-space bounding sphere of this cluster's triangles, used for culling.
    pub center: [f32; 3],
    pub radius: f32,
    /// Object-space sphere over which `lod_error` is measured. Shared by all
    /// clusters that were generated from the same group.
    pub lod_center: [f32; 3],
    pub lod_radius: f32,
    /// Simplification error of this cluster, in object-space units.
    pub lod_error: f32,
    /// First vertex in the shared vertex pool.
    pub vertex_offset: u32,
    /// First triangle in the shared triangle pool.
    pub triangle_offset: u32,
    /// `vertex_count | triangle_count << 8`.
    pub counts: u32,
}

/// A node of the per-mesh hierarchy used to skip whole subtrees of clusters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PackedNode {
    /// Object-space sphere bounding all clusters below this node.
    pub center: [f32; 3],
    pub radius: f32,
    /// Sphere bounding the *parent* LOD spheres below this node.
    pub lod_center: [f32; 3],
    pub lod_radius: f32,
    /// Largest parent error below this node. When even this projects below the
    /// error threshold, every cluster below is finer than necessary and the
    /// whole subtree can be skipped.
    pub max_parent_error: f32,
    /// First child node, or first cluster if this is a leaf.
    pub first: u32,
    /// Number of children or clusters.
    pub count: u32,
    /// Non-zero if `first` indexes clusters instead of nodes.
    pub is_leaf: u32,
}

impl PackedCluster {
    pub fn vertex_count(&self) -> u32 {
        self.counts & 0xff
    }

    pub fn triangle_count(&self) -> u32 {
        (self.counts >> 8) & 0xff
    }
}

/// Packs three cluster-local vertex indices into one `u32`.
pub fn pack_triangle(a: u8, b: u8, c: u8) -> u32 {
    u32::from(a) | u32::from(b) << 8 | u32::from(c) << 16
}

/// Inverse of [`pack_triangle`].
pub fn unpack_triangle(packed: u32) -> [u8; 3] {
    [packed as u8, (packed >> 8) as u8, (packed >> 16) as u8]
}

const _: () = {
    assert!(std::mem::size_of::<PackedVertex>() == 16);
    assert!(std::mem::size_of::<PackedCluster>() == 48);
    assert!(std::mem::size_of::<PackedNode>() == 48);
};
