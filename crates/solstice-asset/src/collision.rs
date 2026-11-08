//! What the player can touch in a scene (docs/gameplay.md, Colliding with the scene).
//!
//! A scene with gameplay carries collision shapes made by the bake: coarse triangle meshes, convex
//! hulls for small props, and a heightfield for terrain. They are shared by every instance of a
//! mesh. The shapes live in the file's data area, listed by an optional section in the manifest,
//! so they are fetched only when someone plays. A scene that is only looked at never downloads
//! them, and a scene with no gameplay has none.

use bytemuck::{Pod, Zeroable};

use crate::file::DecodeError;

/// [`Collision::instance_shapes`] of an instance that nothing collides with.
pub const NO_SHAPE: u32 = u32::MAX;

/// How an instance collides, as its scene line asks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Collide {
    /// A coarse triangle mesh, or a convex hull for a small prop.
    #[default]
    Auto,
    /// Nothing: grass, rigging, or what an actor carries with a collider of its own.
    None,
    /// A convex hull, however big it is.
    Hull,
    /// A coarser triangle mesh, for very large meshes such as ground.
    Coarse,
}

/// One shape, in the mesh's own space.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollisionShape {
    /// The convex hull of `vertices`, rather than the triangles.
    pub hull: bool,
    pub vertices: Vec<[f32; 3]>,
    /// Empty for a hull.
    pub triangles: Vec<[u32; 3]>,
}

/// A regular grid of ground heights, in world space.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Heightfield {
    pub columns: u32,
    pub rows: u32,
    /// World x and z of the first sample.
    pub origin: [f32; 2],
    /// Metres between samples, along x and z.
    pub spacing: [f32; 2],
    /// `rows` rows of `columns` heights each, z then x.
    pub heights: Vec<f32>,
}

/// A scene's collision, whole.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Collision {
    pub shapes: Vec<CollisionShape>,
    /// For each instance of the scene, its shape, or [`NO_SHAPE`].
    pub instance_shapes: Vec<u32>,
    pub heightfield: Option<Heightfield>,
}

/// Where one shape's data is, in the file's data area.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct ShapeEntry {
    /// Offset of the deflated data from the start of the data area.
    pub offset: u64,
    pub stored_bytes: u32,
    pub hull: u32,
    pub vertices: u32,
    pub triangles: u32,
}

/// Where a heightfield's data is, and its grid.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct HeightfieldEntry {
    pub offset: u64,
    pub stored_bytes: u32,
    pub columns: u32,
    pub rows: u32,
    pub origin: [f32; 2],
    pub spacing: [f32; 2],
    pub _pad: u32,
}

/// A scene's collision as the manifest lists it: everything but the shapes' data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CollisionIndex {
    pub shapes: Vec<ShapeEntry>,
    pub instance_shapes: Vec<u32>,
    pub heightfield: Option<HeightfieldEntry>,
}

impl CollisionIndex {
    /// Where each piece of data to fetch is, from the start of the data area, as offset and
    /// stored size: every shape, then the heightfield if there is one. Each is deflated.
    pub fn ranges(&self) -> Vec<(u64, u32)> {
        self.shapes
            .iter()
            .map(|s| (s.offset, s.stored_bytes))
            .chain(self.heightfield.iter().map(|h| (h.offset, h.stored_bytes)))
            .collect()
    }

    /// Puts a scene's collision together from the pieces [`CollisionIndex::ranges`] lists,
    /// each inflated. Checks every count and index, as a scene file comes from outside.
    pub fn decode(&self, pieces: &[Vec<u8>]) -> Result<Collision, DecodeError> {
        if pieces.len() != self.ranges().len() {
            return Err(DecodeError::BadCollision);
        }
        let mut shapes = Vec::with_capacity(self.shapes.len());
        for (entry, piece) in self.shapes.iter().zip(pieces) {
            shapes.push(decode_shape(entry, piece)?);
        }
        let heightfield = match &self.heightfield {
            None => None,
            Some(entry) => Some(decode_heightfield(entry, &pieces[self.shapes.len()])?),
        };
        if self
            .instance_shapes
            .iter()
            .any(|&s| s != NO_SHAPE && s as usize >= shapes.len())
        {
            return Err(DecodeError::BadCollision);
        }
        Ok(Collision {
            shapes,
            instance_shapes: self.instance_shapes.clone(),
            heightfield,
        })
    }
}

/// A shape's data: its vertices, then its triangles.
pub(crate) fn encode_shape(shape: &CollisionShape) -> Vec<u8> {
    let mut bytes = bytemuck::cast_slice(&shape.vertices).to_vec();
    bytes.extend_from_slice(bytemuck::cast_slice(&shape.triangles));
    bytes
}

fn pod_vec<T: Pod>(bytes: &[u8]) -> Vec<T> {
    // A copy, as the bytes may not be aligned for `T`.
    let mut out = vec![T::zeroed(); bytes.len() / std::mem::size_of::<T>()];
    bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(bytes);
    out
}

fn decode_shape(entry: &ShapeEntry, bytes: &[u8]) -> Result<CollisionShape, DecodeError> {
    let vertices = entry.vertices as usize;
    let triangles = entry.triangles as usize;
    let split = vertices
        .checked_mul(12)
        .filter(|&s| s.checked_add(triangles.saturating_mul(12)) == Some(bytes.len()))
        .ok_or(DecodeError::BadCollision)?;
    let (vertex_bytes, triangle_bytes) = bytes.split_at(split);
    let shape = CollisionShape {
        hull: entry.hull != 0,
        vertices: pod_vec(vertex_bytes),
        triangles: pod_vec(triangle_bytes),
    };
    let sound = shape.vertices.iter().flatten().all(|c| c.is_finite())
        && shape
            .triangles
            .iter()
            .flatten()
            .all(|&i| (i as usize) < vertices);
    if sound {
        Ok(shape)
    } else {
        Err(DecodeError::BadCollision)
    }
}

fn decode_heightfield(entry: &HeightfieldEntry, bytes: &[u8]) -> Result<Heightfield, DecodeError> {
    let samples = (entry.columns as usize).checked_mul(entry.rows as usize);
    let sound_grid = entry.columns >= 2
        && entry.rows >= 2
        && entry.spacing.iter().all(|s| s.is_finite() && *s > 0.0)
        && entry.origin.iter().all(|o| o.is_finite())
        && samples.and_then(|s| s.checked_mul(4)) == Some(bytes.len());
    if !sound_grid {
        return Err(DecodeError::BadCollision);
    }
    let heights: Vec<f32> = pod_vec(bytes);
    if !heights.iter().all(|h| h.is_finite()) {
        return Err(DecodeError::BadCollision);
    }
    Ok(Heightfield {
        columns: entry.columns,
        rows: entry.rows,
        origin: entry.origin,
        spacing: entry.spacing,
        heights,
    })
}
