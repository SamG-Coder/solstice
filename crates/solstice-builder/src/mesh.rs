//! Input mesh type.

use glam::{Vec2, Vec3};

/// A vertex of an input mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

/// An indexed triangle mesh.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Replaces every normal with the area-weighted average of its adjacent faces.
    pub fn compute_smooth_normals(&mut self) {
        for vertex in &mut self.vertices {
            vertex.normal = Vec3::ZERO;
        }
        for triangle in self.indices.chunks_exact(3) {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]].map(|i| i as usize);
            let face = (self.vertices[b].position - self.vertices[a].position)
                .cross(self.vertices[c].position - self.vertices[a].position);
            self.vertices[a].normal += face;
            self.vertices[b].normal += face;
            self.vertices[c].normal += face;
        }
        for vertex in &mut self.vertices {
            vertex.normal = vertex.normal.normalize_or(Vec3::Y);
        }
    }

    /// Appends another mesh.
    pub fn append(&mut self, other: &Mesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }
}
