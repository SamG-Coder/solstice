//! Writes meshes as a glTF 2.0 file and its binary buffer.

use crate::geometry::Geometry;
use std::fmt::Write as _;
use std::io;
use std::path::Path;

/// A material with no textures.
pub struct Plain {
    pub name: &'static str,
    /// Linear base colour.
    pub color: [f32; 3],
    pub roughness: f32,
}

/// One glTF file being put together. Each named part is a node at the origin, its vertices in
/// world space, so that the scene places it at 0 0 0.
#[derive(Default)]
pub struct File {
    bin: Vec<u8>,
    materials: Vec<String>,
    accessors: Vec<String>,
    views: Vec<String>,
    meshes: Vec<String>,
    nodes: Vec<String>,
}

impl File {
    pub fn material(&mut self, material: &Plain) -> usize {
        let [r, g, b] = material.color;
        self.materials.push(format!(
            r#"{{"name":"{}","pbrMetallicRoughness":{{"baseColorFactor":[{r},{g},{b},1],"metallicFactor":0,"roughnessFactor":{}}}}}"#,
            material.name, material.roughness
        ));
        self.materials.len() - 1
    }

    /// Adds raw little-endian values and an accessor for them, returning the accessor's number.
    fn accessor(
        &mut self,
        bytes: &[u8],
        component: u32,
        count: usize,
        kind: &str,
        bounds: Option<String>,
    ) -> usize {
        let offset = self.bin.len();
        self.bin.extend_from_slice(bytes);
        self.views.push(format!(
            r#"{{"buffer":0,"byteOffset":{offset},"byteLength":{}}}"#,
            bytes.len()
        ));
        let mut accessor = format!(
            r#"{{"bufferView":{},"componentType":{component},"count":{count},"type":"{kind}""#,
            self.views.len() - 1
        );
        if let Some(bounds) = bounds {
            accessor.push_str(&bounds);
        }
        accessor.push('}');
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    fn floats(
        &mut self,
        values: &[f32],
        count: usize,
        kind: &str,
        bounds: Option<String>,
    ) -> usize {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.accessor(&bytes, 5126, count, kind, bounds)
    }

    /// Adds a part made of pieces, each in its own material. Pieces with no triangles are left out.
    pub fn part(&mut self, name: &str, pieces: &[(&Geometry, usize)]) {
        let mut primitives = String::new();
        for (geometry, material) in pieces.iter().filter(|(g, _)| !g.indices.is_empty()) {
            let (mut min, mut max) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
            for &p in &geometry.positions {
                min = min.min(p);
                max = max.max(p);
            }
            let bounds = format!(
                r#","min":[{},{},{}],"max":[{},{},{}]"#,
                min.x, min.y, min.z, max.x, max.y, max.z
            );
            let count = geometry.positions.len();
            let positions: Vec<f32> = geometry
                .positions
                .iter()
                .flat_map(|p| p.to_array())
                .collect();
            let normals: Vec<f32> = geometry.normals.iter().flat_map(|n| n.to_array()).collect();
            let uvs: Vec<f32> = geometry.uvs.iter().flat_map(|uv| uv.to_array()).collect();
            let position = self.floats(&positions, count, "VEC3", Some(bounds));
            let normal = self.floats(&normals, count, "VEC3", None);
            let uv = self.floats(&uvs, count, "VEC2", None);
            let index_bytes: Vec<u8> = geometry
                .indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            let indices = self.accessor(&index_bytes, 5125, geometry.indices.len(), "SCALAR", None);
            if !primitives.is_empty() {
                primitives.push(',');
            }
            let _ = write!(
                primitives,
                r#"{{"attributes":{{"POSITION":{position},"NORMAL":{normal},"TEXCOORD_0":{uv}}},"indices":{indices},"material":{material}}}"#
            );
        }
        if primitives.is_empty() {
            return;
        }
        self.meshes
            .push(format!(r#"{{"primitives":[{primitives}]}}"#));
        self.nodes.push(format!(
            r#"{{"name":"{name}","mesh":{}}}"#,
            self.meshes.len() - 1
        ));
    }

    /// Writes `NAME.gltf` and `NAME.bin` into `folder`.
    pub fn write(&self, folder: &Path, name: &str) -> io::Result<()> {
        let nodes: Vec<String> = (0..self.nodes.len()).map(|i| i.to_string()).collect();
        let json = format!(
            r#"{{"asset":{{"version":"2.0","generator":"townkit"}},"scene":0,"scenes":[{{"nodes":[{}]}}],"nodes":[{}],"meshes":[{}],"materials":[{}],"accessors":[{}],"bufferViews":[{}],"buffers":[{{"uri":"{name}.bin","byteLength":{}}}]}}"#,
            nodes.join(","),
            self.nodes.join(","),
            self.meshes.join(","),
            self.materials.join(","),
            self.accessors.join(","),
            self.views.join(","),
            self.bin.len(),
        );
        std::fs::create_dir_all(folder)?;
        std::fs::write(folder.join(format!("{name}.gltf")), json)?;
        std::fs::write(folder.join(format!("{name}.bin")), &self.bin)
    }
}
