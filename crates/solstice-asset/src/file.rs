//! The scene file: a small manifest, then pages and texture levels that can each be read alone.
//!
//! ```text
//! header    magic, version, where the manifest and data area are
//! manifest  deflated: everything in a [`PagedScene`], then optional tagged sections
//! data      each page deflated on its own, then texture levels as they are
//! ```
//!
//! Each optional section is a four-byte tag, its length in bytes, and its contents. Readers skip
//! tags they don't know, and older readers stop at the manifest's fixed part, so a section can be
//! added without changing the version: files stay readable both ways.
//!
//! Block-compressed textures barely shrink further, so they aren't deflated. Pages are deflated
//! separately so a client can fetch any one with an HTTP Range request. All values are
//! little-endian, and loading is one bounds-checked copy per array.

use crate::collision::{encode_shape, Collision, CollisionIndex, HeightfieldEntry, ShapeEntry};
use crate::gpu::{PackedCluster, PackedNode};
use crate::paged::{
    upper_mip_count, ClusterSet, LoadedScene, MeshInfo, PageInfo, PagedScene, PAGE_BYTES,
};
use crate::scene::{
    GameFile, Instance, Light, Material, Sea, TextureEncoding, TextureLayer, Viewpoint,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::fmt;

const MAGIC: [u8; 8] = *b"SOLSTICE";
const VERSION: u32 = 4;
/// How hard to compress when baking. Unpacking takes the same time at any level.
const DEFLATE_LEVEL: u8 = 9;
/// The optional section holding the scene's [`Sea`]: its level, then its ocean's text.
const SEA_TAG: [u8; 4] = *b"SEA ";
/// The optional section holding the scene's gameplay: how many [`GameFile`]s, then each one's
/// name and text. Written only for scenes that have gameplay.
const GAME_TAG: [u8; 4] = *b"GAME";
/// The optional section listing the scene's collision shapes, whose data is in the data area.
const COLLISION_TAG: [u8; 4] = *b"COLL";

/// Why a scene file could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    BadInstanceReference,
    BadTextures,
    BadPages,
    BadSea,
    BadGameplay,
    BadCollision,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a Solstice scene file"),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported file version {v} (expected {VERSION})")
            }
            Self::Truncated => write!(f, "file is truncated"),
            Self::BadInstanceReference => {
                write!(
                    f,
                    "an instance refers to a mesh or material that does not exist"
                )
            }
            Self::BadTextures => write!(f, "the texture data does not match its stated format"),
            Self::BadPages => write!(f, "the page tables are inconsistent"),
            Self::BadSea => write!(f, "the sea's description is not text"),
            Self::BadGameplay => write!(f, "the scene's gameplay files are malformed"),
            Self::BadCollision => write!(f, "the scene's collision shapes are malformed"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// The fixed start of a file. It says how to fetch the rest.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Header {
    magic: [u8; 8],
    version: u32,
    _pad: u32,
    /// Where the deflated manifest begins.
    pub manifest_offset: u64,
    /// Size of the deflated manifest. The data area follows it.
    pub manifest_bytes: u64,
}

/// Bytes a client must fetch first, to read a [`Header`].
pub const HEADER_BYTES: usize = std::mem::size_of::<Header>();

/// Parses the start of a file.
pub fn decode_header(bytes: &[u8]) -> Result<Header, DecodeError> {
    if bytes.len() < MAGIC.len() || bytes[..MAGIC.len()] != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let raw = bytes.get(..HEADER_BYTES).ok_or(DecodeError::Truncated)?;
    let header: Header = bytemuck::pod_read_unaligned(raw);
    if header.version != VERSION {
        return Err(DecodeError::UnsupportedVersion(header.version));
    }
    Ok(header)
}

/// On-disk form of an [`Instance`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawInstance {
    mesh: u32,
    material: u32,
    transform: [f32; 16],
}

/// On-disk form of a [`MeshInfo`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RawMesh {
    /// `u32::MAX` for a mesh with no geometry.
    root_node: u32,
    quant_origin: [f32; 3],
    quant_step: f32,
    uv_origin: [f32; 2],
    uv_step: [f32; 2],
}

/// Serialises a scene.
pub fn encode(loaded: &LoadedScene) -> Vec<u8> {
    let pinned = loaded.scene.pinned_pages as usize;
    let upper_count = upper_mip_count(loaded.scene.texture_size);

    let mut scene = loaded.scene.clone();
    let mut data = Vec::new();
    for (info, raw) in scene.pages[pinned..].iter_mut().zip(&loaded.pages) {
        let packed = miniz_oxide::deflate::compress_to_vec(raw, DEFLATE_LEVEL);
        info.offset = data.len() as u64;
        info.stored_bytes = packed.len() as u32;
        data.extend_from_slice(&packed);
    }
    scene.upper_mip_offsets.clear();
    for layer in &loaded.upper_textures {
        for mip in 0..upper_count {
            scene.upper_mip_offsets.push(data.len() as u64);
            data.extend_from_slice(&layer.color_mips[mip]);
            data.extend_from_slice(&layer.normal_mips[mip]);
            data.extend_from_slice(&layer.material_mips[mip]);
        }
    }

    if let Some(collision) = &loaded.collision {
        scene.collision = Some(append_collision(collision, &mut data));
    }

    let manifest = miniz_oxide::deflate::compress_to_vec(&encode_manifest(&scene), DEFLATE_LEVEL);
    let header = Header {
        magic: MAGIC,
        version: VERSION,
        _pad: 0,
        manifest_offset: HEADER_BYTES as u64,
        manifest_bytes: manifest.len() as u64,
    };
    let mut out = Vec::with_capacity(HEADER_BYTES + manifest.len() + data.len());
    out.extend_from_slice(bytemuck::bytes_of(&header));
    out.extend_from_slice(&manifest);
    out.extend_from_slice(&data);
    out
}

fn encode_manifest(scene: &PagedScene) -> Vec<u8> {
    let mut w = Writer::default();
    w.slice(&scene.clusters);
    w.slice(&scene.nodes);
    w.slice(&scene.sets);
    w.slice(&scene.generated);
    w.slice(&scene.pages);
    w.pod(&scene.pinned_pages);
    w.slice(&scene.pinned_data);

    let meshes: Vec<RawMesh> = scene
        .meshes
        .iter()
        .map(|mesh| RawMesh {
            root_node: mesh.root_node.unwrap_or(u32::MAX),
            quant_origin: mesh.quant_origin.to_array(),
            quant_step: mesh.quant_step,
            uv_origin: mesh.uv_origin,
            uv_step: mesh.uv_step,
        })
        .collect();
    w.slice(&meshes);
    let instances: Vec<RawInstance> = scene
        .instances
        .iter()
        .map(|i| RawInstance {
            mesh: i.mesh,
            material: i.material,
            transform: i.transform.to_cols_array(),
        })
        .collect();
    w.slice(&instances);
    w.slice(&scene.materials);
    w.slice(&scene.lights);
    w.pod(&u32::from(scene.viewpoint.is_some()));
    w.pod(&scene.viewpoint.unwrap_or_default());
    w.pod(&scene.hierarchy_depth);
    w.pod(&scene.source_triangles);
    w.pod(&scene.unique_triangles);

    w.pod(&scene.texture_size);
    w.pod(&match scene.texture_encoding {
        TextureEncoding::Raw => 0u32,
        TextureEncoding::Block => 1u32,
    });
    w.pod(&(scene.base_textures.len() as u32));
    for layer in &scene.base_textures {
        w.pod(&(layer.color_mips.len() as u32));
        for mip in layer
            .color_mips
            .iter()
            .chain(&layer.normal_mips)
            .chain(&layer.material_mips)
        {
            w.slice(mip);
        }
    }
    w.slice(&scene.upper_mip_offsets);

    if let Some(sea) = &scene.sea {
        let mut contents = sea.level.to_le_bytes().to_vec();
        contents.extend_from_slice(sea.ocean.as_bytes());
        w.section(SEA_TAG, &contents);
    }
    if !scene.gameplay.is_empty() {
        w.section(GAME_TAG, &encode_gameplay(&scene.gameplay));
    }
    if let Some(index) = &scene.collision {
        w.section(COLLISION_TAG, &encode_collision_index(index));
    }
    w.finish()
}

/// Writes a scene's collision shapes into the data area, each deflated on its own so that it
/// can be fetched alone, and lists where they went.
fn append_collision(collision: &Collision, data: &mut Vec<u8>) -> CollisionIndex {
    let mut store = |raw: &[u8]| {
        let packed = miniz_oxide::deflate::compress_to_vec(raw, DEFLATE_LEVEL);
        let offset = data.len() as u64;
        data.extend_from_slice(&packed);
        (offset, packed.len() as u32)
    };
    let shapes = collision
        .shapes
        .iter()
        .map(|shape| {
            let (offset, stored_bytes) = store(&encode_shape(shape));
            ShapeEntry {
                offset,
                stored_bytes,
                hull: u32::from(shape.hull),
                vertices: shape.vertices.len() as u32,
                triangles: shape.triangles.len() as u32,
            }
        })
        .collect();
    let heightfield = collision.heightfield.as_ref().map(|ground| {
        let (offset, stored_bytes) = store(bytemuck::cast_slice(&ground.heights));
        HeightfieldEntry {
            offset,
            stored_bytes,
            columns: ground.columns,
            rows: ground.rows,
            origin: ground.origin,
            spacing: ground.spacing,
            _pad: 0,
        }
    });
    CollisionIndex {
        shapes,
        instance_shapes: collision.instance_shapes.clone(),
        heightfield,
    }
}

fn encode_collision_index(index: &CollisionIndex) -> Vec<u8> {
    let mut w = Writer::default();
    w.slice(&index.shapes);
    w.slice(&index.instance_shapes);
    let heightfield: Vec<HeightfieldEntry> = index.heightfield.iter().copied().collect();
    w.slice(&heightfield);
    w.finish()
}

fn decode_collision_index(bytes: &[u8]) -> Result<CollisionIndex, DecodeError> {
    let mut r = Reader { bytes };
    let bad = |_| DecodeError::BadCollision;
    let shapes: Vec<ShapeEntry> = r.slice().map_err(bad)?;
    let instance_shapes: Vec<u32> = r.slice().map_err(bad)?;
    let heightfield: Vec<HeightfieldEntry> = r.slice().map_err(bad)?;
    if heightfield.len() > 1 || !r.bytes.is_empty() {
        return Err(DecodeError::BadCollision);
    }
    Ok(CollisionIndex {
        shapes,
        instance_shapes,
        heightfield: heightfield.first().copied(),
    })
}

/// A gameplay section's contents: a count, then each file's name and text, each a length and
/// its UTF-8 bytes.
fn encode_gameplay(files: &[GameFile]) -> Vec<u8> {
    let mut bytes = (files.len() as u32).to_le_bytes().to_vec();
    for file in files {
        for text in [&file.name, &file.text] {
            bytes.extend_from_slice(&(text.len() as u32).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
    }
    bytes
}

fn decode_gameplay(mut bytes: &[u8]) -> Result<Vec<GameFile>, DecodeError> {
    fn take<'a>(bytes: &mut &'a [u8], len: usize) -> Result<&'a [u8], DecodeError> {
        let (head, rest) = bytes
            .split_at_checked(len)
            .ok_or(DecodeError::BadGameplay)?;
        *bytes = rest;
        Ok(head)
    }
    fn number(bytes: &mut &[u8]) -> Result<usize, DecodeError> {
        let word = take(bytes, 4)?;
        Ok(u32::from_le_bytes(word.try_into().expect("four bytes")) as usize)
    }
    fn text(bytes: &mut &[u8]) -> Result<String, DecodeError> {
        let len = number(bytes)?;
        String::from_utf8(take(bytes, len)?.to_vec()).map_err(|_| DecodeError::BadGameplay)
    }
    let count = number(&mut bytes)?;
    // Each file takes at least eight bytes, so a count beyond that is a lie, not a large file.
    if count > bytes.len() / 8 {
        return Err(DecodeError::BadGameplay);
    }
    let mut files = Vec::with_capacity(count);
    for _ in 0..count {
        let name = text(&mut bytes)?;
        let text = text(&mut bytes)?;
        files.push(GameFile { name, text });
    }
    if !bytes.is_empty() {
        return Err(DecodeError::BadGameplay);
    }
    Ok(files)
}

/// Parses an unpacked manifest. `data_offset` is where the file's data area begins.
pub fn decode_manifest(bytes: &[u8], data_offset: u64) -> Result<PagedScene, DecodeError> {
    let mut r = Reader { bytes };
    let clusters: Vec<PackedCluster> = r.slice()?;
    let nodes: Vec<PackedNode> = r.slice()?;
    let sets: Vec<ClusterSet> = r.slice()?;
    let generated: Vec<u32> = r.slice()?;
    let pages: Vec<PageInfo> = r.slice()?;
    let pinned_pages: u32 = r.pod()?;
    let pinned_data: Vec<u8> = r.slice()?;

    let raw_meshes: Vec<RawMesh> = r.slice()?;
    let meshes: Vec<MeshInfo> = raw_meshes
        .iter()
        .map(|mesh| MeshInfo {
            root_node: (mesh.root_node != u32::MAX).then_some(mesh.root_node),
            quant_origin: Vec3::from_array(mesh.quant_origin),
            quant_step: mesh.quant_step,
            uv_origin: mesh.uv_origin,
            uv_step: mesh.uv_step,
        })
        .collect();
    let raw_instances: Vec<RawInstance> = r.slice()?;
    let instances: Vec<Instance> = raw_instances
        .iter()
        .map(|raw| Instance {
            mesh: raw.mesh,
            material: raw.material,
            transform: Mat4::from_cols_array(&raw.transform),
        })
        .collect();
    let materials: Vec<Material> = r.slice()?;
    let lights: Vec<Light> = r.slice()?;
    let references_valid = instances
        .iter()
        .all(|i| (i.mesh as usize) < meshes.len() && (i.material as usize) < materials.len());
    if !references_valid {
        return Err(DecodeError::BadInstanceReference);
    }

    let has_viewpoint: u32 = r.pod()?;
    let viewpoint: Viewpoint = r.pod()?;
    let hierarchy_depth: u32 = r.pod()?;
    let source_triangles: u64 = r.pod()?;
    let unique_triangles: u64 = r.pod()?;

    let texture_size: u32 = r.pod()?;
    let texture_encoding = match r.pod::<u32>()? {
        0 => TextureEncoding::Raw,
        1 => TextureEncoding::Block,
        _ => return Err(DecodeError::BadTextures),
    };
    let layer_count: u32 = r.pod()?;
    let mut base_textures = Vec::new();
    for _ in 0..layer_count {
        let mip_count: u32 = r.pod()?;
        let mut layer = TextureLayer::default();
        for _ in 0..mip_count {
            layer.color_mips.push(r.slice()?);
        }
        for _ in 0..mip_count {
            layer.normal_mips.push(r.slice()?);
        }
        for _ in 0..mip_count {
            layer.material_mips.push(r.slice()?);
        }
        if !base_is_well_formed(&layer, texture_size, texture_encoding) {
            return Err(DecodeError::BadTextures);
        }
        base_textures.push(layer);
    }
    let upper_mip_offsets: Vec<u64> = r.slice()?;

    let mut sea = None;
    let mut gameplay = Vec::new();
    let mut collision = None;
    while !r.bytes.is_empty() {
        let tag: [u8; 4] = r.pod()?;
        let contents: Vec<u8> = r.slice()?;
        if tag == SEA_TAG {
            let (level, text) = contents
                .split_first_chunk::<4>()
                .ok_or(DecodeError::Truncated)?;
            sea = Some(Sea {
                level: f32::from_le_bytes(*level),
                ocean: String::from_utf8(text.to_vec()).map_err(|_| DecodeError::BadSea)?,
            });
        } else if tag == GAME_TAG {
            gameplay = decode_gameplay(&contents)?;
        } else if tag == COLLISION_TAG {
            collision = Some(decode_collision_index(&contents)?);
        }
    }

    let scene = PagedScene {
        clusters,
        nodes,
        sets,
        generated,
        pages,
        pinned_pages,
        pinned_data,
        meshes,
        instances,
        materials,
        lights,
        viewpoint: (has_viewpoint != 0).then_some(viewpoint),
        sea,
        gameplay,
        collision,
        hierarchy_depth,
        source_triangles,
        unique_triangles,
        texture_size,
        texture_encoding,
        base_textures,
        upper_mip_offsets,
        data_offset,
    };
    if !tables_are_consistent(&scene) {
        return Err(DecodeError::BadPages);
    }
    Ok(scene)
}

/// Unpacks a whole file held in memory, pages and all.
pub fn decode(bytes: &[u8]) -> Result<LoadedScene, DecodeError> {
    let header = decode_header(bytes)?;
    let end = header
        .manifest_offset
        .checked_add(header.manifest_bytes)
        .ok_or(DecodeError::Truncated)?;
    let packed = usize::try_from(header.manifest_offset)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(start, end)| bytes.get(start..end))
        .ok_or(DecodeError::Truncated)?;
    let manifest = inflate(packed)?;
    let scene = decode_manifest(&manifest, end)?;
    drop(manifest);

    let mut pages = Vec::new();
    for page in 0..scene.pages.len() as u32 {
        if page >= scene.pinned_pages {
            let packed = stored_range(bytes, page_range(&scene, page))?;
            pages.push(unpack_page(&scene, page, packed)?);
        }
    }

    let upper_count = upper_mip_count(scene.texture_size);
    let mut upper_textures = Vec::new();
    for layer in 0..scene.base_textures.len() {
        let mut out = TextureLayer::default();
        for mip in 0..upper_count {
            let range = upper_mip_range(&scene, layer, mip).ok_or(DecodeError::BadTextures)?;
            let stored = stored_range(bytes, range)?;
            let edge = scene.texture_size >> mip;
            let color = scene.texture_encoding.color_mip_bytes(edge);
            let data = scene.texture_encoding.data_mip_bytes(edge);
            out.color_mips.push(stored[..color].to_vec());
            out.normal_mips.push(stored[color..color + data].to_vec());
            out.material_mips.push(stored[color + data..].to_vec());
        }
        upper_textures.push(out);
    }

    let collision = match &scene.collision {
        None => None,
        Some(index) => {
            let mut pieces = Vec::new();
            for (offset, stored) in index.ranges() {
                let start = scene.data_offset + offset;
                let packed = stored_range(bytes, (start, start + u64::from(stored)))?;
                pieces.push(inflate(packed)?);
            }
            Some(index.decode(&pieces)?)
        }
    };

    Ok(LoadedScene {
        scene,
        pages,
        upper_textures,
        collision,
    })
}

/// The bytes of a page in the file, as absolute offsets.
pub fn page_range(scene: &PagedScene, page: u32) -> (u64, u64) {
    let info = &scene.pages[page as usize];
    let start = scene.data_offset + info.offset;
    (start, start + u64::from(info.stored_bytes))
}

/// The bytes of one upper mip level of one layer in the file, as absolute offsets: its colour,
/// normal and material data, in that order. `None` if the file has no such level.
pub fn upper_mip_range(scene: &PagedScene, layer: usize, mip: usize) -> Option<(u64, u64)> {
    let count = upper_mip_count(scene.texture_size);
    if mip >= count {
        return None;
    }
    let offset = *scene.upper_mip_offsets.get(layer * count + mip)?;
    let edge = scene.texture_size >> mip;
    let encoding = scene.texture_encoding;
    let bytes = encoding.color_mip_bytes(edge) + 2 * encoding.data_mip_bytes(edge);
    let start = scene.data_offset + offset;
    Some((start, start + bytes as u64))
}

fn stored_range(bytes: &[u8], (start, end): (u64, u64)) -> Result<&[u8], DecodeError> {
    usize::try_from(start)
        .ok()
        .zip(usize::try_from(end).ok())
        .and_then(|(start, end)| bytes.get(start..end))
        .ok_or(DecodeError::Truncated)
}

/// Unpacks a deflated block.
pub fn inflate(packed: &[u8]) -> Result<Vec<u8>, DecodeError> {
    miniz_oxide::inflate::decompress_to_vec(packed).map_err(|_| DecodeError::Truncated)
}

/// Unpacks a page and checks that it is the size the tables say.
pub fn unpack_page(scene: &PagedScene, page: u32, packed: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let raw = inflate(packed)?;
    if raw.len() != scene.pages[page as usize].raw_bytes() {
        return Err(DecodeError::BadPages);
    }
    Ok(raw)
}

/// Checks everything the renderer would otherwise index blindly while installing pages. A bad
/// file should fail here rather than panic later.
fn tables_are_consistent(scene: &PagedScene) -> bool {
    let clusters = scene.clusters.len() as u64;
    let generated = scene.generated.len() as u64;
    let sets = scene.sets.len() as u64;
    let pages = scene.pages.len() as u64;
    let sets_ok = scene.sets.iter().all(|set| {
        u64::from(set.first_cluster) + u64::from(set.cluster_count) <= clusters
            && u64::from(set.first_generated) + u64::from(set.generated_count) <= generated
            && u64::from(set.page) < pages
            && (set.leaf_node as usize) < scene.nodes.len()
    });
    let pages_ok = scene.pages.iter().all(|page| {
        page.raw_bytes() <= PAGE_BYTES
            && u64::from(page.first_set) + u64::from(page.set_count) <= sets
    });
    let pinned = scene.pinned_pages as usize;
    let pinned_ok = pinned <= scene.pages.len()
        && scene.pages[..pinned]
            .iter()
            .map(PageInfo::raw_bytes)
            .sum::<usize>()
            == scene.pinned_data.len();
    sets_ok
        && pages_ok
        && pinned_ok
        && scene.generated.iter().all(|&c| u64::from(c) < clusters)
        && scene.meshes.iter().all(|mesh| {
            mesh.root_node
                .is_none_or(|root| (root as usize) < scene.nodes.len())
        })
}

/// Checks that every level is the size the GPU will expect, which a bad file could otherwise
/// turn into a validation error much later.
fn base_is_well_formed(layer: &TextureLayer, size: u32, encoding: TextureEncoding) -> bool {
    if !size.is_power_of_two() || (encoding == TextureEncoding::Block && size < 4) {
        return false;
    }
    let upper = upper_mip_count(size);
    let levels = size.ilog2() as usize + 1 - upper;
    let edge = |level: usize| size >> (level + upper);
    layer.color_mips.len() == levels
        && (0..levels).all(|level| {
            layer.color_mips[level].len() == encoding.color_mip_bytes(edge(level))
                && layer.normal_mips[level].len() == encoding.data_mip_bytes(edge(level))
                && layer.material_mips[level].len() == encoding.data_mip_bytes(edge(level))
        })
}

#[derive(Default)]
struct Writer {
    out: Vec<u8>,
}

impl Writer {
    fn bytes(&mut self, data: &[u8]) {
        self.out.extend_from_slice(data);
    }

    fn pod<T: Pod>(&mut self, value: &T) {
        self.bytes(bytemuck::bytes_of(value));
    }

    fn slice<T: Pod>(&mut self, values: &[T]) {
        self.pod(&(values.len() as u64));
        self.bytes(bytemuck::cast_slice(values));
    }

    /// An optional section: its tag, then its contents with their length.
    fn section(&mut self, tag: [u8; 4], contents: &[u8]) {
        self.bytes(&tag);
        self.slice(contents);
    }

    fn finish(self) -> Vec<u8> {
        self.out
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], DecodeError> {
        if count > self.bytes.len() {
            return Err(DecodeError::Truncated);
        }
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }

    fn pod<T: Pod>(&mut self) -> Result<T, DecodeError> {
        let raw = self.take(std::mem::size_of::<T>())?;
        Ok(bytemuck::pod_read_unaligned(raw))
    }

    fn slice<T: Pod>(&mut self) -> Result<Vec<T>, DecodeError> {
        let len: u64 = self.pod()?;
        let byte_len = usize::try_from(len)
            .ok()
            .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
            .ok_or(DecodeError::Truncated)?;
        let raw = self.take(byte_len)?;
        Ok(raw
            .chunks_exact(std::mem::size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::{CollisionShape, Heightfield};
    use crate::gpu::{pack_triangle, PackedVertex, NODE_LEAF};

    /// One pinned page and one that loads on demand, with textures above and below the base.
    fn sample_scene() -> LoadedScene {
        let page_bytes = |vertex: u8| {
            let vertices = [PackedVertex {
                data: [u16::from(vertex); 8],
            }; 3];
            let mut bytes: Vec<u8> = bytemuck::cast_slice(&vertices).to_vec();
            bytes.extend_from_slice(bytemuck::bytes_of(&pack_triangle(0, 1, 2)));
            bytes
        };
        let info = PageInfo {
            vertex_count: 3,
            triangle_count: 1,
            set_count: 1,
            ..Default::default()
        };
        let edge = 256u32;
        let encoding = TextureEncoding::Raw;
        let mip =
            |edge: u32, bytes: usize, fill: u8| vec![fill; edge as usize * edge as usize * bytes];
        let chain = |first: u32, bytes: usize, fill: u8| -> Vec<Vec<u8>> {
            (first..=edge.ilog2())
                .map(|m| mip(edge >> m, bytes, fill))
                .collect()
        };
        let scene = PagedScene {
            clusters: vec![
                PackedCluster {
                    radius: 2.0,
                    counts: 3 | 1 << 8,
                    ..Default::default()
                };
                2
            ],
            nodes: vec![
                PackedNode {
                    is_leaf: NODE_LEAF,
                    ..Default::default()
                },
                PackedNode {
                    is_leaf: NODE_LEAF | 1,
                    first: 1,
                    count: 1,
                    ..Default::default()
                },
            ],
            sets: vec![
                ClusterSet {
                    cluster_count: 1,
                    ..Default::default()
                },
                ClusterSet {
                    first_cluster: 1,
                    cluster_count: 1,
                    generated_count: 1,
                    leaf_node: 1,
                    page: 1,
                    ..Default::default()
                },
            ],
            generated: vec![0],
            pages: vec![
                info,
                PageInfo {
                    first_set: 1,
                    ..info
                },
            ],
            pinned_pages: 1,
            pinned_data: page_bytes(1),
            meshes: vec![MeshInfo {
                root_node: Some(0),
                quant_step: 0.5,
                ..Default::default()
            }],
            instances: vec![Instance {
                mesh: 0,
                material: 0,
                transform: Mat4::from_translation(Vec3::X),
            }],
            materials: vec![Material::default()],
            lights: vec![Light {
                range: 4.0,
                ..Default::default()
            }],
            viewpoint: Some(Viewpoint {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.25,
            }),
            hierarchy_depth: 3,
            source_triangles: 7,
            unique_triangles: 5,
            texture_size: edge,
            texture_encoding: encoding,
            base_textures: vec![TextureLayer {
                color_mips: chain(1, 4, 1),
                normal_mips: chain(1, 2, 2),
                material_mips: chain(1, 2, 3),
            }],
            ..Default::default()
        };
        LoadedScene {
            scene,
            pages: vec![page_bytes(2)],
            upper_textures: vec![TextureLayer {
                color_mips: vec![mip(edge, 4, 4)],
                normal_mips: vec![mip(edge, 2, 5)],
                material_mips: vec![mip(edge, 2, 6)],
            }],
            collision: None,
        }
    }

    #[test]
    fn round_trip_preserves_everything() {
        let loaded = sample_scene();
        let decoded = decode(&encode(&loaded)).unwrap();
        let (a, b) = (&decoded.scene, &loaded.scene);
        assert_eq!(a.clusters, b.clusters);
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.sets, b.sets);
        assert_eq!(a.generated, b.generated);
        assert_eq!(a.pinned_pages, 1);
        assert_eq!(a.pinned_data, b.pinned_data);
        assert_eq!(a.materials, b.materials);
        assert_eq!(a.lights, b.lights);
        assert_eq!(a.viewpoint, b.viewpoint);
        assert_eq!(a.sea, b.sea);
        assert_eq!(a.meshes, b.meshes);
        assert_eq!(a.instances[0].transform, b.instances[0].transform);
        assert_eq!(a.hierarchy_depth, 3);
        assert_eq!(a.source_triangles, 7);
        assert_eq!(
            a.base_textures[0].normal_mips,
            b.base_textures[0].normal_mips
        );
        assert_eq!(decoded.pages, loaded.pages);
        assert_eq!(
            decoded.upper_textures[0].color_mips,
            loaded.upper_textures[0].color_mips
        );
        assert_eq!(
            decoded.upper_textures[0].normal_mips,
            loaded.upper_textures[0].normal_mips
        );
        assert_eq!(
            decoded.upper_textures[0].material_mips,
            loaded.upper_textures[0].material_mips
        );
        assert_eq!(decoded.page(0), loaded.page(0));
        assert_eq!(decoded.page(1), loaded.page(1));
    }

    #[test]
    fn a_sea_travels_with_its_scene() {
        let mut loaded = sample_scene();
        let sea = Sea {
            level: -0.6,
            ocean: "# Clear water\nabsorption 0.35 0.06 0.015\n".to_string(),
        };
        loaded.scene.sea = Some(sea.clone());
        assert_eq!(decode(&encode(&loaded)).unwrap().scene.sea, Some(sea));
        // And a scene without one has none.
        assert_eq!(decode(&encode(&sample_scene())).unwrap().scene.sea, None);
    }

    #[test]
    fn gameplay_travels_with_its_scene() {
        let mut loaded = sample_scene();
        let files = vec![
            GameFile {
                name: "cove/gate.actor".to_string(),
                text: "name gate\nscript gate.star\n".to_string(),
            },
            GameFile {
                name: "cove/gate.star".to_string(),
                text: "def on_begin(self):\n    pass  # «unicode» too\n".to_string(),
            },
            GameFile::default(),
        ];
        loaded.scene.gameplay = files.clone();
        assert_eq!(decode(&encode(&loaded)).unwrap().scene.gameplay, files);
    }

    fn sample_collision() -> Collision {
        Collision {
            shapes: vec![
                CollisionShape {
                    hull: false,
                    vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.5]],
                    triangles: vec![[0, 1, 2]],
                },
                CollisionShape {
                    hull: true,
                    vertices: vec![[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [1.0; 3]],
                    triangles: vec![],
                },
            ],
            instance_shapes: vec![1],
            heightfield: Some(Heightfield {
                columns: 2,
                rows: 3,
                origin: [-5.0, 2.0],
                spacing: [1.5, 2.5],
                heights: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            }),
        }
    }

    #[test]
    fn collision_travels_with_its_scene_and_stays_out_of_the_manifest() {
        let mut loaded = sample_scene();
        let plain = encode(&loaded);
        loaded.collision = Some(sample_collision());
        let bytes = encode(&loaded);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.collision, loaded.collision);
        // The shapes are in the data area, each to be fetched alone.
        let index = decoded.scene.collision.as_ref().unwrap();
        assert_eq!(index.ranges().len(), 3);
        let mut pieces = Vec::new();
        for (offset, stored) in index.ranges() {
            let start = (decoded.scene.data_offset + offset) as usize;
            pieces.push(inflate(&bytes[start..start + stored as usize]).unwrap());
        }
        assert_eq!(index.decode(&pieces).unwrap(), sample_collision());
        // The data shares nothing with the rest: pages and textures stay where they were.
        assert_eq!(decode(&plain).unwrap().scene.collision, None);
        assert_eq!(decoded.pages, sample_scene().pages);
    }

    #[test]
    fn broken_collision_is_an_error_not_a_guess() {
        let collision = sample_collision();
        let mut data = Vec::new();
        let index = append_collision(&collision, &mut data);
        let mut pieces: Vec<Vec<u8>> = index
            .ranges()
            .iter()
            .map(|&(o, s)| inflate(&data[o as usize..o as usize + s as usize]).unwrap())
            .collect();
        assert_eq!(index.decode(&pieces).unwrap(), collision);
        let bad = |index: &CollisionIndex, pieces: &[Vec<u8>]| {
            assert_eq!(index.decode(pieces), Err(DecodeError::BadCollision));
        };
        bad(&index, &pieces[..2]);
        let mut wrong = index.clone();
        wrong.shapes[0].vertices = 4;
        bad(&wrong, &pieces);
        let mut wrong = index.clone();
        wrong.shapes[0].vertices = u32::MAX;
        bad(&wrong, &pieces);
        let mut wrong = index.clone();
        wrong.instance_shapes[0] = 2;
        bad(&wrong, &pieces);
        let mut wrong = index.clone();
        wrong.heightfield.as_mut().unwrap().rows = 2;
        bad(&wrong, &pieces);
        let mut wrong = index.clone();
        wrong.heightfield.as_mut().unwrap().spacing[0] = 0.0;
        bad(&wrong, &pieces);
        // A triangle naming a vertex that isn't there.
        let saved = pieces[0].clone();
        let at = pieces[0].len() - 4;
        pieces[0][at..].copy_from_slice(&3u32.to_le_bytes());
        bad(&index, &pieces);
        pieces[0] = saved;
        // A vertex that isn't a number.
        pieces[1][0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        bad(&index, &pieces);
        // A section cut short.
        let section = encode_collision_index(&index);
        for cut in 0..section.len() {
            assert_eq!(
                decode_collision_index(&section[..cut]),
                Err(DecodeError::BadCollision),
                "cut at {cut}"
            );
        }
    }

    #[test]
    fn a_scene_without_gameplay_is_written_exactly_as_before() {
        // No section at all, so viewers from before gameplay read it as they always did.
        let plain = sample_scene();
        let manifest = encode_manifest(&plain.scene);
        assert!(!manifest.windows(4).any(|w| w == GAME_TAG));
        let mut with = sample_scene();
        with.scene.gameplay = vec![GameFile {
            name: "a".to_string(),
            text: "b".to_string(),
        }];
        let longer = encode_manifest(&with.scene);
        assert_eq!(&longer[..manifest.len()], &manifest[..]);
        assert_eq!(decode_manifest(&manifest, 0).unwrap().gameplay, vec![]);
    }

    #[test]
    fn broken_gameplay_is_an_error_not_a_guess() {
        let good = encode_gameplay(&[GameFile {
            name: "a.star".to_string(),
            text: "x = 1\n".to_string(),
        }]);
        assert!(decode_gameplay(&good).is_ok());
        for cut in 0..good.len() {
            assert_eq!(
                decode_gameplay(&good[..cut]),
                Err(DecodeError::BadGameplay),
                "cut at {cut}"
            );
        }
        let mut extra = good.clone();
        extra.push(0);
        assert_eq!(decode_gameplay(&extra), Err(DecodeError::BadGameplay));
        let mut huge = good.clone();
        huge[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode_gameplay(&huge), Err(DecodeError::BadGameplay));
        let mut not_text = good;
        not_text[8] = 0xff;
        assert_eq!(decode_gameplay(&not_text), Err(DecodeError::BadGameplay));
    }

    #[test]
    fn manifests_from_before_sections_still_read() {
        // A manifest of the fixed part alone, as files were before optional sections, reads
        // as a scene with no sea.
        let mut loaded = sample_scene();
        loaded.scene.sea = None;
        let manifest = encode_manifest(&loaded.scene);
        let scene = decode_manifest(&manifest, 0).unwrap();
        assert_eq!(scene.sea, None);
        assert_eq!(scene.clusters, loaded.scene.clusters);
    }

    #[test]
    fn unknown_sections_are_skipped() {
        // A section from a newer writer, before and after the sea, is passed over.
        let mut loaded = sample_scene();
        let sea = Sea {
            level: 2.0,
            ocean: String::new(),
        };
        loaded.scene.sea = Some(sea.clone());
        let mut w = Writer { out: Vec::new() };
        w.section(*b"NEW1", &[1, 2, 3]);
        let mut manifest = encode_manifest(&loaded.scene);
        // Fixed part, then the unknown section, then the sea.
        let mut fixed = sample_scene().scene;
        fixed.sea = None;
        let fixed_len = encode_manifest(&fixed).len();
        let sea_section = manifest.split_off(fixed_len);
        manifest.extend(w.finish());
        manifest.extend(sea_section);
        assert_eq!(decode_manifest(&manifest, 0).unwrap().sea, Some(sea));
        // A section cut short is an error, not a silent loss.
        manifest.truncate(manifest.len() - 2);
        assert!(decode_manifest(&manifest, 0).is_err());
    }

    #[test]
    fn a_page_and_a_mip_can_be_read_without_the_rest() {
        let loaded = sample_scene();
        let file = encode(&loaded);
        let header = decode_header(&file[..HEADER_BYTES]).unwrap();
        let start = header.manifest_offset as usize;
        let end = start + header.manifest_bytes as usize;
        let scene = decode_manifest(&inflate(&file[start..end]).unwrap(), end as u64).unwrap();

        let (from, to) = page_range(&scene, 1);
        let page = unpack_page(&scene, 1, &file[from as usize..to as usize]).unwrap();
        assert_eq!(page, loaded.pages[0]);

        let (from, to) = upper_mip_range(&scene, 0, 0).unwrap();
        let stored = &file[from as usize..to as usize];
        assert_eq!(stored.len(), 256 * 256 * 8);
        assert_eq!((stored[0], *stored.last().unwrap()), (4, 6));
        assert_eq!(upper_mip_range(&scene, 0, 1), None);
    }

    #[test]
    fn rejects_foreign_and_truncated_data() {
        assert_eq!(decode(b"not a scene").unwrap_err(), DecodeError::BadMagic);
        let bytes = encode(&sample_scene());
        assert_eq!(
            decode(&bytes[..bytes.len() - 1]).unwrap_err(),
            DecodeError::Truncated
        );
    }

    #[test]
    fn rejects_other_versions() {
        let mut bytes = encode(&sample_scene());
        bytes[8] = 3;
        assert_eq!(
            decode(&bytes).unwrap_err(),
            DecodeError::UnsupportedVersion(3)
        );
    }

    #[test]
    fn rejects_textures_of_the_wrong_size() {
        let mut scene = sample_scene();
        scene.scene.base_textures[0].normal_mips[0].pop();
        assert_eq!(
            decode(&encode(&scene)).unwrap_err(),
            DecodeError::BadTextures
        );
    }

    #[test]
    fn rejects_dangling_instance_references() {
        let mut scene = sample_scene();
        scene.scene.instances[0].mesh = 9;
        assert_eq!(
            decode(&encode(&scene)).unwrap_err(),
            DecodeError::BadInstanceReference
        );
    }

    #[test]
    fn rejects_page_tables_that_point_outside_the_scene() {
        let mut scene = sample_scene();
        scene.scene.sets[1].first_generated = 40;
        assert_eq!(decode(&encode(&scene)).unwrap_err(), DecodeError::BadPages);

        let mut scene = sample_scene();
        scene.scene.pinned_data.pop();
        assert_eq!(decode(&encode(&scene)).unwrap_err(), DecodeError::BadPages);
    }
}
