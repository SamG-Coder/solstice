//! Bakes a glTF scene too large to hold in memory, straight to a scene file.
//!
//! [`crate::import_gltf`] reads a whole file and builds a [`SceneAsset`] from it. That cannot work
//! for a scene of billions of triangles, so this does the same job a mesh at a time:
//!
//! 1. Meshes are read and built in parallel, as many at once as a limit on triangles in flight
//!    allows. What a mesh keeps in memory afterwards is its cluster records; its vertices and
//!    triangles go to a spill file.
//! 2. Pages are cut from the spill file, and deflated into a temporary data area.
//! 3. The spill file is deleted, and the header and manifest are written ahead of the data area.
//!
//! It understands `EXT_meshopt_compression` and `EXT_mesh_gpu_instancing`, which large scenes
//! need. If the drive holding the output runs low, the bake stops and cleans up.

use crate::ktx2::{self, Ktx2};
use crate::material::is_invisible;
use crate::meshopt::decompress_view;
use crate::{frame_bounds, read_primitive, Bounds, MeshKey};
use glam::{Mat3, Mat4, Quat, Vec3};
use gltf::{buffer, Document, Gltf, Node, Primitive};
use rayon::prelude::*;
use solstice_asset::file::{deflate_page, encode_header_and_manifest};
use solstice_asset::paged::{upper_mip_count, PagedScene};
use solstice_asset::scene::TextureEncoding;
use solstice_asset::{Instance, Material, PackedVertex, SceneAsset, TextureLayer, VirtualMesh};
use solstice_builder::{
    build_virtual_mesh, paginate_from, BuildOptions, ClusterRange, ClusterRecords, ClusterStore,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

/// How to bake a large scene.
#[derive(Clone, Debug)]
pub struct LargeOptions {
    /// Edge length of the scene's textures.
    pub texture_size: u32,
    /// Whether to bring in the materials' textures.
    pub textures: bool,
    /// Bakes only what lies within this distance of this point, to try out a part of a scene.
    pub region: Option<(Vec3, f32)>,
    /// Stops the bake if the output's drive has less free than this.
    pub min_free_bytes: u64,
    /// How many source triangles may be in memory at once, which bounds the memory used.
    pub triangles_in_flight: u64,
}

impl Default for LargeOptions {
    fn default() -> Self {
        Self {
            texture_size: 1024,
            textures: true,
            region: None,
            min_free_bytes: 50 << 30,
            triangles_in_flight: 120_000_000,
        }
    }
}

/// What a bake made.
#[derive(Debug, Default)]
pub struct LargeReport {
    pub meshes: usize,
    pub instances: usize,
    pub unique_triangles: u64,
    pub instanced_triangles: u64,
    pub clusters: usize,
    pub pages: usize,
    pub file_bytes: u64,
    pub build_seconds: f32,
    pub page_seconds: f32,
    pub texture_seconds: f32,
    pub texture_layers: usize,
    /// Things in the file that the result does not reproduce.
    pub warnings: Vec<String>,
}

/// A built mesh: which of the meshes to build it was, and where its geometry is in the spill.
type FinishedMesh = (usize, VirtualMesh, (u64, u64));

/// One placement of a primitive.
struct Placement {
    source: MeshKey,
    material: Option<usize>,
    transform: Mat4,
}

/// Limits the triangles in flight, waiting while too many are.
struct Budget {
    used: Mutex<u64>,
    freed: Condvar,
    limit: u64,
}

impl Budget {
    fn take(&self, triangles: u64) -> u64 {
        // A mesh over the whole limit still has to be built: alone.
        let triangles = triangles.min(self.limit);
        let mut used = self.used.lock().expect("budget");
        while *used + triangles > self.limit {
            used = self.freed.wait(used).expect("budget");
        }
        *used += triangles;
        triangles
    }

    fn give_back(&self, triangles: u64) {
        *self.used.lock().expect("budget") -= triangles;
        self.freed.notify_all();
    }
}

/// How much is free on the drive that holds `path`, or `None` if that can't be found out.
fn free_bytes(path: &Path) -> Option<u64> {
    let absolute = std::path::absolute(path).ok()?;
    #[cfg(windows)]
    {
        let root = absolute.components().next()?;
        let drive = root.as_os_str().to_str()?.trim_end_matches(':').to_owned();
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!("(Get-PSDrive -Name '{drive}').Free"),
            ])
            .output()
            .ok()?;
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    }
    #[cfg(not(windows))]
    {
        let output = std::process::Command::new("df")
            .args(["-Pk", absolute.parent()?.to_str()?])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&output.stdout).into_owned();
        let kilobytes: u64 = text
            .lines()
            .nth(1)?
            .split_whitespace()
            .nth(3)?
            .parse()
            .ok()?;
        Some(kilobytes * 1024)
    }
}

/// Fails if the drive is running out, so that a bake cannot fill it.
fn check_free(path: &Path, minimum: u64) -> Result<(), String> {
    match free_bytes(path) {
        Some(free) if free < minimum => Err(format!(
            "stopping: only {:.1} GiB free on the output drive, the limit is {:.1} GiB",
            free as f64 / (1u64 << 30) as f64,
            minimum as f64 / (1u64 << 30) as f64,
        )),
        _ => Ok(()),
    }
}

/// Reads a whole accessor of floats that nothing has compressed.
fn read_floats(
    document: &Document,
    accessor: usize,
    components: usize,
    buffers: &[buffer::Data],
) -> Result<Vec<f32>, String> {
    let accessor = document
        .accessors()
        .nth(accessor)
        .ok_or("an instance attribute names an accessor that does not exist")?;
    if accessor.data_type() != gltf::accessor::DataType::F32 {
        return Err("instance attributes must be floats".to_owned());
    }
    let view = accessor.view().ok_or("an instance attribute has no data")?;
    if view.extension_value("EXT_meshopt_compression").is_some() {
        return Err("compressed instance attributes are not supported".to_owned());
    }
    let stride = view.stride().unwrap_or(components * 4);
    let start = view.offset() + accessor.offset();
    let data = &buffers[view.buffer().index()].0;
    let mut values = Vec::with_capacity(accessor.count() * components);
    for element in 0..accessor.count() {
        let at = start + element * stride;
        let bytes = data
            .get(at..at + components * 4)
            .ok_or("an instance attribute runs off the end of its buffer")?;
        values.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes"))),
        );
    }
    Ok(values)
}

/// The instance transforms of a node, or `None` if it isn't instanced.
fn instance_transforms(
    node: &Node<'_>,
    document: &Document,
    buffers: &[buffer::Data],
) -> Result<Option<Vec<Mat4>>, String> {
    let Some(extension) = node.extension_value("EXT_mesh_gpu_instancing") else {
        return Ok(None);
    };
    let accessor_of = |name: &str| {
        extension
            .get("attributes")
            .and_then(|a| a.get(name))
            .and_then(gltf::json::Value::as_u64)
            .map(|a| a as usize)
    };
    let attribute = |name: &str, components: usize| {
        accessor_of(name)
            .map(|accessor| read_floats(document, accessor, components, buffers))
            .transpose()
    };
    let translations = attribute("TRANSLATION", 3)?;
    let rotations = attribute("ROTATION", 4)?;
    let scales = attribute("SCALE", 3)?;
    let count = [
        translations.as_ref().map(|v| v.len() / 3),
        rotations.as_ref().map(|v| v.len() / 4),
        scales.as_ref().map(|v| v.len() / 3),
    ]
    .into_iter()
    .flatten()
    .next()
    .ok_or("an instanced node has no instance attributes")?;
    let mut transforms = Vec::with_capacity(count);
    for i in 0..count {
        let translation = translations
            .as_ref()
            .map_or(Vec3::ZERO, |v| Vec3::from_slice(&v[i * 3..i * 3 + 3]));
        let rotation = rotations.as_ref().map_or(Quat::IDENTITY, |v| {
            Quat::from_slice(&v[i * 4..i * 4 + 4]).normalize()
        });
        let scale = scales
            .as_ref()
            .map_or(Vec3::ONE, |v| Vec3::from_slice(&v[i * 3..i * 3 + 3]));
        transforms.push(Mat4::from_scale_rotation_translation(
            scale,
            rotation,
            translation,
        ));
    }
    Ok(Some(transforms))
}

/// Collects what the scene draws, and where.
struct Walk<'a> {
    document: &'a Document,
    buffers: &'a [buffer::Data],
    region: Option<(Vec3, f32)>,
    placements: Vec<Placement>,
    bounds: Bounds,
    left_out: usize,
}

impl Walk<'_> {
    fn visit(&mut self, node: &Node<'_>, parent: Mat4) -> Result<(), String> {
        let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        if let Some(mesh) = node.mesh() {
            let copies = instance_transforms(node, self.document, self.buffers)?;
            for local in copies.as_deref().unwrap_or(&[Mat4::IDENTITY]) {
                let world = transform * *local;
                let mirrored = Mat3::from_mat4(world).determinant() < 0.0;
                for primitive in mesh.primitives() {
                    self.place(&mesh, &primitive, world, mirrored);
                }
            }
        }
        for child in node.children() {
            self.visit(&child, transform)?;
        }
        Ok(())
    }

    fn place(
        &mut self,
        mesh: &gltf::Mesh<'_>,
        primitive: &Primitive<'_>,
        world: Mat4,
        mirrored: bool,
    ) {
        if primitive.mode() != gltf::mesh::Mode::Triangles || is_invisible(&primitive.material()) {
            return;
        }
        let box_ = primitive.bounding_box();
        if let Some((center, radius)) = self.region {
            let middle = (Vec3::from(box_.min) + Vec3::from(box_.max)) * 0.5;
            if world.transform_point3(middle).distance(center) > radius {
                self.left_out += 1;
                return;
            }
        }
        self.bounds
            .add_box(Vec3::from(box_.min), Vec3::from(box_.max), world);
        self.placements.push(Placement {
            source: MeshKey {
                mesh: mesh.index(),
                primitive: primitive.index(),
                mirrored,
            },
            material: primitive.material().index(),
            transform: world,
        });
    }
}

/// Reads the vertices and triangles of finished meshes back from the spill file.
struct SpillStore {
    file: BufReader<File>,
    at: u64,
    /// Where each mesh's vertices and triangles begin in the file.
    bases: Vec<(u64, u64)>,
}

impl SpillStore {
    fn read_into(&mut self, offset: u64, target: &mut [u8]) {
        if offset != self.at {
            // Within the buffer this costs nothing, and pages read mostly forward.
            let moved = self.file.seek_relative(offset as i64 - self.at as i64);
            moved.expect("seeking in the spill file");
        }
        self.file
            .read_exact(target)
            .expect("reading the spill file");
        self.at = offset + target.len() as u64;
    }
}

impl ClusterStore for SpillStore {
    fn read(&mut self, mesh: u32, range: ClusterRange, out: &mut ClusterRecords) {
        let (vertices, triangles) = self.bases[mesh as usize];
        let start = out.vertices.len();
        out.vertices
            .resize(start + range.vertices, PackedVertex::default());
        let offset = vertices + (range.first_vertex * std::mem::size_of::<PackedVertex>()) as u64;
        self.read_into(offset, bytemuck::cast_slice_mut(&mut out.vertices[start..]));
        let start = out.triangles.len();
        out.triangles.resize(start + range.triangles, 0);
        let offset = triangles + (range.first_triangle * 4) as u64;
        self.read_into(
            offset,
            bytemuck::cast_slice_mut(&mut out.triangles[start..]),
        );
    }
}

/// Reads the buffers one mesh needs, and decompresses its data.
fn load_mesh_buffers(
    document: &Document,
    mesh: &gltf::Mesh<'_>,
    folder: &Path,
) -> Result<Vec<buffer::Data>, String> {
    let mut views = Vec::new();
    for primitive in mesh.primitives() {
        for (_, accessor) in primitive.attributes() {
            views.extend(accessor.view());
        }
        views.extend(primitive.indices().and_then(|a| a.view()));
    }
    views.sort_by_key(gltf::buffer::View::index);
    views.dedup_by_key(|view| view.index());

    let mut buffers: Vec<buffer::Data> = (0..document.buffers().count())
        .map(|_| buffer::Data(Vec::new()))
        .collect();
    let by_index: Vec<_> = document.buffers().collect();
    let load = |index: usize, buffers: &mut [buffer::Data]| -> Result<(), String> {
        if buffers[index].0.is_empty() {
            buffers[index] = read_buffer(&by_index[index], folder)?;
        }
        Ok(())
    };
    for view in &views {
        load(view.buffer().index(), &mut buffers)?;
        if let Some(extension) = view.extension_value("EXT_meshopt_compression") {
            let source = extension
                .get("buffer")
                .and_then(gltf::json::Value::as_u64)
                .ok_or("a compressed view names no buffer")? as usize;
            if source >= by_index.len() {
                return Err("a compressed view names a buffer that does not exist".to_owned());
            }
            load(source, &mut buffers)?;
        }
    }
    for view in &views {
        decompress_view(view, &mut buffers).map_err(|e| format!("mesh {}: {e}", mesh.index()))?;
    }
    Ok(buffers)
}

/// A buffer's bytes: from its file, or room for a compressed view's output if it has no file.
fn read_buffer(buffer: &gltf::Buffer<'_>, folder: &Path) -> Result<buffer::Data, String> {
    match buffer.source() {
        buffer::Source::Uri(uri) => {
            let path = folder.join(uri);
            std::fs::read(&path)
                .map(buffer::Data)
                .map_err(|e| format!("{}: {e}", path.display()))
        }
        buffer::Source::Bin => Ok(buffer::Data(vec![0; buffer.length()])),
    }
}

/// Bakes the glTF file at `path` to a scene file at `output`.
pub fn bake_large(
    path: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &LargeOptions,
) -> Result<LargeReport, String> {
    let (path, output) = (path.as_ref(), output.as_ref());
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let spill_path = with_suffix(output, ".spill.tmp");
    let data_path = with_suffix(output, ".data.tmp");
    let result = bake_inner(path, output, &spill_path, &data_path, options);
    // Whether it worked or not, these must not be left taking up the drive.
    let _ = std::fs::remove_file(&spill_path);
    let _ = std::fs::remove_file(&data_path);
    result
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn bake_inner(
    path: &Path,
    output: &Path,
    spill_path: &Path,
    data_path: &Path,
    options: &LargeOptions,
) -> Result<LargeReport, String> {
    let folder = path.parent().unwrap_or(Path::new("."));
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let document = Gltf::from_slice_without_validation(&bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .document;
    drop(bytes);
    check_free(output, options.min_free_bytes)?;

    let mut report = LargeReport::default();
    // Instance attributes are not compressed, and nothing else in these buffers is read here.
    let buffers: Vec<buffer::Data> = document
        .buffers()
        .map(|buffer| match buffer.source() {
            buffer::Source::Uri(uri) if uri.ends_with(".nodes.bin") => read_buffer(&buffer, folder),
            _ => Ok(buffer::Data(Vec::new())),
        })
        .collect::<Result<_, _>>()?;
    let mut walk = Walk {
        document: &document,
        buffers: &buffers,
        region: options.region,
        placements: Vec::new(),
        bounds: Bounds::default(),
        left_out: 0,
    };
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or("the file has no scene")?;
    for node in scene.nodes() {
        walk.visit(&node, Mat4::IDENTITY)?;
    }
    let Walk {
        placements,
        bounds,
        left_out,
        ..
    } = walk;
    if placements.is_empty() {
        return Err("nothing to draw in that region".to_owned());
    }
    drop(buffers);
    if left_out > 0 {
        println!("{left_out} placements lie outside the region");
    }

    // The meshes to build, and which placement draws which.
    let mut slots: HashMap<MeshKey, u32> = HashMap::new();
    let mut keys: Vec<MeshKey> = Vec::new();
    let mut material_slots: HashMap<Option<usize>, u32> = HashMap::new();
    let mut materials: Vec<Material> = Vec::new();
    let mut layers = LayerTable::default();
    let mut instances = Vec::with_capacity(placements.len());
    for placement in &placements {
        let mesh = *slots.entry(placement.source).or_insert_with(|| {
            keys.push(placement.source);
            keys.len() as u32 - 1
        });
        let material = *material_slots.entry(placement.material).or_insert_with(|| {
            materials.push(convert_material(
                &document,
                placement.material,
                options.textures,
                &mut layers,
                &mut report.warnings,
            ));
            materials.len() as u32 - 1
        });
        instances.push(Instance {
            mesh,
            material,
            transform: placement.transform,
        });
    }

    let mut by_source: HashMap<usize, Vec<usize>> = HashMap::new();
    for (index, key) in keys.iter().enumerate() {
        by_source.entry(key.mesh).or_default().push(index);
    }
    let mut work: Vec<(usize, Vec<usize>, u64)> = by_source
        .into_iter()
        .map(|(mesh, built)| {
            let source = document.meshes().nth(mesh).expect("from walking this file");
            let triangles = built
                .iter()
                .map(|&i| triangles_of(&source, keys[i].primitive))
                .sum();
            (mesh, built, triangles)
        })
        .collect();
    work.sort_by_key(|&(mesh, ..)| mesh);
    let total: u64 = work.iter().map(|w| w.2).sum();
    println!(
        "{} instances, {} meshes to build, {:.3} G triangles",
        instances.len(),
        keys.len(),
        total as f64 / 1e9
    );

    // Build every mesh, spilling its geometry.
    let built_start = Instant::now();
    let budget = Budget {
        used: Mutex::new(0),
        freed: Condvar::new(),
        limit: options.triangles_in_flight,
    };
    struct Spill {
        file: BufWriter<File>,
        at: u64,
        since_check: u64,
    }
    let spill = Mutex::new(Spill {
        file: BufWriter::with_capacity(
            1 << 22,
            File::create(spill_path).map_err(|e| e.to_string())?,
        ),
        at: 0,
        since_check: 0,
    });
    let finished: Mutex<Vec<FinishedMesh>> = Mutex::new(Vec::with_capacity(keys.len()));
    let done = Mutex::new(0u64);
    let build_options = BuildOptions::default();
    work.par_iter()
        .try_for_each(|(mesh, built, triangles)| -> Result<(), String> {
            let taken = budget.take(*triangles);
            let result = (|| -> Result<(), String> {
                let source = document
                    .meshes()
                    .nth(*mesh)
                    .expect("from walking this file");
                let buffers = load_mesh_buffers(&document, &source, folder)?;
                for &index in built {
                    let key = keys[index];
                    let primitive = source
                        .primitives()
                        .nth(key.primitive)
                        .expect("from walking this file");
                    let geometry = read_primitive(&primitive, &buffers, key.mirrored);
                    let mut virtual_mesh = build_virtual_mesh(&geometry, &build_options);
                    drop(geometry);
                    let vertices = std::mem::take(&mut virtual_mesh.vertices);
                    let triangles = std::mem::take(&mut virtual_mesh.triangles);
                    let bases = {
                        let mut spill = spill.lock().expect("spill");
                        let vertex_base = spill.at;
                        let v: &[u8] = bytemuck::cast_slice(&vertices);
                        let t: &[u8] = bytemuck::cast_slice(&triangles);
                        spill.file.write_all(v).map_err(|e| e.to_string())?;
                        let triangle_base = vertex_base + v.len() as u64;
                        spill.file.write_all(t).map_err(|e| e.to_string())?;
                        let written = (v.len() + t.len()) as u64;
                        spill.at = triangle_base + t.len() as u64;
                        spill.since_check += written;
                        if spill.since_check > 8 << 30 {
                            spill.since_check = 0;
                            check_free(output, options.min_free_bytes)?;
                        }
                        (vertex_base, triangle_base)
                    };
                    finished
                        .lock()
                        .expect("finished")
                        .push((index, virtual_mesh, bases));
                }
                Ok(())
            })();
            budget.give_back(taken);
            result?;
            let mut done = done.lock().expect("done");
            *done += triangles;
            let share = *done as f64 / total.max(1) as f64 * 100.0;
            if (*done - triangles) * 50 / total.max(1) != *done * 50 / total.max(1) {
                println!(
                    "built {share:.0}% in {:.0}s",
                    built_start.elapsed().as_secs_f32()
                );
            }
            Ok(())
        })?;
    report.build_seconds = built_start.elapsed().as_secs_f32();
    let mut spill = spill.into_inner().expect("spill");
    spill.file.flush().map_err(|e| e.to_string())?;
    drop(spill);

    let mut finished = finished.into_inner().expect("finished");
    finished.sort_by_key(|entry| entry.0);
    let mut bases = Vec::with_capacity(finished.len());
    let mut meshes = Vec::with_capacity(finished.len());
    for (_, mesh, base) in finished {
        bases.push(base);
        meshes.push(mesh);
    }
    let scene = SceneAsset {
        meshes,
        materials,
        instances,
        texture_size: options.texture_size,
        viewpoint: Some(frame_bounds(&bounds)),
        ..Default::default()
    };
    report.meshes = scene.meshes.len();
    report.instances = scene.instances.len();
    report.unique_triangles = scene.unique_source_triangles();
    report.instanced_triangles = scene.total_source_triangles();

    let page_start = Instant::now();
    let mut store = SpillStore {
        file: BufReader::with_capacity(1 << 20, File::open(spill_path).map_err(|e| e.to_string())?),
        at: 0,
        bases,
    };
    let mut data =
        BufWriter::with_capacity(1 << 22, File::create(data_path).map_err(|e| e.to_string())?);
    let mut placed: Vec<(u64, u32)> = Vec::new();
    let mut data_at = 0u64;
    let mut pending: Vec<Vec<u8>> = Vec::new();
    let mut failure: Option<String> = None;
    let mut since_check = 0u64;
    let mut flush =
        |pending: &mut Vec<Vec<u8>>, failure: &mut Option<String>, placed: &mut Vec<(u64, u32)>| {
            let packed: Vec<Vec<u8>> = pending.par_iter().map(|raw| deflate_page(raw)).collect();
            pending.clear();
            for page in packed {
                placed.push((data_at, page.len() as u32));
                data_at += page.len() as u64;
                since_check += page.len() as u64;
                if failure.is_none() {
                    if let Err(error) = data.write_all(&page) {
                        *failure = Some(error.to_string());
                    }
                }
            }
            if since_check > 8 << 30 && failure.is_none() {
                since_check = 0;
                if let Err(error) = check_free(output, options.min_free_bytes) {
                    *failure = Some(error);
                }
            }
        };
    let mut paged = paginate_from(&scene, &mut store, &mut |page| {
        pending.push(page);
        if pending.len() >= 1024 {
            flush(&mut pending, &mut failure, &mut placed);
        }
    });
    flush(&mut pending, &mut failure, &mut placed);
    if let Some(error) = failure {
        return Err(error);
    }
    let pinned = paged.pinned_pages as usize;
    if paged.pages.len() - pinned != placed.len() {
        return Err("the pages written do not match the pages counted".to_owned());
    }
    for (info, &(offset, stored)) in paged.pages[pinned..].iter_mut().zip(&placed) {
        info.offset = offset;
        info.stored_bytes = stored;
    }
    report.page_seconds = page_start.elapsed().as_secs_f32();

    let texture_start = Instant::now();
    write_textures(
        &document,
        folder,
        &layers.keys,
        options,
        output,
        &mut paged,
        &mut data,
        &mut data_at,
    )?;
    report.texture_seconds = texture_start.elapsed().as_secs_f32();
    report.texture_layers = layers.keys.len();

    report.clusters = paged.clusters.len();
    report.pages = paged.pages.len();

    drop(store);
    data.flush().map_err(|e| e.to_string())?;
    drop(data);
    // The spill has done its job, and the final file needs the room.
    std::fs::remove_file(spill_path).map_err(|e| e.to_string())?;

    check_free(output, options.min_free_bytes)?;
    let head = encode_header_and_manifest(&paged);
    let mut file =
        BufWriter::with_capacity(1 << 22, File::create(output).map_err(|e| e.to_string())?);
    file.write_all(&head).map_err(|e| e.to_string())?;
    let mut source = File::open(data_path).map_err(|e| e.to_string())?;
    std::io::copy(&mut source, &mut file).map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())?;
    report.file_bytes = head.len() as u64 + data_at;
    Ok(report)
}

fn triangles_of(mesh: &gltf::Mesh<'_>, primitive: usize) -> u64 {
    mesh.primitives()
        .nth(primitive)
        .and_then(|p| p.indices())
        .map_or(0, |indices| indices.count() as u64 / 3)
}

/// Which images make up one texture layer.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LayerKey {
    base_color: usize,
    normal: Option<usize>,
    metallic_roughness: Option<usize>,
}

#[derive(Default)]
struct LayerTable {
    keys: Vec<LayerKey>,
    slots: HashMap<LayerKey, i32>,
}

impl LayerTable {
    fn slot(&mut self, key: LayerKey) -> i32 {
        *self.slots.entry(key).or_insert_with(|| {
            self.keys.push(key);
            self.keys.len() as i32 - 1
        })
    }
}

fn convert_material(
    document: &Document,
    index: Option<usize>,
    textures: bool,
    layers: &mut LayerTable,
    warnings: &mut Vec<String>,
) -> Material {
    let Some(source) = index.and_then(|i| document.materials().nth(i)) else {
        return Material {
            base_color: [1.0; 4],
            roughness: 1.0,
            metallic: 1.0,
            ..Default::default()
        };
    };
    let name = source.name().unwrap_or("unnamed");
    if source.alpha_mode() == gltf::material::AlphaMode::Blend {
        warnings.push(format!(
            "material '{name}': alpha mode Blend is drawn as opaque"
        ));
    }
    if source.emissive_texture().is_some() {
        warnings.push(format!(
            "material '{name}': emissive texture is ignored, only its factor is used"
        ));
    }
    let pbr = source.pbr_metallic_roughness();
    let [er, eg, eb] = source.emissive_factor();
    let image = |texture: gltf::Texture<'_>| texture.source().index();
    let base_color = pbr.base_color_texture().map(|t| image(t.texture()));
    let metallic_roughness = pbr.metallic_roughness_texture().map(|t| image(t.texture()));
    let key = base_color.filter(|_| textures).map(|base_color| LayerKey {
        base_color,
        normal: source.normal_texture().map(|t| image(t.texture())),
        metallic_roughness,
    });
    let masked = source.alpha_mode() == gltf::material::AlphaMode::Mask && key.is_some();
    if masked {
        let cutoff = source.alpha_cutoff().unwrap_or(0.5);
        if (cutoff - 0.5).abs() > 0.01 {
            warnings.push(format!(
                "material '{name}': its alpha cut-off of {cutoff} is drawn at 0.5, as its blocks are copied"
            ));
        }
    }
    // The layer's roughness is read as twice what it stores, so that a texture can raise it as
    // well as lower it. These maps store it plainly, so the factor makes up the difference.
    let has_roughness_map = key.is_some() && metallic_roughness.is_some();
    Material {
        base_color: pbr.base_color_factor(),
        emissive: [er, eg, eb],
        flags: if source.double_sided() {
            Material::TWO_SIDED
        } else {
            0
        } | if masked { Material::MASKED } else { 0 },
        roughness: pbr.roughness_factor() * if has_roughness_map { 0.5 } else { 1.0 },
        metallic: pbr.metallic_factor(),
        uv_scale: 1.0,
        normal_strength: source.normal_texture().map_or(0.0, |t| t.scale()),
        texture_layer: key.map_or(-1, |key| layers.slot(key)),
        ..Default::default()
    }
}

/// Opens one of a scene's images, which must be a KTX2 file of one of these formats.
fn open_image(
    document: &Document,
    image: usize,
    folder: &Path,
    formats: &[u32],
) -> Result<Ktx2, String> {
    let source = document.images().nth(image).ok_or("no such image")?;
    let gltf::image::Source::Uri { uri, .. } = source.source() else {
        return Err("large scenes keep their images in files".to_owned());
    };
    let path = folder.join(uri);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let texture = Ktx2::parse(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    if !formats.contains(&texture.format) {
        return Err(format!(
            "{}: format {} is not one this reads",
            path.display(),
            texture.format
        ));
    }
    Ok(texture)
}

/// The levels of one texture layer, all block compressed.
fn compose_layer(
    document: &Document,
    folder: &Path,
    key: LayerKey,
    size: u32,
) -> Result<TextureLayer, String> {
    let color_mips = open_image(
        document,
        key.base_color,
        folder,
        &[ktx2::BC7_SRGB, ktx2::BC7_UNORM],
    )?
    .fit_square(size)?;
    let levels = color_mips.len() as u32;
    let flat = |red: u8, green: u8| -> Vec<Vec<u8>> {
        (0..levels)
            .map(|level| ktx2::constant_bc5(size >> level, red, green))
            .collect()
    };
    let normal_mips = match key.normal {
        Some(image) => {
            let mut levels =
                open_image(document, image, folder, &[ktx2::BC5_UNORM])?.fit_square(size)?;
            // glTF's green points towards decreasing V, and the renderer's derives its frame
            // from the direction of increasing V.
            for level in &mut levels {
                ktx2::invert_bc5_green(level);
            }
            levels
        }
        None => flat(128, 128),
    };
    let material_mips = match key.metallic_roughness {
        Some(image) => open_image(document, image, folder, &[ktx2::BC5_UNORM])?.fit_square(size)?,
        None => flat(128, 255),
    };
    Ok(TextureLayer {
        color_mips,
        normal_mips,
        material_mips,
    })
}

/// Places each layer's upper levels in the data area, and keeps the rest in the manifest.
#[allow(clippy::too_many_arguments)]
fn write_textures(
    document: &Document,
    folder: &Path,
    keys: &[LayerKey],
    options: &LargeOptions,
    output: &Path,
    paged: &mut PagedScene,
    data: &mut impl Write,
    data_at: &mut u64,
) -> Result<(), String> {
    /// Layers composed at a time, which bounds the memory they take.
    const LAYERS_AT_ONCE: usize = 32;
    let size = options.texture_size;
    paged.texture_size = size;
    paged.texture_encoding = TextureEncoding::Block;
    let upper = upper_mip_count(size);
    let mut base = Vec::with_capacity(keys.len());
    let mut offsets = Vec::with_capacity(keys.len() * upper);
    for (group, batch) in keys.chunks(LAYERS_AT_ONCE).enumerate() {
        let composed = batch
            .par_iter()
            .map(|&key| compose_layer(document, folder, key, size))
            .collect::<Result<Vec<_>, _>>()?;
        for layer in composed {
            for mip in 0..upper {
                offsets.push(*data_at);
                for level in [
                    &layer.color_mips[mip],
                    &layer.normal_mips[mip],
                    &layer.material_mips[mip],
                ] {
                    data.write_all(level).map_err(|e| e.to_string())?;
                    *data_at += level.len() as u64;
                }
            }
            base.push(TextureLayer {
                color_mips: layer.color_mips[upper..].to_vec(),
                normal_mips: layer.normal_mips[upper..].to_vec(),
                material_mips: layer.material_mips[upper..].to_vec(),
            });
        }
        if group % 16 == 15 {
            println!(
                "textures: {} of {}",
                (group + 1) * LAYERS_AT_ONCE,
                keys.len()
            );
            check_free(output, options.min_free_bytes)?;
        }
    }
    paged.base_textures = base;
    paged.upper_mip_offsets = offsets;
    Ok(())
}
