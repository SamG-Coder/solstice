//! Bakes scenes into Solstice scene files: the procedural demo, or an imported model.
//!
//! ```text
//! bake [--quality draft|web|high] [--output FILE]
//! bake --gltf MODEL.glb [--fit METRES] [--ground] [--texture-size N] [--output FILE]
//! bake --scene SCENE.txt [--texture-size N] [--output FILE]
//! bake --gallery [--texture-size N] [--output FILE]
//! bake --forest [--texture-size N] [--output FILE]
//! bake --plants [--species FILE.species]... [--grass FILE.grass]... [--texture-size N] [--output FILE]
//! bake --check-character CHARACTER.glb [--roles RIG.roles]
//! bake --scene SCENE.txt --show-collision [--output FILE]
//! ```
//!
//! `--check-character` bakes nothing: it checks a rigged character as the export tool wrote it
//! (`tools/blender/solstice_export.py`), prints what it holds and what is wrong with it, and
//! fails if anything is.
//!
//! `--plants` grows every built-in species and breed of grass, then each species and grass file
//! given, so either can be edited and looked at without touching code.
//!
//! Textures are block compressed unless `--raw-textures` is given.

mod collision_view;
mod forest;
mod gallery;
mod meshes;
mod plants;
mod scene;
mod scene_file;

use scene::Quality;
use solstice_asset::SceneAsset;
use solstice_import::{import_gltf, ImportOptions};
use std::path::PathBuf;
use std::time::Instant;

struct Options {
    quality: Quality,
    output: PathBuf,
    gltf: Option<PathBuf>,
    scene_file: Option<PathBuf>,
    import: ImportOptions,
    /// Replaces the size a scene file asks for.
    scene_texture_size: Option<u32>,
    raw_textures: bool,
    /// Shows every procedural surface, instead of the demo scene.
    gallery: bool,
    /// The procedural meadow and wood, instead of the demo scene.
    forest: bool,
    /// Bakes what the scene collides with, instead of how it looks.
    show_collision: bool,
    /// A row of every species, instead of the demo scene.
    plants: bool,
    /// Species files to add to the row of plants.
    species: Vec<PathBuf>,
    /// Grass files to add to the row of grass.
    grass: Vec<PathBuf>,
    /// A character to check, and its rig's role file.
    check_character: Option<PathBuf>,
    roles: Option<PathBuf>,
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        quality: Quality::High,
        output: PathBuf::from("assets/temple.solstice"),
        gltf: None,
        scene_file: None,
        import: ImportOptions::default(),
        scene_texture_size: None,
        raw_textures: false,
        gallery: false,
        forest: false,
        show_collision: false,
        plants: false,
        species: Vec::new(),
        grass: Vec::new(),
        check_character: None,
        roles: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--quality" => {
                options.quality = match value()?.as_str() {
                    "draft" => Quality::Draft,
                    "web" => Quality::Web,
                    "high" => Quality::High,
                    other => return Err(format!("unknown quality: {other}")),
                };
            }
            "--output" => options.output = PathBuf::from(value()?),
            "--scene" => options.scene_file = Some(PathBuf::from(value()?)),
            "--check-character" => options.check_character = Some(PathBuf::from(value()?)),
            "--roles" => options.roles = Some(PathBuf::from(value()?)),
            "--gltf" => options.gltf = Some(PathBuf::from(value()?)),
            "--fit" => {
                let metres = value()?.parse().map_err(|e| format!("--fit: {e}"))?;
                options.import.fit_size = Some(metres);
            }
            "--ground" => options.import.ground = true,
            "--texture-size" => {
                let size = value()?
                    .parse()
                    .map_err(|e| format!("--texture-size: {e}"))?;
                options.import.texture_size = size;
                options.scene_texture_size = Some(size);
            }
            "--raw-textures" => options.raw_textures = true,
            "--gallery" => options.gallery = true,
            "--forest" => options.forest = true,
            "--show-collision" => options.show_collision = true,
            "--plants" => options.plants = true,
            "--species" => {
                options.plants = true;
                options.species.push(PathBuf::from(value()?));
            }
            "--grass" => {
                options.plants = true;
                options.grass.push(PathBuf::from(value()?));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(options)
}

fn load_scene(options: &Options) -> Result<SceneAsset, String> {
    if options.forest {
        return Ok(forest::build(options.scene_texture_size.unwrap_or(1024)));
    }
    if options.plants {
        return plants::build(
            options.scene_texture_size.unwrap_or(1024),
            &options.species,
            &options.grass,
        );
    }
    if options.gallery {
        return Ok(gallery::build(options.scene_texture_size.unwrap_or(1024)));
    }
    if let Some(path) = &options.scene_file {
        let (scene, mut warnings) = scene_file::build(path, options.scene_texture_size)?;
        warnings.sort();
        warnings.dedup();
        for warning in warnings {
            println!("warning: {warning}");
        }
        return Ok(scene);
    }
    let Some(path) = &options.gltf else {
        return Ok(scene::build_temple(options.quality));
    };
    println!("importing {}...", path.display());
    let imported = import_gltf(path, &options.import).map_err(|e| e.to_string())?;
    for warning in &imported.warnings {
        println!("warning: {warning}");
    }
    Ok(imported.scene)
}

/// Checks a character file, and fails if anything is wrong with it.
fn check_character(path: &std::path::Path, roles: Option<&PathBuf>) -> Result<(), String> {
    use solstice_import::character::{check_character, Roles};
    let roles = match roles {
        Some(file) => {
            let text =
                std::fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
            Some(Roles::parse(&text).map_err(|e| format!("{}: {e}", file.display()))?)
        }
        None => None,
    };
    let report = check_character(path, roles.as_ref())?;
    println!("{}: {}", path.display(), report.summary());
    if report.problems.is_empty() {
        Ok(())
    } else {
        Err(format!("{} is not ready for the engine", path.display()))
    }
}

fn main() -> Result<(), String> {
    let options = parse_args()?;
    if let Some(path) = &options.check_character {
        return check_character(path, options.roles.as_ref());
    }
    let start = Instant::now();
    let mut scene = load_scene(&options)?;
    if options.show_collision {
        scene = collision_view::collision_scene(&scene)?;
    }
    if !options.raw_textures {
        println!("compressing {} texture layers...", scene.textures.len());
        solstice_builder::compress_textures(&mut scene);
    }

    println!(
        "{} meshes, {} instances, {} unique triangles, {} instanced triangles",
        scene.meshes.len(),
        scene.instances.len(),
        scene.unique_source_triangles(),
        scene.total_source_triangles(),
    );

    let bytes = solstice_asset::file::encode(&solstice_builder::paginate(&scene));
    if let Some(parent) = options.output.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&options.output, &bytes).map_err(|e| e.to_string())?;
    println!(
        "wrote {} ({:.1} MB) in {:.1}s",
        options.output.display(),
        bytes.len() as f64 / 1.0e6,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}
