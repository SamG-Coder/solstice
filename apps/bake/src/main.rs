//! Bakes scenes into Solstice scene files: the procedural demo, or an imported model.
//!
//! ```text
//! bake [--quality draft|web|high] [--output FILE]
//! bake --gltf MODEL.glb [--fit METRES] [--ground] [--texture-size N] [--output FILE]
//! bake --scene SCENE.txt [--texture-size N] [--output FILE]
//! ```
//!
//! Textures are block compressed unless `--raw-textures` is given.

mod meshes;
mod noise;
mod scene;
mod scene_file;
mod textures;

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
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(options)
}

fn load_scene(options: &Options) -> Result<SceneAsset, String> {
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

fn main() -> Result<(), String> {
    let options = parse_args()?;
    let start = Instant::now();
    let mut scene = load_scene(&options)?;
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

    let bytes = solstice_asset::file::encode(&scene);
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
