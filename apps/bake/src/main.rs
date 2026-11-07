//! Bakes scenes into Solstice scene files: the procedural demo, or an imported model.
//!
//! ```text
//! bake [--quality draft|web|high] [--output FILE]
//! bake --gltf MODEL.glb [--fit METRES] [--ground] [--texture-size N] [--output FILE]
//! bake --scene SCENE.txt [--output FILE]
//! ```

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
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        quality: Quality::High,
        output: PathBuf::from("assets/temple.solstice"),
        gltf: None,
        scene_file: None,
        import: ImportOptions::default(),
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
                options.import.texture_size = value()?
                    .parse()
                    .map_err(|e| format!("--texture-size: {e}"))?;
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(options)
}

fn load_scene(options: &Options) -> Result<SceneAsset, String> {
    if let Some(path) = &options.scene_file {
        let (scene, mut warnings) = scene_file::build(path)?;
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
    let scene = load_scene(&options)?;

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
