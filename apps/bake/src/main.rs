//! Bakes the procedural demo scene into a Solstice scene file.
//!
//! ```text
//! bake [--quality draft|web|high] [--output assets/temple.solstice]
//! ```

mod meshes;
mod noise;
mod scene;
mod textures;

use scene::Quality;
use std::path::PathBuf;
use std::time::Instant;

struct Options {
    quality: Quality,
    output: PathBuf,
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        quality: Quality::High,
        output: PathBuf::from("assets/temple.solstice"),
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
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(options)
}

fn main() -> Result<(), String> {
    let options = parse_args()?;
    let start = Instant::now();
    let scene = scene::build_temple(options.quality);

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
