//! Generates the harbour scene's kit as glTF, because no CC0 scan exists of what it holds.
//!
//! ```text
//! townkit [--output FOLDER]     # target/town by default
//! ```
//!
//! `ground.gltf` is the land and seabed, and `stone.gltf` the dressed stone, both placed whole.
//! `town.gltf` holds named parts, placed by `scenes/harbour.txt`. `docs/harbour.md` is the design.
//! The textures come first: `python scripts/fetch-scans.py scenes/harbour.surfaces 2k`.

mod geometry;
mod gltf;
mod ground;
mod stone;
mod town;

use gltf::{File, Plain};
use ground::{Surface, PATCHES, SURFACES};
use std::path::{Path, PathBuf};

const fn plain(name: &'static str, color: [f32; 3], roughness: f32) -> Plain {
    Plain {
        name,
        color,
        roughness,
    }
}

fn ground_material(surface: Surface) -> Plain {
    match surface {
        Surface::Sand => plain("sand", [0.42, 0.36, 0.26], 0.95),
        Surface::Grass => plain("grass", [0.11, 0.16, 0.06], 0.95),
        Surface::Dirt => plain("dirt", [0.2, 0.16, 0.11], 0.95),
        Surface::Rock => plain("rock", [0.17, 0.16, 0.15], 0.9),
    }
}

/// The ground, each patch split by surface.
fn ground_file() -> File {
    let mut file = File::default();
    let materials = SURFACES.map(|surface| file.material(&ground_material(surface)));
    for row in 0..PATCHES {
        for column in 0..PATCHES {
            let geometry = ground::patch(column, row);
            let pieces: Vec<_> = geometry.iter().zip(materials).collect();
            file.part(&format!("ground_{column}_{row}"), &pieces);
        }
    }
    file
}

/// The solid stonework and the massing. `stonework` adds the material they are drawn in.
fn town_file(stonework: impl FnOnce(&mut File) -> usize) -> File {
    let mut file = File::default();
    let granite = stonework(&mut file);
    let limewash = file.material(&plain("limewash", [0.72, 0.7, 0.64], 0.9));
    let terracotta = file.material(&plain("terracotta", [0.42, 0.18, 0.1], 0.8));
    let timber = file.material(&plain("timber", [0.14, 0.09, 0.06], 0.8));

    file.part("quay", &[(&town::quay(), granite)]);
    file.part("mole", &[(&town::mole(), granite)]);
    let massing = town::massing();
    file.part(
        "massing",
        &[
            (&massing.walls, limewash),
            (&massing.roofs, terracotta),
            (&massing.stone, granite),
            (&massing.timber, timber),
        ],
    );
    file
}

fn output_folder() -> Result<PathBuf, String> {
    let mut args = std::env::args().skip(1);
    let mut folder = PathBuf::from("target/town");
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => folder = args.next().ok_or("--output needs a folder")?.into(),
            other => return Err(format!("unknown option '{other}'")),
        }
    }
    Ok(folder)
}

fn write_kit(folder: &Path) -> std::io::Result<()> {
    ground_file().write(folder, "ground")?;
    town_file(|file| file.textured(&stone::FLAGS)).write(folder, "town")?;
    stone::file(&folder.join("../scan"))?.write(folder, "stone")
}

fn main() -> Result<(), String> {
    let folder = output_folder()?;
    write_kit(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    println!("wrote the kit to {}", folder.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use solstice_import::{import_gltf, ImportOptions};

    #[test]
    fn the_importer_reads_the_town_and_finds_every_part_by_name() {
        let folder = std::env::temp_dir().join(format!("townkit-{}", std::process::id()));
        // In a plain material, since a test has no fetched textures to read.
        town_file(|file| file.material(&plain("granite", [0.3, 0.29, 0.27], 0.85)))
            .write(&folder, "town")
            .unwrap();
        let imported = import_gltf(folder.join("town.gltf"), &ImportOptions::default());
        std::fs::remove_dir_all(&folder).unwrap();
        let imported = imported.unwrap();
        assert!(imported.warnings.is_empty(), "{:?}", imported.warnings);
        for part in ["quay", "mole", "massing"] {
            assert!(imported.parts.contains_key(part), "{part} is missing");
        }
        assert_eq!(imported.parts["massing"].len(), 4, "one per material");
    }
}
