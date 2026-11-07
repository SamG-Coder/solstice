//! Builds a scene from a small text file listing models and where to place them.
//!
//! ```text
//! # Paths are relative to this file. Everything after a # is ignored.
//! texture_size 1024
//! terrain 800 1024 0.5 60         # size, cells per side, height, X where the shore begins
//! water -0.6 0.02 0.09 0.11       # height, then colour
//! camera 40 5 20 1.2 -0.05        # x y z yaw pitch
//! light 0 2 0 16 40 14 3          # x y z range r g b
//! model rocks/rock.gltf 0 0 0 45 2     # path x y z yaw_degrees scale
//! model rocks/rock.gltf 9 ground 0 45 2     # standing on the terrain at this spot
//! model rocks/rock.gltf 9 ground:0.2 0 45 2 # with its lowest 20% buried, as rocks are
//! part kit/kit.gltf wall_01 0 0 0 90 1  # one named piece of a model, by its own origin
//! ```
//!
//! A model placed several times is imported once and shared, so repeats cost almost nothing.

use crate::meshes;
use glam::{Mat4, Quat, Vec3};
use solstice_asset::scene::Viewpoint;
use solstice_asset::{Instance, Light, Material, SceneAsset};
use solstice_builder::{build_virtual_mesh, BuildOptions};
use solstice_import::{append_model, import_gltf, ImportOptions, ModelPlacements};
use solstice_procgen::noise::fbm;
use solstice_procgen::surfaces;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One line of a scene file, parsed.
#[derive(Debug, PartialEq)]
enum Command {
    TextureSize(u32),
    Terrain {
        size: f32,
        cells: u32,
        height: f32,
        shore_x: f32,
    },
    Water {
        height: f32,
        color: [f32; 3],
    },
    Camera(Viewpoint),
    Light(Light),
    Model {
        path: String,
        position: Vec3,
        /// If set, `position.y` is ignored and the model stands on the terrain, with this
        /// fraction of its height below ground.
        seat: Option<f32>,
        yaw_degrees: f32,
        scale: f32,
    },
    Part {
        path: String,
        name: String,
        position: Vec3,
        yaw_degrees: f32,
        scale: f32,
    },
}

fn parse_line(line: &str) -> Result<Option<Command>, String> {
    let line = line.split('#').next().unwrap_or("").trim();
    let mut words = line.split_whitespace();
    let Some(keyword) = words.next() else {
        return Ok(None);
    };
    let values: Vec<&str> = words.collect();
    let number = |index: usize| -> Result<f32, String> {
        let text = values
            .get(index)
            .ok_or_else(|| format!("'{keyword}' is missing value {}", index + 1))?;
        text.parse()
            .map_err(|_| format!("'{text}' is not a number"))
    };
    let expect = |count: usize| -> Result<(), String> {
        if values.len() == count {
            Ok(())
        } else {
            Err(format!(
                "'{keyword}' takes {count} values, found {}",
                values.len()
            ))
        }
    };
    Ok(Some(match keyword {
        "texture_size" => {
            expect(1)?;
            Command::TextureSize(number(0)? as u32)
        }
        "water" => {
            expect(4)?;
            Command::Water {
                height: number(0)?,
                color: [number(1)?, number(2)?, number(3)?],
            }
        }
        "part" => {
            expect(7)?;
            Command::Part {
                path: values[0].to_string(),
                name: values[1].to_string(),
                position: Vec3::new(number(2)?, number(3)?, number(4)?),
                yaw_degrees: number(5)?,
                scale: number(6)?,
            }
        }
        "terrain" => {
            expect(4)?;
            Command::Terrain {
                size: number(0)?,
                cells: number(1)? as u32,
                height: number(2)?,
                shore_x: number(3)?,
            }
        }
        "camera" => {
            expect(5)?;
            Command::Camera(Viewpoint {
                position: [number(0)?, number(1)?, number(2)?],
                yaw: number(3)?,
                pitch: number(4)?,
            })
        }
        "light" => {
            expect(7)?;
            Command::Light(Light {
                position: [number(0)?, number(1)?, number(2)?],
                range: number(3)?,
                intensity: [number(4)?, number(5)?, number(6)?],
                _pad: 0,
            })
        }
        "model" => {
            expect(6)?;
            let (height, seat) = parse_height(values[2])?;
            Command::Model {
                path: values[0].to_string(),
                position: Vec3::new(number(1)?, height, number(3)?),
                seat,
                yaw_degrees: number(4)?,
                scale: number(5)?,
            }
        }
        other => return Err(format!("unknown command '{other}'")),
    }))
}

/// Reads a model's height: a number, `ground`, or `ground:FRACTION`.
fn parse_height(text: &str) -> Result<(f32, Option<f32>), String> {
    /// Enough to close the gap left by small bumps, without visibly sinking anything.
    const STANDING: f32 = 0.02;
    match text.split_once(':') {
        None if text == "ground" => Ok((0.0, Some(STANDING))),
        Some(("ground", fraction)) => {
            let fraction: f32 = fraction
                .parse()
                .map_err(|_| format!("'{fraction}' is not a number"))?;
            if (0.0..1.0).contains(&fraction) {
                Ok((0.0, Some(fraction)))
            } else {
                Err(format!("'{text}': the buried fraction must be below 1"))
            }
        }
        _ => text
            .parse()
            .map(|height| (height, None))
            .map_err(|_| format!("'{text}' is not a number")),
    }
}

/// Every model vertex, thinned out, in the model's own space.
fn model_points(scene: &SceneAsset) -> Vec<Vec3> {
    const MAX_PER_INSTANCE: usize = 6000;
    let mut points = Vec::new();
    for instance in &scene.instances {
        let mesh = &scene.meshes[instance.mesh as usize];
        let stride = (mesh.vertices.len() / MAX_PER_INSTANCE).max(1);
        for vertex in mesh.vertices.iter().step_by(stride) {
            let grid = Vec3::new(
                f32::from(vertex.data[0]),
                f32::from(vertex.data[1]),
                f32::from(vertex.data[2]),
            );
            let local = mesh.quant_origin + grid * mesh.quant_step;
            points.push(instance.transform.transform_point3(local));
        }
    }
    points
}

/// The height to place a model at so that its lowest `buried` fraction is wholly below `ground`.
/// Rounded rock touches flat ground at a point and appears to float, so it has to sink into it,
/// and on a slope this leaves no gap on the downhill side.
fn seat_height(
    points: &[Vec3],
    position: Vec3,
    yaw_degrees: f32,
    scale: f32,
    buried: f32,
    ground: impl Fn(f32, f32) -> f32,
) -> f32 {
    let transform = placement_transform(Vec3::new(position.x, 0.0, position.z), yaw_degrees, scale);
    let placed: Vec<Vec3> = points
        .iter()
        .map(|&p| transform.transform_point3(p))
        .collect();
    let lowest = placed.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let highest = placed.iter().map(|p| p.y).fold(f32::MIN, f32::max);
    let band = lowest + (highest - lowest) * buried;
    placed
        .iter()
        .filter(|p| p.y <= band)
        .map(|p| ground(p.x, p.z) - p.y)
        .fold(f32::MAX, f32::min)
}

/// Gently rolling ground that falls away into the sea beyond `shore_x`.
fn terrain_height(x: f32, z: f32, amplitude: f32, shore_x: f32) -> f32 {
    let p = Vec3::new(x, 0.0, z);
    let broad = fbm(p * 0.015, 4, 301) * amplitude;
    let ripples = fbm(p * 0.25, 4, 303) * 0.04;
    // Bays and points, so that the waterline wanders rather than running ruler-straight.
    let wander = fbm(Vec3::new(0.0, 7.0, z) * 0.011, 4, 305) * 28.0;
    let t = ((x - shore_x - wander) / 90.0).clamp(0.0, 1.0);
    let seabed = t * t * (3.0 - 2.0 * t) * 9.0;
    broad + ripples - seabed
}

/// Adds the terrain as one mesh, with its own ground material and texture.
fn add_terrain(scene: &mut SceneAsset, size: f32, cells: u32, height: f32, shore_x: f32) -> i32 {
    let mesh = meshes::terrain(size, cells, |x, z| terrain_height(x, z, height, shore_x));
    let layer = scene.textures.len() as i32;
    scene
        .textures
        .push(surfaces::ground(0).bake(scene.texture_size));
    scene.materials.push(Material {
        // The texture is a bright desert. Darkened to real sand, about 0.35 albedo, so that
        // it sits correctly next to scanned materials and does not skew auto exposure.
        base_color: [0.55, 0.53, 0.5, 1.0],
        roughness: 1.0,
        uv_scale: 0.3,
        normal_strength: 0.5,
        texture_layer: layer,
        macro_variation: 1.0,
        ..Default::default()
    });
    scene.instances.push(Instance {
        mesh: scene.meshes.len() as u32,
        material: scene.materials.len() as u32 - 1,
        transform: Mat4::IDENTITY,
    });
    scene
        .meshes
        .push(build_virtual_mesh(&mesh, &BuildOptions::default()));
    layer
}

/// Adds a flat, glossy water surface. The ground texture supplies its ripples, at a scale
/// where the pebbles read as small waves, and is too dark to be seen as colour.
fn add_water(scene: &mut SceneAsset, height: f32, color: [f32; 3], ripples_layer: i32) {
    // Ripples need fine UVs, which the 16-bit grid only allows over a limited area.
    const SIZE: f32 = 4000.0;
    // Beyond that, a plain surface carries the sea out to the horizon, just below the first.
    const HORIZON_SIZE: f32 = 80000.0;
    let mesh = meshes::terrain(SIZE, 96, |_, _| height);
    scene.materials.push(Material {
        base_color: [color[0], color[1], color[2], 1.0],
        roughness: 0.14,
        uv_scale: 0.35,
        normal_strength: 0.22,
        texture_layer: ripples_layer,
        ..Default::default()
    });
    scene.instances.push(Instance {
        mesh: scene.meshes.len() as u32,
        material: scene.materials.len() as u32 - 1,
        transform: Mat4::IDENTITY,
    });
    scene
        .meshes
        .push(build_virtual_mesh(&mesh, &BuildOptions::default()));

    let far = meshes::terrain(HORIZON_SIZE, 32, |_, _| height - 0.3);
    scene.materials.push(Material {
        base_color: [color[0], color[1], color[2], 1.0],
        roughness: 0.2,
        ..Default::default()
    });
    scene.instances.push(Instance {
        mesh: scene.meshes.len() as u32,
        material: scene.materials.len() as u32 - 1,
        transform: Mat4::IDENTITY,
    });
    scene
        .meshes
        .push(build_virtual_mesh(&far, &BuildOptions::default()));
}

fn placement_transform(position: Vec3, yaw_degrees: f32, scale: f32) -> Mat4 {
    let rotation = Quat::from_rotation_y(yaw_degrees.to_radians());
    Mat4::from_scale_rotation_translation(Vec3::splat(scale), rotation, position)
}

/// The placements for a model, registering it in first-seen order on first use.
fn placements_of<'a>(
    order: &mut Vec<PathBuf>,
    models: &'a mut HashMap<PathBuf, ModelPlacements>,
    base: &Path,
    model: String,
) -> &'a mut ModelPlacements {
    let full = base.join(model);
    if !order.contains(&full) {
        order.push(full.clone());
    }
    models.entry(full).or_default()
}

/// Builds the scene described by the file at `path`. Returns it with any import warnings.
pub fn build(path: &Path, texture_size: Option<u32>) -> Result<(SceneAsset, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));

    let mut scene = SceneAsset {
        texture_size: 1024,
        ..Default::default()
    };
    let mut terrain = None;
    let mut water = None;
    // Grouped by file, keeping first-seen order, so that each model is imported only once.
    let mut order: Vec<PathBuf> = Vec::new();
    let mut models: HashMap<PathBuf, ModelPlacements> = HashMap::new();
    // Placed once the model has been imported, since its shape decides the height.
    let mut seated: HashMap<PathBuf, Vec<(Vec3, f32, f32, f32)>> = HashMap::new();
    for (number, line) in text.lines().enumerate() {
        let command =
            parse_line(line).map_err(|e| format!("{}:{}: {e}", path.display(), number + 1))?;
        match command {
            None => {}
            Some(Command::TextureSize(size)) => scene.texture_size = size,
            Some(Command::Terrain {
                size,
                cells,
                height,
                shore_x,
            }) => {
                terrain = Some((size, cells, height, shore_x));
            }
            Some(Command::Water { height, color }) => water = Some((height, color)),
            Some(Command::Camera(viewpoint)) => scene.viewpoint = Some(viewpoint),
            Some(Command::Light(light)) => scene.lights.push(light),
            Some(Command::Model {
                path: model,
                position,
                seat,
                yaw_degrees,
                scale,
            }) => {
                let full = base.join(&model);
                let placements = placements_of(&mut order, &mut models, base, model);
                match seat {
                    Some(buried) => {
                        seated
                            .entry(full)
                            .or_default()
                            .push((position, yaw_degrees, scale, buried))
                    }
                    None => {
                        placements
                            .whole
                            .push(placement_transform(position, yaw_degrees, scale))
                    }
                }
            }
            Some(Command::Part {
                path: model,
                name,
                position,
                yaw_degrees,
                scale,
            }) => {
                let transform = placement_transform(position, yaw_degrees, scale);
                placements_of(&mut order, &mut models, base, model)
                    .parts
                    .push((name, transform));
            }
        }
    }

    if let Some(size) = texture_size {
        scene.texture_size = size;
    }
    let options = ImportOptions {
        texture_size: scene.texture_size,
        ..Default::default()
    };
    let mut warnings = Vec::new();
    for model in &order {
        println!("importing {}...", model.display());
        let imported =
            import_gltf(model, &options).map_err(|e| format!("{}: {e}", model.display()))?;
        let name = model
            .file_stem()
            .map_or(String::new(), |s| s.to_string_lossy().to_string());
        warnings.extend(imported.warnings.iter().map(|w| format!("{name}: {w}")));
        if let Some(list) = seated.get(model) {
            let Some((_, _, amplitude, shore_x)) = terrain else {
                return Err(format!("{name}: 'ground' needs a terrain line"));
            };
            // Water hides whatever is below it, so the sea floor does not count.
            let water_level = water.map_or(f32::MIN, |(level, _)| level);
            let ground = |x: f32, z: f32| terrain_height(x, z, amplitude, shore_x).max(water_level);
            let points = model_points(&imported.scene);
            let placements = models.get_mut(model).expect("registered while parsing");
            for &(position, yaw_degrees, scale, buried) in list {
                let y = seat_height(&points, position, yaw_degrees, scale, buried, ground);
                println!("  at {:.0} {:.0}: height {y:.2}", position.x, position.z);
                placements.whole.push(placement_transform(
                    Vec3::new(position.x, y, position.z),
                    yaw_degrees,
                    scale,
                ));
            }
        }
        append_model(&mut scene, imported, &models[model])
            .map_err(|e| format!("{}: {e}", model.display()))?;
    }

    if let Some((size, cells, height, shore_x)) = terrain {
        let ground_layer = add_terrain(&mut scene, size, cells, height, shore_x);
        if let Some((level, color)) = water {
            add_water(&mut scene, level, color, ground_layer);
        }
    }
    Ok((scene, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_lines_and_comments_are_ignored() {
        assert_eq!(parse_line(""), Ok(None));
        assert_eq!(parse_line("   # just a note"), Ok(None));
    }

    #[test]
    fn model_line_parses_every_field() {
        let command = parse_line("model a/b.gltf 1 2 3 90 0.5  # note").unwrap();
        assert_eq!(
            command,
            Some(Command::Model {
                path: "a/b.gltf".to_string(),
                position: Vec3::new(1.0, 2.0, 3.0),
                seat: None,
                yaw_degrees: 90.0,
                scale: 0.5,
            })
        );
    }

    #[test]
    fn a_height_can_ask_for_the_ground() {
        assert_eq!(parse_height("1.5"), Ok((1.5, None)));
        assert_eq!(parse_height("ground:0.25"), Ok((0.0, Some(0.25))));
        assert!(parse_height("ground").unwrap().1.is_some());
        assert!(parse_height("ground:2").is_err());
        assert!(parse_height("grund").is_err());
    }

    /// A tall column, narrowest at its foot: one point at the bottom, then a wide ring.
    fn column() -> Vec<Vec3> {
        vec![
            Vec3::ZERO,
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
        ]
    }

    #[test]
    fn seating_on_flat_ground_rests_the_lowest_point_on_it() {
        let y = seat_height(&column(), Vec3::ZERO, 0.0, 1.0, 0.0, |_, _| 2.0);
        assert!((y - 2.0).abs() < 1.0e-5);
    }

    #[test]
    fn burying_sinks_the_model_by_that_share_of_its_height() {
        let y = seat_height(&column(), Vec3::ZERO, 0.0, 1.0, 0.2, |_, _| 0.0);
        assert!(
            (y + 1.0).abs() < 1.0e-5,
            "the ring at height 1 is at ground level: {y}"
        );
    }

    #[test]
    fn seating_leaves_no_gap_on_a_slope() {
        let ground = |x: f32, _: f32| x * 0.5;
        let y = seat_height(&column(), Vec3::ZERO, 0.0, 1.0, 0.2, ground);
        for point in &column()[..3] {
            assert!(
                point.y + y <= ground(point.x, point.z) + 1.0e-5,
                "{point} floats"
            );
        }
    }

    #[test]
    fn part_and_water_lines_parse() {
        assert_eq!(
            parse_line("part kit.gltf wall_01 1 2 3 90 2").unwrap(),
            Some(Command::Part {
                path: "kit.gltf".to_string(),
                name: "wall_01".to_string(),
                position: Vec3::new(1.0, 2.0, 3.0),
                yaw_degrees: 90.0,
                scale: 2.0,
            })
        );
        assert_eq!(
            parse_line("water -1 0.1 0.2 0.3").unwrap(),
            Some(Command::Water {
                height: -1.0,
                color: [0.1, 0.2, 0.3],
            })
        );
    }

    #[test]
    fn the_seabed_falls_below_the_shore_only_on_the_sea_side() {
        assert!(
            terrain_height(-50.0, 0.0, 0.0, 40.0).abs() < 0.1,
            "inland stays level"
        );
        assert!(
            terrain_height(10.0, 0.0, 0.0, 40.0).abs() < 0.1,
            "level behind the shoreline, wherever it wanders"
        );
        assert!(
            terrain_height(130.0, 0.0, 0.0, 40.0) < -8.0,
            "deep offshore"
        );
    }

    #[test]
    fn mistakes_are_reported_clearly() {
        assert!(parse_line("nonsense 1 2")
            .unwrap_err()
            .contains("unknown command"));
        assert!(parse_line("camera 1 2 3")
            .unwrap_err()
            .contains("takes 5 values"));
        assert!(parse_line("terrain 1 x 3 4")
            .unwrap_err()
            .contains("not a number"));
    }
}
