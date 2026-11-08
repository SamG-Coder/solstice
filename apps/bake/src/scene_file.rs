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
//! actor cove/gate.actor 20.7 0 0 90 locked=true  # an actor, with its properties' values
//! input cove/play.input                # the input actions, if not the built-in ones
//! ```
//!
//! Actors, their scripts and the input actions travel inside the scene file, as its gameplay
//! (`solstice_game::scene`). An actor's mesh is placed here, like a model, and moves with it.
//!
//! A model placed several times is imported once and shared, so repeats cost almost nothing.

use crate::meshes;
use glam::{Mat4, Quat, Vec3};
use solstice_asset::scene::Viewpoint;
use solstice_asset::{GameFile, Instance, Light, Material, SceneAsset, Sea};
use solstice_builder::{build_virtual_mesh, BuildOptions};
use solstice_game::{resolve, ActorFile, ActorLine, InputMap, SceneGame, GAME_FILE};
use solstice_import::{append_model, import_gltf, ImportOptions, ModelPlacements};
use solstice_procgen::noise::fbm;
use solstice_procgen::surface::TILE_METRES;
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
    /// An open sea at `height`: a built-in ocean's name, or a `.ocean` file beside the scene.
    Ocean {
        height: f32,
        ocean: String,
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
    Actor(ActorLine),
    Input(String),
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
        "ocean" => {
            expect(2)?;
            Command::Ocean {
                height: number(0)?,
                ocean: values[1].to_string(),
            }
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
        "actor" => Command::Actor(ActorLine::parse(&values)?),
        "input" => {
            expect(1)?;
            Command::Input(values[0].to_string())
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

/// How far inland the coast around the rest of the island lies from the origin, before its bays
/// and points. With the slope down to the seabed, this keeps the land well inside the terrain
/// square, so that its edges lie under the sea rather than showing as straight lines.
const ISLAND_RADIUS: f32 = 270.0;
/// How far the seabed lies beyond the waterline.
const SHELF_WIDTH: f32 = 90.0;
const SEABED_DEPTH: f32 = 9.0;

/// Gently rolling ground that falls away into the sea beyond `shore_x`, and beyond a wandering
/// coast all the way round, so that the land is an island whose edges never show.
fn terrain_height(x: f32, z: f32, amplitude: f32, shore_x: f32) -> f32 {
    let p = Vec3::new(x, 0.0, z);
    let broad = fbm(p * 0.015, 4, 301) * amplitude;
    let ripples = fbm(p * 0.25, 4, 303) * 0.04;
    let shelf = |offshore: f32| {
        let t = (offshore / SHELF_WIDTH).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    // Bays and points, so that the waterline wanders rather than running ruler-straight.
    let wander = fbm(Vec3::new(0.0, 7.0, z) * 0.011, 4, 305) * 28.0;
    let cove = shelf(x - shore_x - wander);
    // The coast round the rest of the island, sampled around a circle so that it wanders
    // seamlessly: broad headlands and bays, with smaller points along them.
    let radius = x.hypot(z);
    let around = Vec3::new(x, 0.0, z) / radius.max(1.0);
    let headlands = fbm(around * 1.6 + Vec3::new(0.0, 3.0, 0.0), 3, 307) * 45.0;
    let points = fbm(around * 7.0 + Vec3::new(0.0, 11.0, 0.0), 3, 309) * 14.0;
    let island = shelf(radius - ISLAND_RADIUS - headlands - points);
    broad + ripples - cove.max(island) * SEABED_DEPTH
}

/// Adds the terrain as one mesh, with its own ground material and texture.
fn add_terrain(scene: &mut SceneAsset, size: f32, cells: u32, height: f32, shore_x: f32) -> i32 {
    let mesh = meshes::terrain(size, cells, |x, z| terrain_height(x, z, height, shore_x));
    let layer = scene.textures.len() as i32;
    scene
        .textures
        .push(surfaces::sand(0).bake(scene.texture_size));
    scene.materials.push(Material {
        // Darkened to about 0.35 albedo, as damp beach sand is, so that it sits correctly next to
        // scanned materials and does not skew auto exposure.
        base_color: [0.68, 0.68, 0.68, 1.0],
        roughness: 1.0,
        uv_scale: 1.0 / TILE_METRES,
        normal_strength: 1.0,
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
/// where the sand's ripples read as small waves, and is too dark to be seen as colour.
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

/// A scene's gameplay, gathered while its file is read.
#[derive(Default)]
struct Gameplay {
    game: SceneGame,
    /// Each actor file, script and input file, by name, once each.
    files: Vec<GameFile>,
    /// For each actor with a mesh: which actor, which model, and which of its whole placements.
    meshes: Vec<(usize, PathBuf, usize)>,
    /// Whether an actor has a camera already: only one can.
    has_player: bool,
}

impl Gameplay {
    /// Reads a gameplay file once, by its name relative to the scene.
    fn read(&mut self, base: &Path, name: &str) -> Result<&str, String> {
        if let Some(index) = self.files.iter().position(|f| f.name == name) {
            return Ok(&self.files[index].text);
        }
        let path = base.join(name);
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        self.files.push(GameFile {
            name: name.to_owned(),
            text,
        });
        Ok(&self.files.last().expect("just pushed").text)
    }

    /// Adds an actor, reading and checking its file, its settings and its script. Returns its
    /// mesh, if it has one: the model's path relative to the scene, and its scale.
    fn add_actor(
        &mut self,
        base: &Path,
        mut line: ActorLine,
    ) -> Result<Option<(String, f32)>, String> {
        if line.instances.is_some() {
            return Err("`instances` is worked out by the bake, not written by hand".to_owned());
        }
        line.file = resolve(GAME_FILE, &line.file);
        let actor = ActorFile::parse(self.read(base, &line.file)?)
            .map_err(|e| format!("{}: {e}", line.file))?;
        // Each of these would stop the game starting, so they stop the bake instead.
        if actor.script.is_none() && !line.settings.is_empty() {
            return Err(format!("{} has no script, so no properties", line.file));
        }
        if actor.camera.is_some() {
            if self.has_player {
                return Err(format!(
                    "{} has a camera, and another actor already does: only one can",
                    line.file
                ));
            }
            self.has_player = true;
        }
        for (name, value) in &line.settings {
            let property = actor
                .property(name)
                .ok_or_else(|| format!("{} has no property `{name}`", line.file))?;
            property
                .kind
                .parse_setting(value)
                .map_err(|e| format!("`{name}`: {e}"))?;
        }
        if let Some(script) = &actor.script {
            let name = resolve(&line.file, script);
            let source = self.read(base, &name)?.to_owned();
            // Checked now, so a mistake is found while baking, not when someone plays.
            solstice_script::ScriptHost::new()
                .load(&name, &source)
                .map_err(|e| e.message)?;
        }
        let mesh = actor
            .mesh
            .as_ref()
            .map(|mesh| (resolve(&line.file, &mesh.path), mesh.scale));
        self.game.actors.push(line);
        Ok(mesh)
    }

    fn set_input(&mut self, base: &Path, name: &str) -> Result<(), String> {
        if self.game.input.is_some() {
            return Err("`input` is given twice".to_owned());
        }
        let name = resolve(GAME_FILE, name);
        InputMap::parse(self.read(base, &name)?).map_err(|e| format!("{name}: {e}"))?;
        self.game.input = Some(name);
        Ok(())
    }

    /// The scene's gameplay files, with the list of its actors first, or none if it has none.
    fn finish(mut self) -> Vec<GameFile> {
        if self.game.actors.is_empty() && self.game.input.is_none() {
            return Vec::new();
        }
        self.files.insert(
            0,
            GameFile {
                name: GAME_FILE.to_owned(),
                text: self.game.to_text(),
            },
        );
        self.files
    }
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
    let mut gameplay = Gameplay::default();
    for (number, line) in text.lines().enumerate() {
        let at = |e: String| format!("{}:{}: {e}", path.display(), number + 1);
        let command = parse_line(line).map_err(at)?;
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
            Some(Command::Ocean { height, ocean }) => {
                // A file's text travels with the scene; a built-in ocean goes by its name.
                let ocean = if ocean.ends_with(".ocean") {
                    let file = base.join(&ocean);
                    std::fs::read_to_string(&file)
                        .map_err(|e| format!("{}: {e}", file.display()))?
                } else {
                    ocean
                };
                scene.sea = Some(Sea {
                    level: height,
                    ocean,
                });
            }
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
            Some(Command::Actor(actor)) => {
                let (position, yaw) = (actor.position, actor.yaw_degrees);
                let index = gameplay.game.actors.len();
                if let Some((model, scale)) = gameplay.add_actor(base, actor).map_err(at)? {
                    let full = base.join(&model);
                    let whole = &mut placements_of(&mut order, &mut models, base, model).whole;
                    gameplay.meshes.push((index, full, whole.len()));
                    whole.push(placement_transform(position, yaw, scale));
                }
            }
            Some(Command::Input(input)) => gameplay.set_input(base, &input).map_err(at)?,
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
        // Each whole placement adds the model's instances, in order, so an actor's are found by
        // which placement it was.
        let (first, per_placement) = (
            scene.instances.len() as u32,
            imported.scene.instances.len() as u32,
        );
        for (actor, _, placement) in gameplay.meshes.iter().filter(|(_, m, _)| m == model) {
            let start = first + *placement as u32 * per_placement;
            gameplay.game.actors[*actor].instances = Some(start..start + per_placement);
        }
        append_model(&mut scene, imported, &models[model])
            .map_err(|e| format!("{}: {e}", model.display()))?;
    }
    scene.gameplay = gameplay.finish();

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
        assert_eq!(
            parse_line("ocean -0.6 coastal").unwrap(),
            Some(Command::Ocean {
                height: -0.6,
                ocean: "coastal".to_string(),
            })
        );
        assert!(parse_line("ocean -0.6").is_err());
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
    fn the_terrain_edges_lie_under_the_sea_on_every_side() {
        // The cove's terrain is 900 m square, with rolling ground of amplitude 0.3.
        let water = -0.6;
        for i in 0..=200 {
            let along = -450.0 + 4.5 * i as f32;
            for (x, z) in [
                (-445.0, along),
                (along, 445.0),
                (along, -445.0),
                (445.0, along),
            ] {
                let height = terrain_height(x, z, 0.3, 52.0);
                assert!(
                    height < water - 4.0,
                    "({x}, {z}) is {height}, above the seabed"
                );
            }
        }
        // The fort and the ground around it are untouched.
        for (x, z) in [
            (0.0, 0.0),
            (-68.0, 0.0),
            (-30.0, 92.0),
            (-30.0, -92.0),
            (-150.0, 0.0),
        ] {
            assert!(
                terrain_height(x, z, 0.0, 52.0).abs() < 0.1,
                "({x}, {z}) moved"
            );
        }
    }

    /// A scratch folder holding the files given, removed when dropped.
    struct Folder(PathBuf);

    impl Folder {
        fn with(files: &[(&str, &[u8])]) -> Self {
            static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("solstice-bake-{}-{unique}", std::process::id()));
            for (name, bytes) in files {
                let path = dir.join(name);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, bytes).unwrap();
            }
            Self(dir)
        }
    }

    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A glTF of one triangle drawn twice, by two nodes, so that each placement is two instances.
    fn two_triangles() -> (Vec<u8>, Vec<u8>) {
        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        let json = r#"{
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 1]}],
            "nodes": [{"mesh": 0}, {"mesh": 0, "translation": [0, 2, 0]}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "buffers": [{"uri": "tri.bin", "byteLength": 36}],
            "bufferViews": [{"buffer": 0, "byteLength": 36}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                           "min": [0, 0, 0], "max": [1, 1, 0]}]
        }"#;
        (json.as_bytes().to_vec(), bin)
    }

    #[test]
    fn actors_their_meshes_and_scripts_are_baked_into_the_scene() {
        let (gltf, bin) = two_triangles();
        let folder = Folder::with(&[
            (
                "scene.txt",
                b"model models/tri.gltf 0 0 0 0 1\nactor fort/gate.actor 20 0 3 90 locked=true\nactor fort/gate.actor -5 1 0 0\ninput fort/play.input\n",
            ),
            ("models/tri.gltf", &gltf),
            ("models/tri.bin", &bin),
            (
                "fort/gate.actor",
                b"name gate\nmesh ../models/tri.gltf 2\nscript gate.star\nproperty locked bool false\n",
            ),
            ("fort/gate.star", b"def on_tick(self, dt):\n    self.yaw += dt\n"),
            ("fort/play.input", b"interact key KeyF\n"),
        ]);
        let (scene, _) = build(&folder.0.join("scene.txt"), None).unwrap();
        // The model line's two instances, then two for each gate.
        assert_eq!(scene.instances.len(), 6);
        let names: Vec<&str> = scene.gameplay.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            [
                GAME_FILE,
                "fort/gate.actor",
                "fort/gate.star",
                "fort/play.input"
            ]
        );
        let game = SceneGame::parse(&scene.gameplay[0].text).unwrap();
        assert_eq!(game.input.as_deref(), Some("fort/play.input"));
        assert_eq!(game.actors[0].instances, Some(2..4));
        assert_eq!(game.actors[1].instances, Some(4..6));
        let second_node = scene.instances[3].transform.transform_point3(Vec3::ZERO);
        let expected = placement_transform(Vec3::new(20.0, 0.0, 3.0), 90.0, 2.0)
            .transform_point3(Vec3::new(0.0, 2.0, 0.0));
        assert!(second_node.abs_diff_eq(expected, 1.0e-3), "{second_node}");

        // And it plays: drawn first where it was baked.
        let instances = scene.instances.clone();
        let mut play = solstice_game::Play::load(&scene.gameplay, |id| {
            instances.get(id as usize).map(|i| i.transform)
        })
        .unwrap();
        let mut placed = Vec::new();
        // One step in, before the script has turned it far.
        play.frame(
            1.0 / 60.0 * 1.001,
            &mut solstice_game::Devices::new(),
            |id, at| placed.push((id, at)),
        );
        assert_eq!(placed.len(), 4);
        for (id, at) in placed {
            assert!(
                at.abs_diff_eq(scene.instances[id as usize].transform, 0.1),
                "{id}"
            );
        }
    }

    #[test]
    fn a_scene_without_actors_has_no_gameplay() {
        let (gltf, bin) = two_triangles();
        let folder = Folder::with(&[
            ("scene.txt", b"model tri.gltf 0 0 0 0 1\n"),
            ("tri.gltf", &gltf),
            ("tri.bin", &bin),
        ]);
        let (scene, _) = build(&folder.0.join("scene.txt"), None).unwrap();
        assert!(scene.gameplay.is_empty());
    }

    #[test]
    fn gameplay_mistakes_are_found_while_baking() {
        let cases: [(&[u8], &str); 8] = [
            (
                b"actor gate.actor 0 0 0 0 colour=red\n",
                "has no property `colour`",
            ),
            (b"actor gate.actor 0 0 0 0 locked=maybe\n", "true or false"),
            (b"actor nowhere.actor 0 0 0 0\n", "nowhere.actor"),
            (b"actor broken.actor 0 0 0 0\n", "broken.star:2"),
            (b"input bad.input\n", "bad.input: line 1"),
            (
                b"actor rock.actor 0 0 0 0 size=2\n",
                "has no script, so no properties",
            ),
            (
                b"actor eye.actor 0 0 0 0\nactor eye.actor 1 0 0 0\n",
                "only one can",
            ),
            (
                b"actor gate.actor 0 0 0 0 instances=0+5\n",
                "worked out by the bake",
            ),
        ];
        for (scene, part) in cases {
            let folder = Folder::with(&[
                ("scene.txt", scene),
                (
                    "gate.actor",
                    b"name gate\nscript gate.star\nproperty locked bool false\n",
                ),
                ("gate.star", b"def on_begin(self):\n    pass\n"),
                ("rock.actor", b"name rock\n"),
                ("eye.actor", b"name eye\ncamera 0 1 0\n"),
                ("broken.actor", b"name broken\nscript broken.star\n"),
                ("broken.star", b"def on_tick(self, dt):\n    x = = 1\n"),
                ("bad.input", b"jump\n"),
            ]);
            let error = build(&folder.0.join("scene.txt"), None).unwrap_err();
            assert!(error.contains(part), "{error}");
            assert!(error.contains("scene.txt:"), "{error}");
        }
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
