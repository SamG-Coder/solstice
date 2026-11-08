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
//!
//! # Many at once, laid out at random over a rectangle of ground, from x0 z0 to x1 z1:
//! hills 40 300 120                      # relief: height, wavelength, and the level radius
//! scatter rocks/rock.gltf 500 -300 -300 300 300 ground:0.2 0.5 2 1  # count, rect, height, scales, seed
//! trees oak 2000 -400 -400 400 400 6 1  # species, count, rect, spacing, seed
//! grass meadow -100 -100 100 100        # breed, rect
//! lights 64 -40 -40 40 40 3 12 40 14 3 1  # count, rect, height, range, r g b, seed
//! ```
//!
//! A model placed several times is imported once and shared, so repeats cost almost nothing.

use crate::flora::{self, Ground, Layers, Rect, Stand};
use crate::meshes;
use glam::{Mat4, Quat, Vec3};
use solstice_asset::scene::Viewpoint;
use solstice_asset::{Instance, Light, Material, SceneAsset, Sea};
use solstice_builder::{build_virtual_mesh, BuildOptions};
use solstice_import::{append_model, import_gltf, ImportOptions, ModelPlacements};
use solstice_procgen::foliage::grass::Meadow;
use solstice_procgen::foliage::tree::Species;
use solstice_procgen::foliage::Rng;
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
    Hills(Hills),
    /// Copies of a model thrown at random over a rectangle of ground.
    Scatter {
        path: String,
        count: u32,
        area: Rect,
        /// A fixed height, or `Some(buried)` to stand each on the terrain.
        height: f32,
        seat: Option<f32>,
        scale: (f32, f32),
        seed: u32,
    },
    Trees {
        species: String,
        count: u32,
        area: Rect,
        spacing: f32,
        seed: u32,
    },
    Grass {
        breed: String,
        area: Rect,
    },
    Lights {
        count: u32,
        area: Rect,
        height: f32,
        range: f32,
        intensity: [f32; 3],
        seed: u32,
    },
}

/// Broad relief added to the terrain: hills of `amplitude` over about `wavelength` metres,
/// level within `flat_radius` of the origin so that whatever stands there keeps its footing, and
/// fading out towards the coast so that the shore still meets the sea.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Hills {
    amplitude: f32,
    wavelength: f32,
    flat_radius: f32,
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
        "hills" => {
            expect(3)?;
            Command::Hills(Hills {
                amplitude: number(0)?,
                wavelength: number(1)?,
                flat_radius: number(2)?,
            })
        }
        "scatter" => {
            expect(10)?;
            let (height, seat) = parse_height(values[6])?;
            Command::Scatter {
                path: values[0].to_string(),
                count: number(1)? as u32,
                area: Rect::new(number(2)?, number(3)?, number(4)?, number(5)?),
                height,
                seat,
                scale: (number(7)?, number(8)?),
                seed: number(9)? as u32,
            }
        }
        "trees" => {
            expect(8)?;
            Command::Trees {
                species: values[0].to_string(),
                count: number(1)? as u32,
                area: Rect::new(number(2)?, number(3)?, number(4)?, number(5)?),
                spacing: number(6)?,
                seed: number(7)? as u32,
            }
        }
        "grass" => {
            expect(5)?;
            Command::Grass {
                breed: values[0].to_string(),
                area: Rect::new(number(1)?, number(2)?, number(3)?, number(4)?),
            }
        }
        "lights" => {
            expect(11)?;
            Command::Lights {
                count: number(0)? as u32,
                area: Rect::new(number(1)?, number(2)?, number(3)?, number(4)?),
                height: number(5)?,
                range: number(6)?,
                intensity: [number(7)?, number(8)?, number(9)?],
                seed: number(10)? as u32,
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

/// How far inland the coast around the rest of the island lies from the origin, before its bays
/// and points, as a fraction of the terrain's width. With the slope down to the seabed, this
/// keeps the land well inside the terrain square, so that its edges lie under the sea rather
/// than showing as straight lines.
const ISLAND_FRACTION: f32 = 0.3;
/// How far the seabed lies beyond the waterline.
const SHELF_WIDTH: f32 = 90.0;
const SEABED_DEPTH: f32 = 9.0;

/// The ground of a scene: a square of gently rolling land that falls away into the sea beyond
/// `shore_x`, and beyond a wandering coast all the way round, so that the land is an island
/// whose edges never show.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Terrain {
    size: f32,
    cells: u32,
    /// Height of the rolling ground's bumps.
    amplitude: f32,
    shore_x: f32,
    hills: Option<Hills>,
}

impl Terrain {
    /// How far the shelf has fallen towards the seabed, 0 on land to 1 at full depth.
    fn shelf(offshore: f32) -> f32 {
        let t = (offshore / SHELF_WIDTH).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// How far out to sea a point is, 0 on land to 1 over the seabed.
    fn offshore(&self, x: f32, z: f32) -> f32 {
        // Bays and points, so that the waterline wanders rather than running ruler-straight.
        let wander = fbm(Vec3::new(0.0, 7.0, z) * 0.011, 4, 305) * 28.0;
        let cove = Self::shelf(x - self.shore_x - wander);
        // The coast round the rest of the island, sampled around a circle so that it wanders
        // seamlessly: broad headlands and bays, with smaller points along them.
        let radius = x.hypot(z);
        let around = Vec3::new(x, 0.0, z) / radius.max(1.0);
        let headlands = fbm(around * 1.6 + Vec3::new(0.0, 3.0, 0.0), 3, 307) * 45.0;
        let points = fbm(around * 7.0 + Vec3::new(0.0, 11.0, 0.0), 3, 309) * 14.0;
        let island = Self::shelf(radius - self.size * ISLAND_FRACTION - headlands - points);
        cove.max(island)
    }

    fn height(&self, x: f32, z: f32) -> f32 {
        let p = Vec3::new(x, 0.0, z);
        let broad = fbm(p * 0.015, 4, 301) * self.amplitude;
        let ripples = fbm(p * 0.25, 4, 303) * 0.04;
        let offshore = self.offshore(x, z);
        let hills = self.hills.map_or(0.0, |hills| {
            // Always above the level ground, so no valley sinks below the sea inland, and
            // rising smoothly from the level circle over a distance of half its radius.
            let relief = (fbm(p / hills.wavelength, 5, 311) * 0.5 + 0.5) * hills.amplitude;
            let from_centre =
                ((x.hypot(z) - hills.flat_radius) / (hills.flat_radius * 0.5)).clamp(0.0, 1.0);
            let rise = from_centre * from_centre * (3.0 - 2.0 * from_centre);
            relief * rise * (1.0 - offshore)
        });
        broad + ripples + hills - offshore * SEABED_DEPTH
    }
}

/// Adds the terrain as one mesh, with its own ground material and texture.
fn add_terrain(scene: &mut SceneAsset, terrain: &Terrain) -> i32 {
    let mesh = meshes::terrain(terrain.size, terrain.cells, |x, z| terrain.height(x, z));
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

/// Builds the scene described by the file at `path`. Returns it with any import warnings.
pub fn build(path: &Path, texture_size: Option<u32>) -> Result<(SceneAsset, Vec<String>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));

    let mut scene = SceneAsset {
        texture_size: 1024,
        ..Default::default()
    };
    let mut terrain: Option<Terrain> = None;
    let mut hills = None;
    let mut water = None;
    // Grouped by file, keeping first-seen order, so that each model is imported only once.
    let mut order: Vec<PathBuf> = Vec::new();
    let mut models: HashMap<PathBuf, ModelPlacements> = HashMap::new();
    // Placed once the model has been imported, since its shape decides the height.
    let mut seated: HashMap<PathBuf, Vec<(Vec3, f32, f32, f32)>> = HashMap::new();
    // Grown once the terrain is known, whichever line it is on.
    let mut flora = Vec::new();
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
                terrain = Some(Terrain {
                    size,
                    cells,
                    amplitude: height,
                    shore_x,
                    hills: None,
                });
            }
            Some(Command::Hills(relief)) => hills = Some(relief),
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
            Some(Command::Scatter {
                path: model,
                count,
                area,
                height,
                seat,
                scale,
                seed,
            }) => {
                let full = base.join(&model);
                let placements = placements_of(&mut order, &mut models, base, model);
                let mut rng = Rng::new(seed, 17);
                for _ in 0..count {
                    let (x, z) = area.sample(&mut rng);
                    let yaw = rng.range(0.0, 360.0);
                    let size = rng.range(scale.0, scale.1);
                    let position = Vec3::new(x, height, z);
                    match seat {
                        Some(buried) => seated
                            .entry(full.clone())
                            .or_default()
                            .push((position, yaw, size, buried)),
                        None => placements
                            .whole
                            .push(placement_transform(position, yaw, size)),
                    }
                }
            }
            Some(command @ (Command::Trees { .. } | Command::Grass { .. })) => {
                flora.push(command);
            }
            Some(Command::Lights {
                count,
                area,
                height,
                range,
                intensity,
                seed,
            }) => {
                // Lit once the terrain is known, so each hangs at `height` above the ground.
                flora.push(Command::Lights {
                    count,
                    area,
                    height,
                    range,
                    intensity,
                    seed,
                });
            }
        }
    }
    if let Some(hills) = hills {
        let Some(terrain) = &mut terrain else {
            return Err("'hills' needs a terrain line".to_string());
        };
        terrain.hills = Some(hills);
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
            let Some(terrain) = terrain else {
                return Err(format!("{name}: 'ground' needs a terrain line"));
            };
            // Water hides whatever is below it, so the sea floor does not count.
            let water_level = water.map_or(f32::MIN, |(level, _)| level);
            let ground = |x: f32, z: f32| terrain.height(x, z).max(water_level);
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

    if let Some(terrain) = terrain {
        let ground_layer = add_terrain(&mut scene, &terrain);
        if let Some((level, color)) = water {
            add_water(&mut scene, level, color, ground_layer);
        }
    }
    if !flora.is_empty() {
        let Some(terrain) = terrain else {
            return Err("'trees', 'grass' and 'lights' need a terrain line".to_string());
        };
        let sea_level = water
            .map(|(level, _)| level)
            .or(scene.sea.as_ref().map(|sea| sea.level))
            .unwrap_or(f32::MIN);
        let height = |x: f32, z: f32| terrain.height(x, z);
        let ground = Ground {
            height: &height,
            sea_level,
        };
        let mut layers = Layers::default();
        for command in flora {
            match command {
                Command::Trees {
                    species,
                    count,
                    area,
                    spacing,
                    seed,
                } => {
                    let grown = Species::by_name(&species).ok_or_else(|| {
                        let names: Vec<_> = Species::names().collect();
                        format!("no species '{species}': there are {}", names.join(", "))
                    })?;
                    println!("growing {species}...");
                    let stand = Stand {
                        count,
                        area,
                        spacing,
                        seed,
                    };
                    let placed =
                        flora::add_trees(&mut scene, &mut layers, &grown, &species, stand, ground);
                    println!("  {placed} of {count} {species} placed");
                }
                Command::Grass { breed, area } => {
                    let meadow = Meadow::by_name(&breed).ok_or_else(|| {
                        let names: Vec<_> = Meadow::names().collect();
                        format!("no grass '{breed}': there are {}", names.join(", "))
                    })?;
                    println!("growing {breed} grass...");
                    let placed =
                        flora::add_grass(&mut scene, &mut layers, &meadow, &breed, area, ground);
                    println!("  {placed} patches");
                }
                Command::Lights {
                    count,
                    area,
                    height,
                    range,
                    intensity,
                    seed,
                } => {
                    let mut rng = Rng::new(seed, 19);
                    for _ in 0..count {
                        let (x, z) = area.sample(&mut rng);
                        scene.lights.push(Light {
                            position: [x, terrain.height(x, z).max(sea_level) + height, z],
                            range,
                            intensity,
                            _pad: 0,
                        });
                    }
                }
                _ => unreachable!("only flora and lights wait for the terrain"),
            }
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

    /// The cove's terrain: 900 m square, with rolling ground of amplitude 0.3.
    fn cove(amplitude: f32) -> Terrain {
        Terrain {
            size: 900.0,
            cells: 16,
            amplitude,
            shore_x: 52.0,
            hills: None,
        }
    }

    #[test]
    fn the_seabed_falls_below_the_shore_only_on_the_sea_side() {
        let terrain = Terrain {
            shore_x: 40.0,
            ..cove(0.0)
        };
        assert!(terrain.height(-50.0, 0.0).abs() < 0.1, "inland stays level");
        assert!(
            terrain.height(10.0, 0.0).abs() < 0.1,
            "level behind the shoreline, wherever it wanders"
        );
        assert!(terrain.height(130.0, 0.0) < -8.0, "deep offshore");
    }

    #[test]
    fn the_terrain_edges_lie_under_the_sea_on_every_side() {
        let water = -0.6;
        let terrain = cove(0.3);
        for i in 0..=200 {
            let along = -450.0 + 4.5 * i as f32;
            for (x, z) in [
                (-445.0, along),
                (along, 445.0),
                (along, -445.0),
                (445.0, along),
            ] {
                let height = terrain.height(x, z);
                assert!(
                    height < water - 4.0,
                    "({x}, {z}) is {height}, above the seabed"
                );
            }
        }
        // The fort and the ground around it are untouched.
        let terrain = cove(0.0);
        for (x, z) in [
            (0.0, 0.0),
            (-68.0, 0.0),
            (-30.0, 92.0),
            (-30.0, -92.0),
            (-150.0, 0.0),
        ] {
            assert!(terrain.height(x, z).abs() < 0.1, "({x}, {z}) moved");
        }
    }

    #[test]
    fn a_wider_terrain_has_a_wider_island() {
        // Twice the cove's width: land at the cove's island radius, on the side away from the
        // shore, where the cove is already under the sea.
        let wide = Terrain {
            size: 1800.0,
            ..cove(0.0)
        };
        assert!(wide.height(-300.0, 0.0).abs() < 0.1);
        assert!(cove(0.0).height(-300.0, 0.0) < -1.0);
        // And its edges are under the sea all the same.
        for i in 0..=100 {
            let along = -890.0 + 17.8 * i as f32;
            assert!(wide.height(-890.0, along) < -5.0);
            assert!(wide.height(along, 890.0) < -5.0);
        }
    }

    #[test]
    fn hills_rise_inland_but_leave_the_centre_and_the_coast_level() {
        let terrain = Terrain {
            size: 1800.0,
            hills: Some(Hills {
                amplitude: 60.0,
                wavelength: 200.0,
                flat_radius: 120.0,
            }),
            ..cove(0.0)
        };
        let flat = cove(0.0);
        for (x, z) in [(0.0, 0.0), (80.0, 60.0), (-110.0, 0.0)] {
            assert!(
                (terrain.height(x, z) - flat.height(x, z)).abs() < 1.0e-4,
                "({x}, {z}) is not level"
            );
        }
        // Somewhere inland is high ground, and nowhere is lower than the level ground.
        let mut highest: f32 = 0.0;
        for i in 0..40 {
            for j in 0..40 {
                let (x, z) = (-500.0 + i as f32 * 25.0, -500.0 + j as f32 * 25.0);
                let rise = terrain.height(x, z) - flat.height(x, z);
                assert!(rise >= -1.0e-4, "({x}, {z}) sank by {rise}");
                highest = highest.max(rise);
            }
        }
        assert!(highest > 20.0, "the highest hill rises only {highest}");
        // The hills have gone by the shore, so the sea still meets the land.
        assert!(terrain.height(60.0, 0.0) - flat.height(60.0, 0.0) < 0.5);
    }

    #[test]
    fn the_new_commands_parse() {
        assert_eq!(
            parse_line("hills 40 300 120").unwrap(),
            Some(Command::Hills(Hills {
                amplitude: 40.0,
                wavelength: 300.0,
                flat_radius: 120.0,
            }))
        );
        assert_eq!(
            parse_line("scatter r.gltf 50 -10 -20 10 20 ground:0.2 0.5 2 7").unwrap(),
            Some(Command::Scatter {
                path: "r.gltf".to_string(),
                count: 50,
                area: Rect::new(-10.0, -20.0, 10.0, 20.0),
                height: 0.0,
                seat: Some(0.2),
                scale: (0.5, 2.0),
                seed: 7,
            })
        );
        assert_eq!(
            parse_line("trees oak 2000 -400 -400 400 400 6 1").unwrap(),
            Some(Command::Trees {
                species: "oak".to_string(),
                count: 2000,
                area: Rect::new(-400.0, -400.0, 400.0, 400.0),
                spacing: 6.0,
                seed: 1,
            })
        );
        assert_eq!(
            parse_line("grass meadow -100 -100 100 100").unwrap(),
            Some(Command::Grass {
                breed: "meadow".to_string(),
                area: Rect::new(-100.0, -100.0, 100.0, 100.0),
            })
        );
        assert_eq!(
            parse_line("lights 64 -40 -40 40 40 3 12 40 14 3 1").unwrap(),
            Some(Command::Lights {
                count: 64,
                area: Rect::new(-40.0, -40.0, 40.0, 40.0),
                height: 3.0,
                range: 12.0,
                intensity: [40.0, 14.0, 3.0],
                seed: 1,
            })
        );
        assert!(
            parse_line("scatter r.gltf 50 -10 -20 10 20 ground:0.2 0.5 2")
                .unwrap_err()
                .contains("takes 10 values")
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
