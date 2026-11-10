//! Play mode: a scene's game, running.
//!
//! [`Play::load`] builds a world from a baked scene's gameplay files: an entity for each placed
//! actor, with its name, its place, and its script with the placement's property values. Each
//! frame, [`Play::frame`] runs the game, then says where to draw each actor's instances, between
//! the last two steps, and where the player's camera is.

use crate::actor::ActorFile;
use crate::animation::{AnimatedActors, AnimationAssets};
use crate::drive::{self, Chase, DrawnWheel, Hud, Wheels, HUB_REACH, SPIN_REACH};
use crate::input::{Devices, InputMap};
use crate::race::{format_time, Lap, LapEvent};
use crate::resolve;
use crate::scene::{SceneGame, GAME_FILE};
use crate::track::Track;
use glam::{EulerRot, Mat4, Quat, Vec3};
use solstice_asset::NO_POSE;
use solstice_asset::{Collision, GameFile};
use solstice_physics::vehicle::{Surface, Vehicle, VehicleSpec};
use solstice_physics::{Car, Overlap, Physics, GRAVITY};
use solstice_script::{Script, ScriptEvent, ScriptHost};
use solstice_world::{
    Entity, Game, GlobalTransform, Name, Played, Schedule, TickGroup, Transform, World,
    STEP_SECONDS,
};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

/// How many messages are kept for the viewer to show, newest last.
pub const MAX_MESSAGES: usize = 200;
/// How far the player's camera may look up or down, in radians: not quite straight.
pub const MAX_PITCH: f32 = 1.5;

/// A line for the viewer to show: something a script logged, or an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub error: bool,
    pub text: String,
}

/// Where the player's camera is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub position: Vec3,
    /// Around +Y, in radians; zero looks along -Z, as the renderer's camera does.
    pub yaw: f32,
    pub pitch: f32,
}

/// A placed actor with instances to move.
struct Drawn {
    entity: Entity,
    /// Its `.anim` file, if it is animated, by its path in the scene's gameplay.
    animation: Option<String>,
    /// Each instance, and where it is relative to the actor: or, for a car's wheel, relative
    /// to the wheel's own middle.
    instances: Vec<(u32, Mat4)>,
    /// For a car, which wheel each instance is carried by, if any, and whether it turns with
    /// it: a brake caliper is carried, and doesn't.
    wheels: Vec<Option<(usize, bool)>>,
    /// Where the actor was last drawn, to move its instances only when it moves.
    drawn: Option<Mat4>,
    /// Where a car's wheels were last drawn, relative to it: rolled, and as their uprights.
    rolled: Option<[(Mat4, Mat4); 4]>,
}

/// The actor the player plays, and where its eye is.
struct Player {
    entity: Entity,
    eye: Vec3,
    /// Whether it is a car, followed by the chase camera.
    car: bool,
}

const WHEEL_NAMES: [&str; 4] = ["front left", "front right", "rear left", "rear right"];

/// A scene's game, running.
pub struct Play {
    game: Game,
    host: Rc<RefCell<ScriptHost>>,
    drawn: Vec<Drawn>,
    player: Option<Player>,
    map: InputMap,
    /// How far the camera looks up, as of the last step.
    pitch: f32,
    /// Seconds since the last step, over which the devices' movement has built up.
    unstepped: f64,
    /// Actions to press in the next frame, from [`Play::press`].
    presses: Vec<String>,
    scripts: Vec<String>,
    messages: VecDeque<Message>,
    physics: Rc<RefCell<Physics>>,
    /// Everything animated, once [`Play::attach_animation`] has been called.
    animated: Rc<RefCell<AnimatedActors>>,
    /// Bones to pose, for the viewer to hand to the renderer.
    poses: Vec<(u32, Vec<Mat4>)>,
    viewer: Vec3,
    /// Whether the player has left whoever they play, to fly the camera.
    flying: bool,
    /// The circuit, if the scene is one.
    track: Option<Rc<Track>>,
    /// The camera that follows the player's car.
    chase: Chase,
    /// The player's car's instruments, as of the latest frame.
    hud: Option<Hud>,
    /// What the timing of laps has to say, for the viewer to show.
    race_log: Rc<RefCell<Vec<String>>>,
}

impl Play {
    /// Builds the game in a baked scene's gameplay `files`. `instance` gives where the bake
    /// placed a renderer instance. Every mistake found is reported, each with its file and line.
    ///
    /// `collision` is what the scene's static world is made of, if it has been fetched: without
    /// it, characters have nothing to stand on.
    pub fn load(
        files: &[GameFile],
        instance: impl Fn(u32) -> Option<Mat4>,
        collision: Option<&Collision>,
    ) -> Result<Self, Vec<String>> {
        let file = |name: &str| files.iter().find(|f| f.name == name).map(|f| &f.text);
        let mut errors = Vec::new();
        let scene = file(GAME_FILE)
            .ok_or_else(|| vec![format!("the scene's gameplay has no {GAME_FILE}")])?;
        let scene = SceneGame::parse(scene).map_err(|e| vec![format!("{GAME_FILE}: {e}")])?;
        // Whether any actor is a car: only then are the built-in driving actions read.
        let cars = scene.actors.iter().any(|line| {
            file(&line.file)
                .and_then(|text| ActorFile::parse(text).ok())
                .is_some_and(|actor| actor.vehicle.is_some())
        });
        let map = match &scene.input {
            None => InputMap::built_in(cars),
            Some(name) => match file(name).map(|text| InputMap::parse(text)) {
                Some(Ok(map)) => map,
                Some(Err(e)) => {
                    errors.push(format!("{name}: {e}"));
                    InputMap::built_in(cars)
                }
                None => {
                    errors.push(format!("{name}: missing from the scene's gameplay"));
                    InputMap::built_in(cars)
                }
            },
        };

        let track = match &scene.track {
            None => None,
            Some(name) => match file(name).map(|text| Track::parse(text)) {
                Some(Ok(track)) => Some(Rc::new(track)),
                Some(Err(e)) => {
                    errors.push(format!("{name}: {e}"));
                    None
                }
                None => {
                    errors.push(format!("{name}: missing from the scene's gameplay"));
                    None
                }
            },
        };

        let host = Rc::new(RefCell::new(ScriptHost::new()));
        let physics = Rc::new(RefCell::new(Physics::new()));
        if let Some(track) = &track {
            // What each tyre is on. A car's wheels are near each other, and near where they
            // were, so the search starts from the last place found.
            let (track, near) = (track.clone(), Cell::new(None));
            physics.borrow_mut().set_surfaces(move |point| {
                let place = track.locate(point, near.get());
                near.set(Some(place.sample));
                place.surface
            });
        }
        if let Some(collision) = collision {
            physics.borrow_mut().add_scene(collision, &instance);
        }
        for blocker in &scene.blockers {
            physics.borrow_mut().add_blocker(
                blocker.centre,
                blocker.size,
                blocker.yaw_degrees.to_radians(),
            );
        }
        let mut world = World::new();
        let mut drawn = Vec::new();
        let mut player = None;
        let mut scripts: Vec<String> = Vec::new();
        for (index, line) in scene.actors.iter().enumerate() {
            let at = format!("{GAME_FILE}: actor {} ({})", index + 1, line.file);
            let Some(text) = file(&line.file) else {
                errors.push(format!("{at}: missing from the scene's gameplay"));
                continue;
            };
            let actor = match ActorFile::parse(text) {
                Ok(actor) => actor,
                Err(e) => {
                    errors.push(format!("{}: {e}", line.file));
                    continue;
                }
            };
            let rotation = Quat::from_rotation_y(line.yaw_degrees.to_radians());
            let mut components = hecs_builder();
            components
                .add(Name::new(&actor.name))
                .add(Transform::from_translation(line.position).with_rotation(rotation));
            if let Some(relative) = &actor.script {
                let name = resolve(&line.file, relative);
                let mut script = Script::new(&name);
                for property in &actor.properties {
                    script
                        .properties
                        .insert(property.name.clone(), property.default.clone());
                }
                for (setting, value) in &line.settings {
                    match actor.property(setting) {
                        None => {
                            errors.push(format!("{at}: {} has no property `{setting}`", line.file))
                        }
                        Some(property) => match property.kind.parse_setting(value) {
                            Ok(value) => {
                                script.properties.insert(setting.clone(), value);
                            }
                            Err(e) => errors.push(format!("{at}: `{setting}`: {e}")),
                        },
                    }
                }
                if !scripts.contains(&name) {
                    match file(&name) {
                        None => errors.push(format!(
                            "{}: its script {name} is missing from the scene's gameplay",
                            line.file
                        )),
                        Some(source) => {
                            if let Err(e) = host.borrow_mut().load(&name, source) {
                                errors.push(e.message);
                            }
                        }
                    }
                    scripts.push(name);
                }
                components.add(script);
            } else if !line.settings.is_empty() {
                errors.push(format!(
                    "{at}: {} has no script, so no properties",
                    line.file
                ));
            }
            if let Some(collider) = actor.collider {
                components.add(collider);
            }
            if let Some(character) = actor.character {
                components.add(character);
            }
            if actor.camera.is_some() && player.is_none() {
                components.add(Played);
            }
            // A car: its numbers, its wheels as they are drawn, and its laps.
            let mut hubs = None;
            if let Some(vehicle) = &actor.vehicle {
                let name = resolve(&line.file, &vehicle.path);
                match file(&name).map(|text| VehicleSpec::parse(text)) {
                    Some(Ok(spec)) => {
                        let car = Car::new(Vehicle::new(spec, GRAVITY), vehicle.automatic);
                        hubs = Some(car.vehicle.spec().hubs());
                        components.add(Wheels::at_rest(&car.vehicle)).add(car);
                        if let Some(track) = &track {
                            components.add(Lap::new(track));
                        }
                    }
                    Some(Err(e)) => errors.push(format!("{name}: {e}")),
                    None => errors.push(format!(
                        "{}: its vehicle {name} is missing from the scene's gameplay",
                        line.file
                    )),
                }
            }
            let entity = world.spawn(components.build());
            if let Some(eye) = actor.camera {
                if player.is_some() {
                    errors.push(format!(
                        "{at}: only one actor can have a camera, and one already does"
                    ));
                }
                player = Some(Player {
                    entity,
                    eye,
                    car: hubs.is_some(),
                });
            }
            if let Some(range) = &line.instances {
                let placed = Mat4::from_rotation_translation(rotation, line.position);
                let mut instances = Vec::new();
                let mut wheels = Vec::new();
                for id in range.clone() {
                    let Some(world) = instance(id) else {
                        errors.push(format!("{at}: the scene has no instance {id}"));
                        break;
                    };
                    let local = placed.inverse() * world;
                    // A piece of a car's mesh with its own origin at a hub is that wheel.
                    let origin = local.w_axis.truncate();
                    let wheel = hubs.and_then(|hubs: [Vec3; 4]| {
                        hubs.iter()
                            .position(|hub| hub.distance(origin) < HUB_REACH)
                            .map(|index| (index, hubs[index]))
                    });
                    match wheel {
                        Some((index, hub)) => {
                            let turns = hub.distance(origin) < SPIN_REACH;
                            instances.push((id, Mat4::from_translation(-hub) * local));
                            wheels.push(Some((index, turns)));
                        }
                        None => {
                            instances.push((id, local));
                            wheels.push(None);
                        }
                    }
                }
                if let Some(hubs) = hubs {
                    for (index, hub) in hubs.iter().enumerate() {
                        if !wheels.contains(&Some((index, true))) {
                            errors.push(format!(
                                "{at}: its mesh has no piece with its origin at the {} hub, \
                                 {} {} {}: a car's wheels are pieces of its mesh, each with its \
                                 own origin at its middle",
                                WHEEL_NAMES[index], hub.x, hub.y, hub.z
                            ));
                        }
                    }
                }
                drawn.push(Drawn {
                    entity,
                    animation: actor.animation.as_ref().map(|a| resolve(&line.file, a)),
                    instances,
                    wheels,
                    drawn: None,
                    rolled: None,
                });
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }

        let mut schedule = Schedule::new();
        ScriptHost::install(&host, &mut schedule);
        // The player's car takes what the player asks of it, after scripts have had their say.
        let driven = player.as_ref().filter(|p| p.car).map(|p| p.entity);
        let circuit = track.clone();
        // Whether the car is driving itself round the circuit, and where on it it last was.
        let (mut itself, mut last_at) = (false, None);
        schedule.add(TickGroup::PrePhysics, "driving", move |world| {
            let Some(entity) = driven else {
                return;
            };
            // While the player is away flying the camera, the keys that fly it are not the
            // car's: it is asked for nothing, as a walker left standing hears nothing.
            let away = world.input().away();
            let mut asked = if away {
                Default::default()
            } else {
                drive::asked(world.input())
            };
            let reset = !away && world.input().pressed("reset");
            if !away && world.input().pressed("autopilot") {
                itself = !itself;
                last_at = None;
            }
            if let (true, Some(track)) = (itself, &circuit) {
                let car = world
                    .entities
                    .query_one_mut::<(&Transform, &Car)>(entity)
                    .ok();
                if let Some((transform, car)) = car {
                    let (driven, at) = drive::autopilot(
                        track,
                        transform.translation,
                        transform.rotation * Vec3::NEG_Z,
                        car.vehicle.telemetry().speed,
                        last_at,
                    );
                    (asked, last_at) = (driven, Some(at));
                }
            }
            if let Ok(mut car) = world.entities.get::<&mut Car>(entity) {
                car.asked = asked;
            }
            // Put back on the track when the player asks, and when the car has fallen far
            // below it: off a cliff, or into the sea.
            let fallen = circuit.as_ref().is_some_and(|track| {
                world.entities.get::<&Transform>(entity).is_ok_and(|at| {
                    let place = track.locate(at.translation, last_at);
                    // Kept as the next step's hint, or every step searches the whole lap.
                    last_at = Some(place.sample);
                    at.translation.y < track.samples[place.sample].position.y - drive::FALLEN
                })
            });
            if let (true, Some(track)) = (reset || fallen, &circuit) {
                drive::put_back(world, entity, track);
                last_at = None;
            }
        });
        Physics::install(&physics, &mut schedule);
        // Each car's wheels as they are after the step, and its laps.
        let race_log: Rc<RefCell<Vec<String>>> = Rc::default();
        let (circuit, log) = (track.clone(), race_log.clone());
        schedule.add(TickGroup::PostPhysics, "racing", move |world| {
            let mut log = log.borrow_mut();
            for (car, wheels, transform, lap) in world
                .entities
                .query::<(&Car, &mut Wheels, &Transform, Option<&mut Lap>)>()
                .iter()
            {
                wheels.follow(&car.vehicle);
                let (Some(lap), Some(track)) = (lap, &circuit) else {
                    continue;
                };
                // Off the track: every wheel that is on the ground is on neither asphalt nor
                // kerb.
                let mut grounded = car.vehicle.wheels.iter().filter(|wheel| wheel.on_ground);
                let off_track = grounded.clone().next().is_some()
                    && grounded
                        .all(|wheel| !matches!(wheel.surface, Surface::Asphalt | Surface::Kerb));
                for event in lap.step(track, transform.translation, off_track, STEP_SECONDS) {
                    match event {
                        LapEvent::Started => log.push("lap 1 started".to_owned()),
                        LapEvent::Sector { .. } => {}
                        LapEvent::Finished {
                            lap,
                            seconds,
                            valid,
                            best,
                        } => {
                            let note = if !valid {
                                "  (off track: doesn't count)".to_owned()
                            } else {
                                match best {
                                    Some(best) if seconds < best => {
                                        format!("  best, by {:.3}", best - seconds)
                                    }
                                    Some(best) => format!("  +{:.3}", seconds - best),
                                    None => String::new(),
                                }
                            };
                            log.push(format!("lap {lap}: {}{note}", format_time(seconds)));
                        }
                    }
                }
            }
        });
        let animated: Rc<RefCell<AnimatedActors>> = Default::default();
        let running = animated.clone();
        schedule.add(TickGroup::PostPhysics, "animation", move |world| {
            let commands = world.read_cloned::<solstice_script::AnimCommand>();
            let mut running = running.borrow_mut();
            running.apply(&commands);
            let mut notifies = Vec::new();
            running.step(world, &mut notifies);
            for (entity, name) in notifies {
                world.send(ScriptEvent::to(entity, "anim").arg(name.as_str()));
            }
        });
        // What enters and leaves a sensor reaches its script as `on_enter` and `on_exit`.
        schedule.add(TickGroup::PostPhysics, "touches", |world| {
            for overlap in world.read_cloned::<Overlap>() {
                let name = if overlap.entered { "enter" } else { "exit" };
                world.send(ScriptEvent::to(overlap.sensor, name).arg(overlap.other));
            }
        });
        let mut messages = VecDeque::new();
        for warning in physics.borrow_mut().warnings.drain(..) {
            messages.push_back(Message {
                error: true,
                text: warning,
            });
        }
        Ok(Self {
            game: Game::new(world, schedule),
            host,
            drawn,
            player,
            map,
            pitch: 0.0,
            unstepped: 0.0,
            presses: Vec::new(),
            scripts,
            messages,
            physics,
            animated,
            poses: Vec::new(),
            viewer: Vec3::ZERO,
            flying: false,
            track,
            chase: Chase::default(),
            hud: None,
            race_log,
        })
    }

    /// The circuit, if the scene is one.
    pub fn track(&self) -> Option<&Track> {
        self.track.as_deref()
    }

    /// The player's car's instruments as of the latest frame, if the player drives a car.
    pub fn hud(&self) -> Option<&Hud> {
        self.hud.as_ref()
    }

    /// Whether the player is a car, whose instruments [`Play::hud`] has while it is driven.
    pub fn drives(&self) -> bool {
        self.player.as_ref().is_some_and(|player| player.car)
    }

    /// Gives every animated actor its skeleton and clips, from what the scene holds. Returns every
    /// mistake found.
    pub fn attach_animation(
        &mut self,
        files: &[GameFile],
        assets: &AnimationAssets<'_>,
    ) -> Result<(), Vec<String>> {
        let actors: Vec<_> = self
            .drawn
            .iter()
            .filter_map(|d| {
                let path = d.animation.as_ref()?;
                // A rigid mesh can sit beside the skinned ones: the first that has a pose.
                let posed = d
                    .instances
                    .iter()
                    .map(|(id, _)| *id)
                    .find(|id| {
                        assets
                            .instance_poses
                            .get(*id as usize)
                            .is_some_and(|pose| *pose != NO_POSE)
                    })
                    .or_else(|| d.instances.first().map(|(id, _)| *id))?;
                Some((d.entity, path.clone(), posed))
            })
            .collect();
        *self.animated.borrow_mut() = AnimatedActors::load(files, assets, &actors)?;
        Ok(())
    }

    /// Where the camera is, which decides how often distant characters update, and is what
    /// scripts get from `viewer()`.
    pub fn set_viewer(&mut self, position: Vec3) {
        self.viewer = position;
    }

    /// Takes the poses to give the renderer: each pose's number, and its bones in the mesh's
    /// own space.
    pub fn take_poses(&mut self) -> Vec<(u32, Vec<Mat4>)> {
        std::mem::take(&mut self.poses)
    }

    /// Every renderer instance an actor draws with, which must be made movable before play.
    pub fn instances(&self) -> impl Iterator<Item = u32> + '_ {
        self.drawn
            .iter()
            .flat_map(|d| d.instances.iter().map(|(id, _)| *id))
    }

    /// The scripts the game runs, by name, which a viewer may watch and [`Play::reload`].
    pub fn scripts(&self) -> &[String] {
        &self.scripts
    }

    /// Whether some actor is the player, with a camera to look through.
    pub fn has_player(&self) -> bool {
        self.player.is_some()
    }

    /// Whether the camera is the player's now: there is a player, and they are not flying.
    pub fn has_camera(&self) -> bool {
        self.player.is_some() && !self.flying
    }

    /// Lets the player leave whoever they play and fly the camera, or brings them back. Flying,
    /// [`Play::frame`] gives no view, and the script of the one they left hears no input, so
    /// the keys that fly the camera don't walk them too. Everything else hears it all: a door
    /// opens to a camera flown up to it.
    pub fn set_flying(&mut self, flying: bool) {
        self.flying = flying;
    }

    /// Whether the player is flying the camera, away from whoever they play.
    pub fn flying(&self) -> bool {
        self.flying && self.player.is_some()
    }

    /// Presses an action in the next frame, whatever the devices do: for scripted runs.
    pub fn press(&mut self, action: &str) {
        self.presses.push(action.to_owned());
    }

    /// Runs a frame `seconds` long, with the devices as they are. `place` is told where to draw
    /// each instance that moved. Returns where the player's camera is, if there is a player
    /// and they are not [flying](Play::set_flying).
    ///
    /// Movement (the mouse, dragging) builds up in `devices` until a step uses it, and is
    /// given to that step as a rate over the time since the step before. So none is lost in
    /// frames that run no step, as at 144 Hz, and none is counted twice. Looking up and down
    /// follows the movement every frame, for a camera that never lags the mouse.
    pub fn frame(
        &mut self,
        seconds: f64,
        devices: &mut Devices,
        mut place: impl FnMut(u32, Mat4),
    ) -> Option<View> {
        self.unstepped += seconds;
        let mut input = self.map.actions(devices, self.unstepped as f32);
        input.set_viewer(self.viewer);
        let flying = self.flying();
        input.set_away(flying);
        for action in self.presses.drain(..) {
            input.press(&action);
        }
        let steps = self.game.frame(seconds, &input);
        devices.end_frame();
        self.animated.borrow_mut().frame(
            steps.alpha,
            &self.game.world,
            self.viewer,
            &mut self.poses,
        );
        // Flying, the mouse turns the flown camera: the player's own looks where it did.
        let looked = if flying { 0.0 } else { input.value("look_up") };
        let pitch = (self.pitch + looked * self.unstepped as f32).clamp(-MAX_PITCH, MAX_PITCH);
        if steps.steps > 0 {
            devices.clear_movement();
            self.unstepped = 0.0;
            self.pitch = pitch;
        }
        if input.pressed("camera") {
            self.chase.view = self.chase.view.next();
        }
        self.gather_messages();
        let entities = &self.game.world.entities;
        for drawn in &mut self.drawn {
            let Ok(global) = entities.get::<&GlobalTransform>(drawn.entity) else {
                continue;
            };
            let world = global.interpolated(steps.alpha);
            // A car's wheels, between the last two steps as the car is. Only a car's instances
            // name wheels, so nothing else looks for them.
            let car = drawn.wheels.iter().any(Option::is_some);
            let wheels = car
                .then(|| entities.get::<&Wheels>(drawn.entity).ok())
                .flatten();
            let rolled = wheels.map(|wheels| {
                std::array::from_fn(|index| {
                    let (before, now) = (&wheels.before[index], &wheels.now[index]);
                    (
                        DrawnWheel::between(before, now, steps.alpha),
                        DrawnWheel::upright(before, now, steps.alpha),
                    )
                })
            });
            if drawn.drawn == Some(world) && drawn.rolled == rolled {
                continue;
            }
            let moved = drawn.drawn != Some(world);
            for ((id, local), wheel) in drawn.instances.iter().zip(&drawn.wheels) {
                match (wheel, &rolled) {
                    (Some((wheel, true)), Some(rolled)) => {
                        place(*id, world * rolled[*wheel].0 * *local)
                    }
                    (Some((wheel, false)), Some(rolled)) => {
                        place(*id, world * rolled[*wheel].1 * *local)
                    }
                    _ if moved => place(*id, world * *local),
                    _ => {}
                }
            }
            drawn.drawn = Some(world);
            drawn.rolled = rolled;
        }

        if flying {
            // The car is not being driven, and its instruments go with the player.
            self.hud = None;
        }
        let player = self.player.as_ref().filter(|_| !flying)?;
        let world = entities
            .get::<&GlobalTransform>(player.entity)
            .ok()?
            .interpolated(steps.alpha);
        if player.car {
            let car = entities.get::<&Car>(player.entity).ok()?;
            let telemetry = *car.vehicle.telemetry();
            let look = drive::Look {
                back: input.held("look_back"),
                around: input.value("look_round"),
                over: input.value("look_over"),
            };
            let shot = self
                .chase
                .frame(world, telemetry.speed, look, seconds as f32);
            // Filled in place each frame: the lap's times are copied into what the last frame
            // had, and the track's name is set once.
            let hud = self.hud.get_or_insert_with(|| Hud {
                telemetry,
                limit: 0.0,
                throttle: 0.0,
                brake: 0.0,
                steer: 0.0,
                lap: None,
                track: self
                    .track
                    .as_ref()
                    .map(|track| track.name.clone())
                    .unwrap_or_default(),
                view: self.chase.view,
            });
            hud.telemetry = telemetry;
            hud.limit = car.vehicle.spec().engine.limit;
            hud.throttle = car.controls.throttle;
            hud.brake = car.controls.brake;
            hud.steer = car.controls.steer;
            hud.view = self.chase.view;
            match (entities.get::<&Lap>(player.entity).ok(), &mut hud.lap) {
                (Some(lap), Some(shown)) => shown.clone_from(&lap),
                (Some(lap), shown) => *shown = Some((*lap).clone()),
                (None, shown) => *shown = None,
            }
            return Some(View {
                position: shot.position,
                yaw: shot.yaw,
                pitch: shot.pitch,
            });
        }
        let (_, rotation, translation) = world.to_scale_rotation_translation();
        Some(View {
            position: translation + rotation * player.eye,
            yaw: rotation.to_euler(EulerRot::YXZ).0,
            pitch,
        })
    }

    /// Loads a new version of a script, keeping every actor's state. If it fails to load, the
    /// old one keeps running, and the error says why.
    pub fn reload(&mut self, name: &str, source: &str) -> Result<(), String> {
        self.host
            .borrow_mut()
            .load(name, source)
            .map_err(|e| e.message)
    }

    /// What scripts have logged, and every error, since this was last asked, oldest first.
    pub fn take_messages(&mut self) -> Vec<Message> {
        self.gather_messages();
        self.messages.drain(..).collect()
    }

    /// The simulation, for tests and tools to look into.
    pub fn physics(&self) -> std::cell::Ref<'_, Physics> {
        self.physics.borrow()
    }

    /// The world, for tests and tools to look into.
    pub fn world(&self) -> &World {
        &self.game.world
    }

    fn gather_messages(&mut self) {
        let mut host = self.host.borrow_mut();
        let world = &self.game.world;
        let who = |entity: Option<Entity>| {
            entity
                .and_then(|e| {
                    world
                        .entities
                        .get::<&Name>(e)
                        .ok()
                        .map(|n| n.as_str().to_owned())
                })
                .unwrap_or_default()
        };
        for error in host.take_errors() {
            let text = match error.step {
                Some(step) => format!("step {step}, {}: {}", who(error.entity), error.message),
                None => error.message,
            };
            self.messages.push_back(Message { error: true, text });
        }
        for line in host.take_log() {
            let text = format!("{}: {}", who(Some(line.entity)), line.text);
            self.messages.push_back(Message { error: false, text });
        }
        for text in self.race_log.borrow_mut().drain(..) {
            self.messages.push_back(Message { error: false, text });
        }
        // What the physics finds wrong while playing, not only while loading.
        for text in self.physics.borrow_mut().warnings.drain(..) {
            self.messages.push_back(Message { error: true, text });
        }
        let dropped = host.take_dropped();
        if dropped > 0 {
            self.messages.push_back(Message {
                error: true,
                text: format!("{dropped} more lines dropped"),
            });
        }
        while self.messages.len() > MAX_MESSAGES {
            self.messages.pop_front();
        }
    }
}

fn hecs_builder() -> solstice_world::hecs::EntityBuilder {
    solstice_world::hecs::EntityBuilder::new()
}
