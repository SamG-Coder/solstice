//! Play mode: a scene's game, running.
//!
//! [`Play::load`] builds a world from a baked scene's gameplay files: an entity for each placed
//! actor, with its name, its place, and its script with the placement's property values. Each
//! frame, [`Play::frame`] runs the game, then says where to draw each actor's instances, between
//! the last two steps, and where the player's camera is.

use crate::actor::ActorFile;
use crate::animation::{AnimatedActors, AnimationAssets};
use crate::input::{Devices, InputMap};
use crate::resolve;
use crate::scene::{SceneGame, GAME_FILE};
use glam::{EulerRot, Mat4, Quat, Vec3};
use solstice_asset::NO_POSE;
use solstice_asset::{Collision, GameFile};
use solstice_physics::{Overlap, Physics};
use solstice_script::{Script, ScriptEvent, ScriptHost};
use solstice_world::{Entity, Game, GlobalTransform, Name, Schedule, TickGroup, Transform, World};
use std::cell::RefCell;
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
    /// Each instance, and where it is relative to the actor.
    instances: Vec<(u32, Mat4)>,
    /// Where the actor was last drawn, to move its instances only when it moves.
    drawn: Option<Mat4>,
}

/// The actor the player plays, and where its eye is.
struct Player {
    entity: Entity,
    eye: Vec3,
}

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
        let map = match &scene.input {
            None => InputMap::default(),
            Some(name) => match file(name).map(|text| InputMap::parse(text)) {
                Some(Ok(map)) => map,
                Some(Err(e)) => {
                    errors.push(format!("{name}: {e}"));
                    InputMap::default()
                }
                None => {
                    errors.push(format!("{name}: missing from the scene's gameplay"));
                    InputMap::default()
                }
            },
        };

        let host = Rc::new(RefCell::new(ScriptHost::new()));
        let physics = Rc::new(RefCell::new(Physics::new()));
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
            let entity = world.spawn(components.build());
            if let Some(eye) = actor.camera {
                if player.is_some() {
                    errors.push(format!(
                        "{at}: only one actor can have a camera, and one already does"
                    ));
                }
                player = Some(Player { entity, eye });
            }
            if let Some(range) = &line.instances {
                let placed = Mat4::from_rotation_translation(rotation, line.position);
                let mut instances = Vec::new();
                for id in range.clone() {
                    match instance(id) {
                        Some(world) => instances.push((id, placed.inverse() * world)),
                        None => {
                            errors.push(format!("{at}: the scene has no instance {id}"));
                            break;
                        }
                    }
                }
                drawn.push(Drawn {
                    entity,
                    animation: actor.animation.as_ref().map(|a| resolve(&line.file, a)),
                    instances,
                    drawn: None,
                });
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }

        let mut schedule = Schedule::new();
        ScriptHost::install(&host, &mut schedule);
        Physics::install(&physics, &mut schedule);
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
        })
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

    /// Where the camera is, which decides how often distant characters update.
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

    /// Presses an action in the next frame, whatever the devices do: for scripted runs.
    pub fn press(&mut self, action: &str) {
        self.presses.push(action.to_owned());
    }

    /// Runs a frame `seconds` long, with the devices as they are. `place` is told where to draw
    /// each instance that moved. Returns where the player's camera is, if there is a player.
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
        let pitch = (self.pitch + input.value("look_up") * self.unstepped as f32)
            .clamp(-MAX_PITCH, MAX_PITCH);
        if steps.steps > 0 {
            devices.clear_movement();
            self.unstepped = 0.0;
            self.pitch = pitch;
        }
        self.gather_messages();
        let entities = &self.game.world.entities;
        for drawn in &mut self.drawn {
            let Ok(global) = entities.get::<&GlobalTransform>(drawn.entity) else {
                continue;
            };
            let world = global.interpolated(steps.alpha);
            if drawn.drawn == Some(world) {
                continue;
            }
            for (id, local) in &drawn.instances {
                place(*id, world * *local);
            }
            drawn.drawn = Some(world);
        }

        let player = self.player.as_ref()?;
        let world = entities
            .get::<&GlobalTransform>(player.entity)
            .ok()?
            .interpolated(steps.alpha);
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
