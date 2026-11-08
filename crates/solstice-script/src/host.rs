//! Script classes, the actors that use them, and running their handlers each step.
//!
//! A script is a class of actor: a file of handlers. `on_begin(self)` runs once, the first step
//! an actor has the script; `on_tick(self, dt)` runs every step, if the script has one; and
//! `on_<name>(self, ...)` runs for each [`ScriptEvent`] named `<name>` sent to the actor, or to
//! every actor, by a script or by Rust.
//!
//! Each step, in `PrePhysics`, handlers run in a fixed order: beginnings, then events in the
//! order they were sent, then ticks, each in the world's order of entities.
//!
//! **Loading.** A script is parsed, run once, and frozen. Its handlers are then shared by every
//! actor that uses it. Loading a script again under the same name replaces it at once: actors
//! keep their state, which lives in their components, and run the new handlers from their next
//! call. A script that fails to load leaves the one before running, and reports why.
//!
//! **Limits.** Each call has a budget of work, counted in Starlark's own steps, so that it stops
//! at the same point natively and on the web, and of memory. A handler that runs out of either,
//! or fails any other way, stops; its error, with file and line, is reported; and it changes
//! nothing. The actor carries on with its next call.

use crate::actor::ActorValue;
use crate::api::{engine, Context, Timers};
use crate::value::ScriptValue;
use glam::{EulerRot, Quat, Vec3};
use solstice_world::{Entity, Schedule, TickGroup, Transform, World, STEP_SECONDS};
use starlark::environment::{FrozenModule, Globals, GlobalsBuilder, Module};
use starlark::eval::Evaluator;
use starlark::syntax::{AstModule, Dialect};
use starlark::values::OwnedFrozenValue;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

/// An actor's script: which one, and the values it keeps.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Script {
    /// The script's name, as it was loaded.
    pub class: Arc<str>,
    /// The properties the actor's file declares, which the script can read and set as
    /// `self.<name>`, but not add to.
    pub properties: BTreeMap<String, ScriptValue>,
    /// Anything else the script keeps: `self.vars.<name>`.
    pub vars: BTreeMap<String, ScriptValue>,
    /// Whether `on_begin` has run.
    begun: bool,
}

impl Script {
    pub fn new(class: &str) -> Self {
        Self {
            class: Arc::from(class),
            ..Self::default()
        }
    }

    pub fn with_property(mut self, name: &str, value: impl Into<ScriptValue>) -> Self {
        self.properties.insert(name.to_owned(), value.into());
        self
    }
}

/// An entity's name, for scripts to `find` it by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name(pub Arc<str>);

impl Name {
    pub fn new(name: &str) -> Self {
        Self(Arc::from(name))
    }
}

/// An event for scripts: handled by `on_<name>(self, *args)`, in the actor `to`, or in every
/// actor whose script handles it.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptEvent {
    pub to: Option<Entity>,
    pub name: Arc<str>,
    pub args: Vec<ScriptValue>,
}

/// A script that failed to load, or a handler that failed.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptError {
    /// The step it happened on.
    pub step: u64,
    pub script: Arc<str>,
    /// The actor whose handler failed, or none for a script that failed to load.
    pub entity: Option<Entity>,
    /// What went wrong, with the file, line and call stack.
    pub message: String,
}

/// A line a script wrote with `log`.
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptLog {
    pub step: u64,
    pub entity: Entity,
    pub text: String,
}

/// How much one handler call may do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    /// Starlark's steps of work: each statement and loop turn counts.
    pub work: u64,
    /// Bytes of memory the call may hold.
    pub memory: usize,
}

impl Default for Limits {
    fn default() -> Self {
        // About 2 ms of work on the web: a hundred times what a handler usually needs, and
        // short enough that a runaway one costs one hitch of a frame rather than a stall.
        Self {
            work: 100_000,
            memory: 16 << 20,
        }
    }
}

/// A loaded script: its frozen module and the handlers in it.
struct Class {
    handlers: HashMap<String, OwnedFrozenValue>,
    /// Kept so the handlers' heap stays alive.
    _module: FrozenModule,
}

/// Every loaded script, and what running them has produced.
pub struct ScriptHost {
    globals: Globals,
    classes: HashMap<Arc<str>, Class>,
    pub limits: Limits,
    errors: Vec<ScriptError>,
    log: Vec<ScriptLog>,
    timers: Timers,
    /// Scripts named by actors but never loaded, reported once each.
    missing: HashSet<Arc<str>>,
}

/// The language scripts are written in: Starlark, with types and f-strings.
fn dialect() -> Dialect {
    Dialect {
        enable_f_strings: true,
        ..Dialect::Extended
    }
}

impl Default for ScriptHost {
    fn default() -> Self {
        Self::new()
    }
}

impl ScriptHost {
    pub fn new() -> Self {
        Self {
            globals: GlobalsBuilder::standard().with(engine).build(),
            classes: HashMap::new(),
            limits: Limits::default(),
            errors: Vec::new(),
            log: Vec::new(),
            timers: Timers::default(),
            missing: HashSet::new(),
        }
    }

    /// Loads a script under `name`, or loads it again, replacing the one before. On failure the
    /// one before, if any, stays, and the error is reported as well as returned.
    pub fn load(&mut self, name: &str, source: &str) -> Result<(), ScriptError> {
        let name: Arc<str> = Arc::from(name);
        let limits = self.limits;
        let globals = &self.globals;
        let loaded = (|| -> anyhow::Result<Class> {
            let ast = AstModule::parse(&name, source.to_owned(), &dialect())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let module = Module::with_temp_heap(|module| -> anyhow::Result<FrozenModule> {
                {
                    let mut eval = Evaluator::new(&module);
                    eval.set_max_tick_count(limits.work)?;
                    eval.eval_module(ast, globals)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                }
                module.freeze().map_err(|e| anyhow::anyhow!("{e:?}"))
            })?;
            let mut handlers = HashMap::new();
            for handler in module.names() {
                let handler = handler.as_str();
                if handler.starts_with("on_") {
                    handlers.insert(handler.to_owned(), module.get(handler)?);
                }
            }
            Ok(Class {
                handlers,
                _module: module,
            })
        })();
        match loaded {
            Ok(class) => {
                self.missing.remove(&name);
                self.classes.insert(name, class);
                Ok(())
            }
            Err(error) => {
                let error = ScriptError {
                    step: 0,
                    script: name,
                    entity: None,
                    message: error.to_string(),
                };
                self.errors.push(error.clone());
                Err(error)
            }
        }
    }

    /// Whether a script of this name is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.classes.contains_key(name)
    }

    /// Errors since they were last taken.
    pub fn take_errors(&mut self) -> Vec<ScriptError> {
        std::mem::take(&mut self.errors)
    }

    /// Lines scripts have logged since they were last taken.
    pub fn take_log(&mut self) -> Vec<ScriptLog> {
        std::mem::take(&mut self.log)
    }

    /// Adds the system that runs scripts each step, in `PrePhysics`.
    pub fn install(host: &Rc<RefCell<Self>>, schedule: &mut Schedule) {
        let host = host.clone();
        schedule.add(
            TickGroup::PrePhysics,
            "scripts",
            move |world: &mut World| {
                host.borrow_mut().run(world);
            },
        );
    }

    /// Runs this step's handlers: beginnings, then events, then ticks.
    pub fn run(&mut self, world: &mut World) {
        let mut calls: Vec<(Entity, String, Vec<ScriptValue>)> = Vec::new();
        let actors: Vec<(Entity, Arc<str>, bool)> = world
            .entities
            .query::<(Entity, &Script)>()
            .iter()
            .map(|(entity, script)| (entity, script.class.clone(), script.begun))
            .collect();
        for (_, class, _) in &actors {
            if !self.classes.contains_key(class) && self.missing.insert(class.clone()) {
                self.errors.push(ScriptError {
                    step: world.step(),
                    script: class.clone(),
                    entity: None,
                    message: format!("no script named `{class}` is loaded"),
                });
            }
        }
        let handles = |class: &Arc<str>, handler: &str| {
            self.classes
                .get(class)
                .is_some_and(|c| c.handlers.contains_key(handler))
        };

        for (entity, class, begun) in &actors {
            if !begun && self.classes.contains_key(class) {
                if let Ok(mut script) = world.entities.get::<&mut Script>(*entity) {
                    script.begun = true;
                }
                if handles(class, "on_begin") {
                    calls.push((*entity, "on_begin".to_owned(), Vec::new()));
                }
            }
        }
        for event in world.read_cloned::<ScriptEvent>() {
            let handler = format!("on_{}", event.name);
            for (entity, class, _) in &actors {
                if event.to.is_none_or(|to| to == *entity) && handles(class, &handler) {
                    calls.push((*entity, handler.clone(), event.args.clone()));
                }
            }
        }
        for (entity, class, _) in &actors {
            if handles(class, "on_tick") {
                calls.push((
                    *entity,
                    "on_tick".to_owned(),
                    vec![ScriptValue::Float(STEP_SECONDS)],
                ));
            }
        }

        for (entity, handler, args) in calls {
            self.call(world, entity, &handler, &args);
        }
    }

    /// Runs one handler for one actor, and writes back what it changed if it succeeds.
    fn call(&mut self, world: &mut World, entity: Entity, handler: &str, args: &[ScriptValue]) {
        let Some(class) = world
            .entities
            .get::<&Script>(entity)
            .ok()
            .map(|s| s.class.clone())
        else {
            return;
        };
        let Some(function) = self
            .classes
            .get(&class)
            .and_then(|class| class.handlers.get(handler))
            .cloned()
        else {
            return;
        };
        let transform = world
            .entities
            .get::<&Transform>(entity)
            .map(|t| *t)
            .unwrap_or_default();
        let (yaw, pitch, roll) = transform.rotation.to_euler(EulerRot::YXZ);
        let step = world.step();
        let limits = self.limits;

        let result = Module::with_temp_heap(|module| -> Result<_, String> {
            let heap = module.heap();
            // The script's view of the actor is a copy on the call's heap, made straight from
            // its component.
            let actor = {
                let script = world
                    .entities
                    .get::<&Script>(entity)
                    .map_err(|e| e.to_string())?;
                ActorValue::alloc(
                    heap,
                    entity,
                    &script.properties,
                    &script.vars,
                    transform.translation,
                    yaw,
                )
            };
            let context = Context {
                world: RefCell::new(&mut *world),
                actor: entity,
                timers: RefCell::new(&mut self.timers),
                log: RefCell::new(&mut self.log),
            };
            let mut eval = Evaluator::new(&module);
            eval.extra = Some(&context);
            eval.set_max_tick_count(limits.work)
                .map_err(|e| e.to_string())?;
            eval.set_max_heap_size(limits.memory)
                .map_err(|e| e.to_string())?;
            let mut values = vec![actor];
            values.extend(args.iter().map(|arg| arg.to_starlark(heap)));
            // SAFETY: this ties the handler's frozen heap to the call's module, which keeps it
            // alive for as long as the module, and so the call, lasts.
            let function = unsafe { function.owned_frozen_value(module.frozen_heap()) };
            eval.eval_function(function.to_value(), &values, &[])
                .map_err(|e| e.to_string())?;
            // Starlark checks the heap only now and then while a call runs, so check it once
            // more at the end: a call that went over fails, whenever it did.
            let used = heap.peak_allocated_bytes();
            if used > limits.memory {
                return Err(format!(
                    "the call used {} bytes of memory, more than the {} it may",
                    used, limits.memory
                ));
            }
            ActorValue::state(actor)
        });

        match result {
            Ok(state) => {
                if let Ok(mut script) = world.entities.get::<&mut Script>(entity) {
                    script.properties = state.properties;
                    script.vars = state.vars;
                }
                let placed = match world.entities.get::<&mut Transform>(entity) {
                    Ok(mut t) => {
                        t.translation = state.position;
                        if state.yaw != yaw {
                            t.rotation = Quat::from_euler(EulerRot::YXZ, state.yaw, pitch, roll);
                        }
                        true
                    }
                    Err(_) => false,
                };
                if !placed && (state.position != Vec3::ZERO || state.yaw != 0.0) {
                    world.commands().insert_one(
                        entity,
                        Transform::from_translation(state.position)
                            .with_rotation(Quat::from_rotation_y(state.yaw)),
                    );
                }
            }
            Err(message) => self.errors.push(ScriptError {
                step,
                script: class,
                entity: Some(entity),
                message: format!("{handler}: {message}"),
            }),
        }
    }
}

/// The engine's API for scripts, as Starlark's documentation renders it: each function, its
/// arguments and what it does.
pub fn api_reference() -> String {
    let globals = GlobalsBuilder::new().with(engine).build();
    let mut out = String::new();
    for (name, item) in globals.documentation().members {
        out += &starlark::docs::markdown::render_doc_item_no_link(&name, &item);
        out += "\n\n";
    }
    out
}
