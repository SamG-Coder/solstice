//! The engine, as scripts see it: the functions every script can call.
//!
//! This one module is the engine's whole API for scripts. Starlark's own tools read it, with
//! its doc comments, for editors and for [`crate::host::api_reference`].

use crate::host::{ScriptEvent, ScriptLog};
use crate::value::{EntityValue, ScriptValue, Vec3Value};
use solstice_world::{Entity, TimerId, World};
use starlark::any::ProvidesStaticType;
use starlark::environment::GlobalsBuilder;
use starlark::eval::Evaluator;
use starlark::starlark_module;
use starlark::values::float::UnpackFloat;
use starlark::values::list::AllocList;
use starlark::values::none::NoneType;
use starlark::values::tuple::UnpackTuple;
use starlark::values::{Value, ValueLike};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

/// What a handler's calls into the engine reach: the world, the actor it runs for, and the
/// host's own records.
#[derive(ProvidesStaticType)]
pub(crate) struct Context<'w> {
    pub world: RefCell<&'w mut World>,
    pub actor: Entity,
    pub timers: RefCell<&'w mut Timers>,
    pub log: RefCell<&'w mut Vec<ScriptLog>>,
}

/// Timers scripts have set, by the handle they were given.
#[derive(Default)]
pub(crate) struct Timers {
    next: i32,
    ids: HashMap<i32, TimerId>,
}

impl Timers {
    fn add(&mut self, id: TimerId) -> i32 {
        self.next += 1;
        self.ids.insert(self.next, id);
        self.next
    }
}

fn context<'a, 'e>(eval: &Evaluator<'_, 'a, 'e>) -> anyhow::Result<&'a Context<'e>> {
    eval.extra
        .and_then(|extra| extra.downcast_ref::<Context>())
        .ok_or_else(|| anyhow::anyhow!("engine functions can only be called from a handler"))
}

/// Builds a script event from a name and Starlark arguments.
fn event(to: Option<Entity>, name: &str, args: &[Value]) -> anyhow::Result<ScriptEvent> {
    if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        anyhow::bail!("event names are made of letters, digits and _, not `{name}`");
    }
    let args = args
        .iter()
        .map(|&arg| ScriptValue::from_starlark(arg).map_err(anyhow::Error::msg))
        .collect::<anyhow::Result<_>>()?;
    Ok(ScriptEvent {
        to,
        name: Arc::from(name),
        args,
    })
}

#[starlark_module]
pub(crate) fn engine(builder: &mut GlobalsBuilder) {
    /// A vector: `vec3(1, 0, 0)`.
    fn vec3(x: UnpackFloat, y: UnpackFloat, z: UnpackFloat) -> anyhow::Result<Vec3Value> {
        Ok(Vec3Value(glam::Vec3::new(
            x.0 as f32, y.0 as f32, z.0 as f32,
        )))
    }

    /// Sends an event to an entity, which its script handles in `on_<name>(self, ...)`, or to
    /// every actor whose script handles it, if `to` is `None`. Handlers run next step.
    fn send<'v>(
        to: Value<'v>,
        name: &str,
        #[starlark(args)] args: UnpackTuple<Value<'v>>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let to = if to.is_none() {
            None
        } else {
            Some(
                to.downcast_ref::<EntityValue>()
                    .ok_or_else(|| anyhow::anyhow!("send to an entity or None, not {to}"))?
                    .0,
            )
        };
        let event = event(to, name, &args.items)?;
        context(eval)?.world.borrow_mut().send(event);
        Ok(NoneType)
    }

    /// Sends an event to this actor after `seconds`, rounded up to whole steps. Returns a
    /// handle to cancel it.
    fn after<'v>(
        seconds: UnpackFloat,
        name: &str,
        #[starlark(args)] args: UnpackTuple<Value<'v>>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<i32> {
        let context = context(eval)?;
        let event = event(Some(context.actor), name, &args.items)?;
        let id = context.world.borrow_mut().after(seconds.0, event);
        Ok(context.timers.borrow_mut().add(id))
    }

    /// Sends an event to this actor every `seconds`, rounded up to whole steps, until cancelled.
    /// Returns a handle to cancel it.
    fn every<'v>(
        seconds: UnpackFloat,
        name: &str,
        #[starlark(args)] args: UnpackTuple<Value<'v>>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<i32> {
        let context = context(eval)?;
        let event = event(Some(context.actor), name, &args.items)?;
        let id = context.world.borrow_mut().every(seconds.0, event);
        Ok(context.timers.borrow_mut().add(id))
    }

    /// Stops a timer set by `after` or `every`. Returns whether it was still waiting.
    fn cancel(timer: i32, eval: &mut Evaluator) -> anyhow::Result<bool> {
        let context = context(eval)?;
        let Some(id) = context.timers.borrow_mut().ids.remove(&timer) else {
            return Ok(false);
        };
        Ok(context.world.borrow_mut().cancel(id))
    }

    /// Every entity with this name, in the world's order.
    fn find<'v>(name: &str, eval: &mut Evaluator<'v, '_, '_>) -> anyhow::Result<Value<'v>> {
        let heap = eval.heap();
        let context = context(eval)?;
        let world = context.world.borrow();
        let found: Vec<Entity> = world
            .entities
            .query::<(Entity, &crate::host::Name)>()
            .iter()
            .filter(|(_, n)| &*n.0 == name)
            .map(|(entity, _)| entity)
            .collect();
        Ok(heap.alloc(AllocList(found.into_iter().map(EntityValue))))
    }

    /// An input action's value this step: 1 held, 0 released, between for a stick.
    fn input(action: &str, eval: &mut Evaluator) -> anyhow::Result<f64> {
        Ok(f64::from(
            context(eval)?.world.borrow().input().value(action),
        ))
    }

    /// Whether an input action is held this step.
    fn held(action: &str, eval: &mut Evaluator) -> anyhow::Result<bool> {
        Ok(context(eval)?.world.borrow().input().held(action))
    }

    /// Whether an input action was pressed since the step before.
    fn pressed(action: &str, eval: &mut Evaluator) -> anyhow::Result<bool> {
        Ok(context(eval)?.world.borrow().input().pressed(action))
    }

    /// Simulated seconds since the game started.
    fn now(eval: &mut Evaluator) -> anyhow::Result<f64> {
        Ok(context(eval)?.world.borrow().seconds())
    }

    /// Writes to the game's log, with the step and the actor.
    fn log<'v>(
        #[starlark(args)] args: UnpackTuple<Value<'v>>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let context = context(eval)?;
        let text = args
            .items
            .iter()
            .map(|v| v.unpack_str().map_or_else(|| v.to_str(), str::to_owned))
            .collect::<Vec<_>>()
            .join(" ");
        let step = context.world.borrow().step();
        context.log.borrow_mut().push(ScriptLog {
            step,
            entity: context.actor,
            text,
        });
        Ok(NoneType)
    }
}
