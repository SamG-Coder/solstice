//! Scripts running in a world: handlers, state, events, timers, reloading, limits and errors.

use glam::Vec3;
use solstice_script::{api_reference, Name, Script, ScriptEvent, ScriptHost, ScriptValue};
use solstice_world::{Entity, Game, Input, Schedule, Transform, World, STEP_SECONDS};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

/// A game with a script host whose scripts are loaded from `scripts`, as (name, source).
fn game_with(scripts: &[(&str, &str)]) -> (Game, Rc<RefCell<ScriptHost>>) {
    let host = Rc::new(RefCell::new(ScriptHost::new()));
    for (name, source) in scripts {
        host.borrow_mut()
            .load(name, source)
            .unwrap_or_else(|e| panic!("{name} failed to load: {}", e.message));
    }
    let mut schedule = Schedule::new();
    ScriptHost::install(&host, &mut schedule);
    (Game::new(World::new(), schedule), host)
}

fn steps(game: &mut Game, count: u32) {
    for _ in 0..count {
        game.step(&Input::new());
    }
}

fn var(game: &Game, entity: Entity, name: &str) -> Option<ScriptValue> {
    game.world
        .entities
        .get::<&Script>(entity)
        .unwrap()
        .vars
        .get(name)
        .cloned()
}

fn property(game: &Game, entity: Entity, name: &str) -> Option<ScriptValue> {
    game.world
        .entities
        .get::<&Script>(entity)
        .unwrap()
        .properties
        .get(name)
        .cloned()
}

const GATE: &str = r#"
def on_begin(self):
    self.vars.open = False
    self.vars.opened = 0

def on_interact(self, who):
    if self.locked:
        log("locked against", who)
        return
    self.vars.open = not self.vars.open
    self.vars.opened += 1
    self.yaw = 1.5 if self.vars.open else 0.0

def on_lock(self):
    self.locked = True
"#;

const VISITOR: &str = r#"
def on_tick(self, dt):
    gate = find("gate")[0]
    if now() < 3 * dt:
        send(gate, "interact", self.entity)
"#;

#[test]
fn scripts_keep_state_handle_events_and_move_their_actor() {
    let (mut game, host) = game_with(&[("gate", GATE), ("visitor", VISITOR)]);
    let gate = game.world.spawn((
        Name::new("gate"),
        Transform::default(),
        Script::new("gate").with_property("locked", false),
    ));
    let visitor = game.world.spawn((Script::new("visitor"),));
    // The visitor knocks in steps 1 and 2; each knock is handled the step after.
    steps(&mut game, 4);
    assert_eq!(host.borrow_mut().take_errors(), vec![]);
    assert_eq!(var(&game, gate, "opened"), Some(ScriptValue::Int(2)));
    assert_eq!(var(&game, gate, "open"), Some(ScriptValue::Bool(false)));
    // Open again: the gate turns.
    game.world.send(ScriptEvent {
        to: Some(gate),
        name: Arc::from("interact"),
        args: vec![ScriptValue::Entity(visitor)],
    });
    steps(&mut game, 1);
    let yaw = game
        .world
        .entities
        .get::<&Transform>(gate)
        .unwrap()
        .rotation;
    assert!((yaw.to_euler(glam::EulerRot::YXZ).0 - 1.5).abs() < 1.0e-5);
    // Locked by a broadcast, it logs instead.
    game.world.send(ScriptEvent {
        to: None,
        name: Arc::from("lock"),
        args: vec![],
    });
    steps(&mut game, 1);
    assert_eq!(
        property(&game, gate, "locked"),
        Some(ScriptValue::Bool(true))
    );
    game.world.send(ScriptEvent {
        to: Some(gate),
        name: Arc::from("interact"),
        args: vec![ScriptValue::Entity(visitor)],
    });
    steps(&mut game, 1);
    let log = host.borrow_mut().take_log();
    assert_eq!(log.len(), 1);
    assert_eq!(
        log[0].text,
        format!("locked against entity({})", visitor.id())
    );
    assert_eq!(var(&game, gate, "opened"), Some(ScriptValue::Int(3)));
}

#[test]
fn ticking_moves_an_actor_every_step() {
    let (mut game, host) = game_with(&[(
        "drift",
        "def on_tick(self, dt):\n    self.position += vec3(1, 0, 0) * dt * self.speed\n",
    )]);
    let boat = game.world.spawn((
        Transform::default(),
        Script::new("drift").with_property("speed", 6.0),
    ));
    steps(&mut game, 30);
    assert!(host.borrow_mut().take_errors().is_empty());
    let x = game
        .world
        .entities
        .get::<&Transform>(boat)
        .unwrap()
        .translation
        .x;
    assert!((x - 3.0).abs() < 1.0e-3, "{x}");
}

const COUNTER: &str = r#"
def on_begin(self):
    self.vars.count = 0
    self.vars.path = []

def on_tick(self, dt):
    self.vars.count += 1
    self.vars.path.append(self.vars.count)
"#;

#[test]
fn reloading_a_script_keeps_every_actors_state() {
    let (mut game, host) = game_with(&[("counter", COUNTER)]);
    let counter = game.world.spawn((Script::new("counter"),));
    steps(&mut game, 3);
    host.borrow_mut()
        .load("counter", &COUNTER.replace("count += 1", "count += 10"))
        .unwrap();
    steps(&mut game, 2);
    assert_eq!(var(&game, counter, "count"), Some(ScriptValue::Int(23)));
    // Lists kept in self.vars are changed in place, and kept.
    let path = var(&game, counter, "path").unwrap();
    assert_eq!(
        path,
        ScriptValue::List([1, 2, 3, 13, 23].map(ScriptValue::Int).to_vec())
    );
    assert!(host.borrow_mut().take_errors().is_empty());
}

#[test]
fn a_broken_reload_keeps_the_old_script_running_and_says_where_it_broke() {
    let (mut game, host) = game_with(&[("counter", COUNTER)]);
    let counter = game.world.spawn((Script::new("counter"),));
    steps(&mut game, 2);
    let broken = COUNTER.replace("self.vars.count += 1", "self.vars.count += 1 1");
    let error = host.borrow_mut().load("counter", &broken).unwrap_err();
    assert!(error.message.contains("counter:7"), "{}", error.message);
    steps(&mut game, 2);
    assert_eq!(var(&game, counter, "count"), Some(ScriptValue::Int(4)));
    assert_eq!(host.borrow_mut().take_errors().len(), 1);
}

#[test]
fn a_failing_handler_changes_nothing_and_the_actor_carries_on() {
    let source = r#"
def on_begin(self):
    self.vars.ticks = 0

def on_tick(self, dt):
    self.vars.ticks += 1
    self.position = vec3(9, 9, 9)
    if int(now() / dt + 0.5) == 2:
        fail("bad luck")
"#;
    let (mut game, host) = game_with(&[("flaky", source)]);
    let actor = game
        .world
        .spawn((Transform::default(), Script::new("flaky")));
    steps(&mut game, 2);
    // The second tick failed: its increment and move are gone.
    let errors = host.borrow_mut().take_errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("bad luck"));
    assert!(
        errors[0].message.contains("flaky:9"),
        "{}",
        errors[0].message
    );
    assert_eq!(errors[0].entity, Some(actor));
    assert_eq!(var(&game, actor, "ticks"), Some(ScriptValue::Int(1)));
    assert_eq!(
        game.world
            .entities
            .get::<&Transform>(actor)
            .unwrap()
            .translation,
        Vec3::new(9.0, 9.0, 9.0),
        "the first tick's move stands"
    );
    steps(&mut game, 1);
    // And it carried on from where it was.
    assert_eq!(var(&game, actor, "ticks"), Some(ScriptValue::Int(2)));
}

#[test]
fn a_runaway_handler_is_stopped_and_reported_and_others_still_run() {
    let source = r#"
def on_tick(self, dt):
    for i in range(1000000000):
        if i % 1000 == 0:
            log(i)
"#;
    let (mut game, host) = game_with(&[("runaway", source), ("counter", COUNTER)]);
    game.world.spawn((Script::new("runaway"),));
    let counter = game.world.spawn((Script::new("counter"),));
    steps(&mut game, 1);
    let errors = host.borrow_mut().take_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0]
            .message
            .contains("limit of 100000 ticks has been exceeded"),
        "{}",
        errors[0].message
    );
    assert_eq!(var(&game, counter, "count"), Some(ScriptValue::Int(1)));
    // Stopped by counting work, not time: always at the same point, the same natively and on
    // the web (tools/gameplay-bench, b_script).
    assert_eq!(host.borrow_mut().take_log().len(), 101);
    steps(&mut game, 1);
    assert_eq!(host.borrow_mut().take_log().len(), 101);
}

#[test]
fn a_handler_that_hoards_memory_is_stopped() {
    let source = r#"
def on_tick(self, dt):
    big = []
    for i in range(100000):
        big.append(str(i) * 1000)
"#;
    let (mut game, host) = game_with(&[("hoarder", source)]);
    host.borrow_mut().limits.work = 100_000_000;
    game.world.spawn((Script::new("hoarder"),));
    steps(&mut game, 1);
    let errors = host.borrow_mut().take_errors();
    assert_eq!(errors.len(), 1);
    assert!(
        errors[0].message.to_lowercase().contains("heap") || errors[0].message.contains("memory"),
        "{}",
        errors[0].message
    );
}

#[test]
fn a_call_that_ends_over_its_memory_is_stopped() {
    // One large allocation at the very end of the call. Starlark checks the heap now and then,
    // and the host once more as the call ends; either way, the call is stopped.
    let source = r#"
def on_tick(self, dt):
    self.vars.big = str(self.vars.n) * (20 * 1024 * 1024)

def on_begin(self):
    self.vars.n = 7
"#;
    let (mut game, host) = game_with(&[("gulp", source)]);
    let actor = game.world.spawn((Script::new("gulp"),));
    steps(&mut game, 1);
    let errors = host.borrow_mut().take_errors();
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(
        errors[0].message.contains("memory"),
        "{}",
        errors[0].message
    );
    assert_eq!(var(&game, actor, "big"), None);
}

#[test]
fn an_event_sent_to_one_actor_reaches_only_it() {
    let source = "def on_poke(self):
    self.vars.poked = True
";
    let (mut game, host) = game_with(&[("pokeable", source)]);
    let first = game.world.spawn((Script::new("pokeable"),));
    let second = game.world.spawn((Script::new("pokeable"),));
    game.world.send(ScriptEvent {
        to: Some(second),
        name: Arc::from("poke"),
        args: vec![],
    });
    steps(&mut game, 1);
    assert!(host.borrow_mut().take_errors().is_empty());
    assert_eq!(var(&game, first, "poked"), None);
    assert_eq!(var(&game, second, "poked"), Some(ScriptValue::Bool(true)));
}

#[test]
fn tuples_are_kept_as_lists() {
    let source = "def on_tick(self, dt):
    self.vars.pair = (1, \"two\")
";
    let (mut game, _) = game_with(&[("pairs", source)]);
    let actor = game.world.spawn((Script::new("pairs"),));
    steps(&mut game, 1);
    assert_eq!(
        var(&game, actor, "pair"),
        Some(ScriptValue::List(vec![
            ScriptValue::Int(1),
            ScriptValue::from("two")
        ]))
    );
}

#[test]
fn timers_from_scripts_fire_on_their_step_and_can_be_cancelled() {
    let source = r#"
def on_begin(self):
    self.vars.rings = []
    after(0.05, "ring", "once")
    self.vars.repeat = every(0.1, "ring", "again")

def on_ring(self, what):
    self.vars.rings.append(what)
    if len(self.vars.rings) == 3:
        cancel(self.vars.repeat)
"#;
    let (mut game, host) = game_with(&[("bell", source)]);
    let bell = game.world.spawn((Script::new("bell"),));
    steps(&mut game, 1);
    // A second bell, set going a step later: its timers ring for it alone.
    let other = game.world.spawn((Script::new("bell"),));
    steps(&mut game, 59);
    assert_eq!(
        var(&game, other, "rings").map(|r| match r {
            ScriptValue::List(items) => items.len(),
            _ => 0,
        }),
        Some(3)
    );
    assert!(host.borrow_mut().take_errors().is_empty());
    let rings = var(&game, bell, "rings").unwrap();
    assert_eq!(
        rings,
        ScriptValue::List(["once", "again", "again"].map(ScriptValue::from).to_vec())
    );
}

#[test]
fn scripts_read_the_steps_input() {
    let source = r#"
def on_tick(self, dt):
    if pressed("jump"):
        self.vars.jumps = self.vars.jumps + 1 if hasattr(self.vars, "jumps") else 1
    self.position += vec3(0, 0, -1) * input("forward") * dt
"#;
    let (mut game, host) = game_with(&[("player", source)]);
    let player = game
        .world
        .spawn((Transform::default(), Script::new("player")));
    let mut input = Input::new();
    input.set("forward", 1.0);
    input.set("jump", 1.0);
    for _ in 0..10 {
        game.frame(STEP_SECONDS + 1.0e-5, &input);
    }
    assert!(host.borrow_mut().take_errors().is_empty());
    assert_eq!(var(&game, player, "jumps"), Some(ScriptValue::Int(1)));
    let z = game
        .world
        .entities
        .get::<&Transform>(player)
        .unwrap()
        .translation
        .z;
    assert!((z + 10.0 * STEP_SECONDS as f32).abs() < 1.0e-4, "{z}");
}

#[test]
fn what_cant_be_kept_or_set_is_refused_with_a_reason() {
    let source = r#"
def on_tick(self, dt):
    if self.vars.mode == 0:
        self.vars.mode = 1
        self.colour = "red"
    elif self.vars.mode == 1:
        self.vars.mode = 2
        self.vars.callback = lambda: 1

def on_begin(self):
    self.vars.mode = 0
"#;
    let (mut game, host) = game_with(&[("strict", source)]);
    let actor = game.world.spawn((Script::new("strict"),));
    steps(&mut game, 1);
    let errors = host.borrow_mut().take_errors();
    assert!(
        errors[0].message.contains("no property `colour`"),
        "{}",
        errors[0].message
    );
    // Refused, so mode never moved on: it stays 0, and the same error comes each step.
    assert_eq!(var(&game, actor, "mode"), Some(ScriptValue::Int(0)));
    let source = source.replace("        self.colour = \"red\"\n", "");
    host.borrow_mut().load("strict", &source).unwrap();
    steps(&mut game, 2);
    let errors = host.borrow_mut().take_errors();
    assert!(
        errors.iter().any(|e| e.message.contains("can't be kept")),
        "{errors:?}"
    );
}

#[test]
fn an_actor_naming_a_script_never_loaded_is_reported_once() {
    let (mut game, host) = game_with(&[]);
    game.world.spawn((Script::new("ghost"),));
    steps(&mut game, 3);
    let errors = host.borrow_mut().take_errors();
    assert_eq!(errors.len(), 1);
    assert!(errors[0].message.contains("`ghost`"));
}

#[test]
fn a_scripted_run_replays_exactly() {
    let source = r#"
def on_begin(self):
    self.vars.wobble = 0.0
    every(0.25, "turn")

def on_turn(self):
    self.vars.wobble = -self.vars.wobble + 0.1 * input("steer")

def on_tick(self, dt):
    self.position += vec3(self.vars.wobble, 0, 1) * dt
    self.yaw += self.vars.wobble * dt
"#;
    let run = |replay: Option<&solstice_world::Recording>| {
        let (mut game, host) = game_with(&[("wanderer", source)]);
        let wanderer = game
            .world
            .spawn((Transform::default(), Script::new("wanderer")));
        match replay {
            Some(recording) => game.replay(recording),
            None => {
                game.record();
                for frame in 0..400_u32 {
                    let mut input = Input::new();
                    input.set("steer", ((frame as f32) * 0.3).sin());
                    game.frame([0.01, 0.03, 0.016][frame as usize % 3], &input);
                }
            }
        }
        assert!(host.borrow_mut().take_errors().is_empty());
        let transform = *game.world.entities.get::<&Transform>(wanderer).unwrap();
        (
            transform.translation.to_array().map(f32::to_bits),
            game.take_recording(),
        )
    };
    let (live, recording) = run(None);
    let (replayed, _) = run(Some(&recording.unwrap()));
    assert_eq!(live, replayed);
    assert_ne!(live, Vec3::ZERO.to_array().map(f32::to_bits));
}

#[test]
fn the_api_reference_lists_every_engine_function() {
    let reference = api_reference();
    for name in [
        "vec3", "send", "after", "every", "cancel", "find", "input", "held", "pressed", "now",
        "log",
    ] {
        assert!(
            reference.contains(name),
            "{name} missing from:\n{reference}"
        );
    }
}
