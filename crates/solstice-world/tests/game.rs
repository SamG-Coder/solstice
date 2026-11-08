//! The world run as a game: tick order, events, timers, deferred changes, and exact replay.

use glam::Vec3;
use solstice_world::{
    Entity, Game, GlobalTransform, Input, Parent, Schedule, TickGroup, Transform, World,
    STEP_SECONDS,
};
use std::cell::RefCell;
use std::rc::Rc;

type Log = Rc<RefCell<Vec<String>>>;

/// A game whose systems write their group's name, and the step, to a log.
fn logging_game() -> (Game, Log) {
    let log: Log = Rc::default();
    let mut schedule = Schedule::new();
    for group in TickGroup::ALL {
        for n in 1..=2 {
            let log = log.clone();
            schedule.add(group, "log", move |world: &mut World| {
                log.borrow_mut()
                    .push(format!("{}:{group:?}{n}", world.step()));
            });
        }
    }
    (Game::new(World::new(), schedule), log)
}

#[test]
fn groups_run_in_order_each_step_and_each_frame() {
    let (mut game, log) = logging_game();
    let steps = game.frame(STEP_SECONDS * 2.0 + 1.0e-4, &Input::new());
    assert_eq!(steps.steps, 2);
    let expected: Vec<String> = [
        "1:PrePhysics1",
        "1:PrePhysics2",
        "1:Physics1",
        "1:Physics2",
        "1:PostPhysics1",
        "1:PostPhysics2",
        "2:PrePhysics1",
        "2:PrePhysics2",
        "2:Physics1",
        "2:Physics2",
        "2:PostPhysics1",
        "2:PostPhysics2",
        "2:Animation1",
        "2:Animation2",
        "2:Late1",
        "2:Late2",
    ]
    .map(String::from)
    .to_vec();
    assert_eq!(*log.borrow(), expected);

    // A frame too short for a step still runs the per-frame groups.
    log.borrow_mut().clear();
    assert_eq!(game.frame(STEP_SECONDS * 0.1, &Input::new()).steps, 0);
    assert_eq!(
        *log.borrow(),
        ["2:Animation1", "2:Animation2", "2:Late1", "2:Late2"]
            .map(String::from)
            .to_vec()
    );
}

#[derive(Clone, Debug, PartialEq)]
struct Hit(u32);
#[derive(Clone, Debug, PartialEq)]
struct Bell(u64);

#[test]
fn events_reach_the_next_group_and_timers_the_step_they_fire_on() {
    let seen: Log = Rc::default();
    let mut schedule = Schedule::new();
    for group in TickGroup::ALL {
        let seen = seen.clone();
        schedule.add(group, "read", move |world: &mut World| {
            for hit in world.read::<Hit>() {
                seen.borrow_mut()
                    .push(format!("{}:{group:?} hit {}", world.step(), hit.0));
            }
            for bell in world.read::<Bell>() {
                seen.borrow_mut()
                    .push(format!("{}:{group:?} bell {}", world.step(), bell.0));
            }
        });
    }
    schedule.add(TickGroup::PrePhysics, "send", |world: &mut World| {
        if world.step() == 1 {
            world.send(Hit(1));
            world.send(Hit(2));
            // Three steps on: step 4.
            world.after(STEP_SECONDS * 3.0, Bell(4));
        }
    });
    schedule.add(TickGroup::PostPhysics, "send late", |world: &mut World| {
        if world.step() == 3 {
            world.send(Hit(3));
        }
    });
    let mut game = Game::new(World::new(), schedule);
    for _ in 0..5 {
        game.step(&Input::new());
    }
    assert_eq!(
        *seen.borrow(),
        [
            "1:Physics hit 1",
            "1:Physics hit 2",
            // Sent after physics, read as the next step starts, before the timer due then.
            "4:PrePhysics hit 3",
            "4:PrePhysics bell 4",
        ]
        .map(String::from)
        .to_vec()
    );
}

#[test]
fn the_per_frame_groups_see_what_the_last_step_delivered() {
    let seen = Rc::new(RefCell::new(0));
    let mut schedule = Schedule::new();
    schedule.add(TickGroup::PostPhysics, "send", |world: &mut World| {
        world.send(Hit(7))
    });
    let counter = seen.clone();
    schedule.add(TickGroup::Late, "read", move |world: &mut World| {
        *counter.borrow_mut() += world.read::<Hit>().len();
    });
    let mut game = Game::new(World::new(), schedule);
    game.frame(STEP_SECONDS * 1.01, &Input::new());
    assert_eq!(*seen.borrow(), 1);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "per-frame tick groups don't send events")]
fn the_per_frame_groups_cannot_send_events() {
    let mut schedule = Schedule::new();
    schedule.add(TickGroup::Animation, "send", |world: &mut World| {
        world.send(Hit(1))
    });
    Game::new(World::new(), schedule).frame(0.0, &Input::new());
}

#[test]
fn spawning_during_a_group_happens_when_it_ends() {
    #[derive(Debug)]
    struct Spark;
    let counts: Rc<RefCell<Vec<(&'static str, usize)>>> = Rc::default();
    let mut schedule = Schedule::new();
    let count = |name: &'static str, counts: &Rc<RefCell<Vec<(&'static str, usize)>>>| {
        let counts = counts.clone();
        move |world: &mut World| {
            let sparks = world.entities.query::<&Spark>().iter().count();
            counts.borrow_mut().push((name, sparks));
        }
    };
    schedule.add(TickGroup::PrePhysics, "spawn", |world: &mut World| {
        world.commands().spawn((Spark,));
    });
    schedule.add(TickGroup::PrePhysics, "count", count("same group", &counts));
    schedule.add(TickGroup::Physics, "count", count("next group", &counts));
    let mut game = Game::new(World::new(), schedule);
    game.step(&Input::new());
    assert_eq!(*counts.borrow(), [("same group", 0), ("next group", 1)]);
}

#[test]
fn a_slow_frame_runs_three_steps_and_the_game_falls_behind() {
    let (mut game, _) = logging_game();
    let steps = game.frame(0.5, &Input::new());
    assert_eq!(steps.steps, 3);
    assert!(steps.slowed);
    assert_eq!(game.world.step(), 3);
}

#[test]
fn transforms_are_placed_after_every_step_with_the_step_before_kept() {
    let mut world = World::new();
    let cart = world.spawn((Transform::default(),));
    let rider = world.spawn((Transform::from_translation(Vec3::Y), Parent(cart)));
    let mut schedule = Schedule::new();
    schedule.add(TickGroup::PrePhysics, "roll", move |world: &mut World| {
        world
            .entities
            .get::<&mut Transform>(cart)
            .unwrap()
            .translation
            .x += 1.0;
    });
    let mut game = Game::new(world, schedule);
    let frame = game.frame(STEP_SECONDS * 2.5, &Input::new());
    assert_eq!(frame.steps, 2);
    let global = *game.world.entities.get::<&GlobalTransform>(rider).unwrap();
    assert_eq!(Vec3::from(global.now.translation), Vec3::new(2.0, 1.0, 0.0));
    assert_eq!(
        Vec3::from(global.before.translation),
        Vec3::new(1.0, 1.0, 0.0)
    );
    let drawn = global.interpolated(frame.alpha).w_axis.truncate();
    assert!(
        drawn.abs_diff_eq(Vec3::new(1.5, 1.0, 0.0), 1.0e-3),
        "{drawn}"
    );
}

// --- Replay -------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Velocity(Vec3);
struct Player;
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shot;
#[derive(Clone, Debug, PartialEq)]
struct Fire;
#[derive(Clone, Debug, PartialEq)]
struct Gone;
#[derive(Default)]
struct Score(u32);

/// A small game with input, timers, spawning, despawning and events: a player steers with
/// "turn" and "forward"; a timer fires a shot from the player every half second while "fire" is
/// held; shots fly and disappear beyond 20 m, and each that does adds to a score.
fn shooting_game() -> (Game, Entity) {
    let mut world = World::new();
    let player = world.spawn((Player, Transform::default(), Velocity(Vec3::ZERO)));
    world.spawn((Score::default(),));
    world.every(0.5, Fire);
    let mut schedule = Schedule::new();
    schedule.add(TickGroup::PrePhysics, "steer", move |world: &mut World| {
        let (turn, forward) = (world.input().value("turn"), world.input().value("forward"));
        let (transform, velocity) = world
            .entities
            .query_one_mut::<(&mut Transform, &mut Velocity)>(player)
            .unwrap();
        transform.rotation *= glam::Quat::from_rotation_y(turn * 0.05);
        velocity.0 = transform.rotation * Vec3::NEG_Z * forward * 4.0;
    });
    schedule.add(TickGroup::PrePhysics, "fire", move |world: &mut World| {
        if world.read::<Fire>().is_empty() || !world.input().held("fire") {
            return;
        }
        let from = *world.entities.get::<&Transform>(player).unwrap();
        let velocity = from.rotation * Vec3::NEG_Z * 15.0;
        world.commands().spawn((Shot, from, Velocity(velocity)));
    });
    schedule.add(TickGroup::Physics, "move", |world: &mut World| {
        for (transform, velocity) in world.entities.query_mut::<(&mut Transform, &Velocity)>() {
            transform.translation += velocity.0 * STEP_SECONDS as f32;
        }
    });
    schedule.add(
        TickGroup::PostPhysics,
        "expire",
        move |world: &mut World| {
            let origin = world
                .entities
                .get::<&Transform>(player)
                .unwrap()
                .translation;
            let far: Vec<Entity> = world
                .entities
                .query::<(Entity, &Transform, &Shot)>()
                .iter()
                .filter(|(_, transform, _)| transform.translation.distance(origin) > 20.0)
                .map(|(entity, _, _)| entity)
                .collect();
            for entity in far {
                world.commands().despawn(entity);
                world.send(Gone);
            }
        },
    );
    schedule.add(TickGroup::PrePhysics, "score", |world: &mut World| {
        let gone = world.read::<Gone>().len() as u32;
        for score in world.entities.query_mut::<&mut Score>() {
            score.0 += gone;
        }
    });
    (Game::new(world, schedule), player)
}

/// Everything that matters in the shooting game, exactly: the score, and every entity's place,
/// as bits, sorted.
fn snapshot(game: &Game) -> (u32, Vec<[u32; 3]>) {
    let score = game
        .world
        .entities
        .query::<&Score>()
        .iter()
        .next()
        .unwrap()
        .0;
    let mut places: Vec<[u32; 3]> = game
        .world
        .entities
        .query::<&Transform>()
        .iter()
        .map(|t| t.translation.to_array().map(f32::to_bits))
        .collect();
    places.sort_unstable();
    (score, places)
}

/// Input that changes as a run goes on.
fn input_at(frame: u32) -> Input {
    let mut input = Input::new();
    input.set("forward", if frame % 200 < 120 { 1.0 } else { 0.3 });
    input.set("turn", ((frame as f32) * 0.05).sin());
    if frame % 90 < 60 {
        input.set("fire", 1.0);
    }
    input
}

#[test]
fn a_recorded_run_replays_exactly_whatever_its_frames_were() {
    let (mut game, _) = shooting_game();
    game.record();
    // Uneven frames, including slow ones that run several steps and short ones that run none.
    for frame in 0..900_u32 {
        let seconds = [0.004, 0.016, 0.031, 0.009, 0.07][frame as usize % 5];
        game.frame(seconds, &input_at(frame));
    }
    let recording = game.take_recording().unwrap();
    let original = snapshot(&game);
    assert!(original.0 > 5, "the run should have scored: {}", original.0);
    assert_eq!(recording.len(), game.world.step());

    let (mut replayed, _) = shooting_game();
    replayed.replay(&recording);
    assert_eq!(snapshot(&replayed), original);
    assert_eq!(replayed.world.step(), game.world.step());
}

#[test]
fn each_step_sees_the_input_of_the_frame_it_runs_in() {
    let seen: Rc<RefCell<Vec<f32>>> = Rc::default();
    let mut schedule = Schedule::new();
    let log = seen.clone();
    schedule.add(TickGroup::Physics, "read", move |world: &mut World| {
        log.borrow_mut().push(world.input().value("forward"));
    });
    let mut game = Game::new(World::new(), schedule);
    let mut input = Input::new();
    for value in [0.25, 0.5, 1.0] {
        input.set("forward", value);
        game.frame(STEP_SECONDS * 2.0 + 1.0e-4, &input);
    }
    assert_eq!(*seen.borrow(), [0.25, 0.25, 0.5, 0.5, 1.0, 1.0]);
}
