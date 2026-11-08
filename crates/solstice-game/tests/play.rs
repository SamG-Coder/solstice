//! A scene's game, from its gameplay files to what is drawn.

use glam::{EulerRot, Mat4, Quat, Vec3};
use solstice_asset::GameFile;
use solstice_game::{Devices, Play, GAME_FILE};
use solstice_script::{Script, ScriptValue};
use solstice_world::{Name, STEP_SECONDS};
use std::collections::HashMap;

fn files(list: &[(&str, &str)]) -> Vec<GameFile> {
    list.iter()
        .map(|(name, text)| GameFile {
            name: (*name).to_owned(),
            text: (*text).to_owned(),
        })
        .collect()
}

/// Where the bake placed each instance: none of them at the actor's own origin.
fn baked(id: u32) -> Option<Mat4> {
    (id < 10).then(|| {
        Mat4::from_rotation_translation(
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            Vec3::new(20.0, 0.0, id as f32),
        ) * Mat4::from_translation(Vec3::new(0.5, 1.0, 0.0))
    })
}

fn var(play: &Play, name: &str, key: &str) -> Option<ScriptValue> {
    let world = &play.world().entities;
    let mut query = world.query::<(&Name, &Script)>();
    let (_, script) = query.iter().find(|(n, _)| n.as_str() == name)?;
    script.vars.get(key).cloned()
}

fn close(a: Mat4, b: Mat4) -> bool {
    a.abs_diff_eq(b, 1.0e-4)
}

const GATE_ACTOR: &str = "name gate\nmesh gate.gltf 1.15\nscript gate.star\nproperty locked bool false\nproperty swing float 1.5\n";
const GATE: &str = r#"
def on_tick(self, dt):
    if pressed("interact") and not self.locked:
        self.yaw += self.swing
"#;

#[test]
fn a_placed_actor_moves_its_instances_when_its_script_moves_it() {
    let scene = files(&[
        (
            GAME_FILE,
            "actor cove/gate.actor 20 0 0 90 swing=0.5 instances=2+2\n",
        ),
        ("cove/gate.actor", GATE_ACTOR),
        ("cove/gate.star", GATE),
    ]);
    let mut play = Play::load(&scene, baked).unwrap();
    assert_eq!(play.instances().collect::<Vec<_>>(), vec![2, 3]);
    assert!(!play.has_player());
    assert_eq!(play.scripts(), ["cove/gate.star"]);

    // Standing still, it is drawn where the bake put it, once, and then left alone.
    let mut placed = HashMap::new();
    let mut devices = Devices::new();
    play.frame(STEP_SECONDS * 1.5, &mut devices, |id, at| {
        placed.insert(id, at);
    });
    assert!(close(placed[&2], baked(2).unwrap()) && close(placed[&3], baked(3).unwrap()));
    let mut again = 0;
    play.frame(STEP_SECONDS, &mut devices, |_, _| again += 1);
    assert_eq!(again, 0, "nothing moved, so nothing is redrawn");

    // Pressed, it turns by its placement's `swing`, about its own origin.
    let mut devices = Devices::new();
    devices.key("KeyE", true);
    // Two whole steps, so the frame ends on a step and nothing is between two.
    play.frame(STEP_SECONDS * 2.0, &mut devices, |id, at| {
        placed.insert(id, at);
    });
    let turned = Mat4::from_rotation_translation(
        Quat::from_rotation_y(90_f32.to_radians() + 0.5),
        Vec3::new(20.0, 0.0, 0.0),
    );
    let origin = Mat4::from_rotation_translation(
        Quat::from_rotation_y(90_f32.to_radians()),
        Vec3::new(20.0, 0.0, 0.0),
    );
    for id in [2, 3] {
        let expected = turned * origin.inverse() * baked(id).unwrap();
        assert!(close(placed[&id], expected), "{id}: {:?}", placed[&id]);
    }
    assert!(play.take_messages().is_empty());
}

#[test]
fn a_placements_settings_override_its_properties() {
    let scene = files(&[
        (
            GAME_FILE,
            "actor gate.actor 0 0 0 0 locked=true\nactor gate.actor 5 0 0 0\n",
        ),
        ("gate.actor", GATE_ACTOR),
        ("gate.star", GATE),
    ]);
    let play = Play::load(&scene, baked).unwrap();
    let world = &play.world().entities;
    let mut locks: Vec<_> = world
        .query::<&Script>()
        .iter()
        .map(|s| s.properties["locked"].clone())
        .collect();
    locks.sort_by_key(|v| format!("{v:?}"));
    assert_eq!(
        locks,
        vec![ScriptValue::Bool(false), ScriptValue::Bool(true)]
    );
}

const RECORDER: &str = r#"
def on_tick(self, dt):
    seen = {}
    for action in ["forward", "turn", "interact", "jump"]:
        if input(action) != 0:
            seen[action] = input(action)
        if pressed(action):
            seen[action + "_pressed"] = True
    self.vars.seen = seen
"#;

/// What the recorder saw of an action.
fn seen(play: &Play, key: &str) -> Option<ScriptValue> {
    match var(play, "recorder", "seen")? {
        ScriptValue::Dict(entries) => entries.into_iter().find(|(k, _)| k == key).map(|(_, v)| v),
        other => panic!("{other:?}"),
    }
}

/// Runs one frame of a game that records each action it sees.
fn recorded(mut devices: Devices) -> Play {
    let scene = files(&[
        (GAME_FILE, "actor recorder.actor 0 0 0 0\n"),
        ("recorder.actor", "name recorder\nscript recorder.star\n"),
        ("recorder.star", RECORDER),
    ]);
    let mut play = Play::load(&scene, baked).unwrap();
    play.frame(STEP_SECONDS * 1.01, &mut devices, |_, _| {});
    let messages = play.take_messages();
    assert!(messages.is_empty(), "{messages:?}");
    play
}

#[test]
fn an_action_reaches_a_script_from_each_kind_of_input() {
    let float = |v: f64| Some(ScriptValue::Float(v));
    let yes = Some(ScriptValue::Bool(true));

    let mut keys = Devices::new();
    keys.key("KeyW", true);
    let play = recorded(keys);
    assert_eq!(seen(&play, "forward"), float(1.0));
    assert_eq!(seen(&play, "forward_pressed"), yes);

    let mut mouse = Devices::new();
    mouse.mouse_button("left", true);
    mouse.mouse_motion(-12.0, 0.0);
    let play = recorded(mouse);
    assert_eq!(seen(&play, "interact_pressed"), yes);
    let turn = match seen(&play, "turn") {
        Some(ScriptValue::Float(t)) => t,
        other => panic!("{other:?}"),
    };
    // 12 pixels left in the frame's time, at the built-in 0.002 radians a pixel.
    let expected = 12.0 / (STEP_SECONDS * 1.01) * 0.002;
    assert!((turn - expected).abs() < 1.0e-4, "{turn} vs {expected}");

    let mut pad = Devices::new();
    pad.gamepad_axis("left_y", -1.0);
    pad.gamepad_button("south", 1.0);
    let play = recorded(pad);
    assert_eq!(seen(&play, "forward"), float(1.0));
    assert_eq!(seen(&play, "jump"), float(1.0));

    let mut touch = Devices::new();
    touch.touch_stick(0.0, -0.5);
    touch.touch_tap();
    let play = recorded(touch);
    assert_eq!(seen(&play, "forward"), float(0.5));
    assert_eq!(seen(&play, "interact_pressed"), yes);
    assert_eq!(
        seen(&play, "interact"),
        None,
        "a tap presses but isn't held"
    );
}

#[test]
fn a_scene_can_bring_its_own_input_actions() {
    let scene = files(&[
        (
            GAME_FILE,
            "input keys.input\nactor recorder.actor 0 0 0 0\n",
        ),
        ("keys.input", "jump key KeyJ\n"),
        ("recorder.actor", "name recorder\nscript recorder.star\n"),
        ("recorder.star", RECORDER),
    ]);
    let mut play = Play::load(&scene, baked).unwrap();
    let mut devices = Devices::new();
    devices.key("KeyJ", true);
    devices.key("KeyW", true);
    play.frame(STEP_SECONDS * 1.01, &mut devices, |_, _| {});
    assert_eq!(seen(&play, "jump"), Some(ScriptValue::Float(1.0)));
    assert_eq!(seen(&play, "forward"), None, "only its own actions");
}

const WALKER: &str = r#"
def on_tick(self, dt):
    self.yaw += input("turn") * dt
    ahead = vec3(0, 0, -1) * input("forward") * dt * 3
    self.position += ahead
"#;

#[test]
fn the_player_camera_follows_the_player_and_looks_up_and_down() {
    let scene = files(&[
        (GAME_FILE, "actor player.actor 1 0 2 0\n"),
        (
            "player.actor",
            "name player\nscript walker.star\ncamera 0 1.6 0\n",
        ),
        ("walker.star", WALKER),
    ]);
    let mut play = Play::load(&scene, baked).unwrap();
    assert!(play.has_player());
    let mut devices = Devices::new();
    devices.key("KeyW", true);
    devices.mouse_motion(0.0, -100.0);
    let mut view = None;
    for _ in 0..60 {
        view = play.frame(STEP_SECONDS, &mut devices, |_, _| {});
    }
    let view = view.unwrap();
    // A second at 3 m/s along -Z, from (1, 0, 2), with the eye 1.6 m up.
    assert!(
        view.position.abs_diff_eq(Vec3::new(1.0, 1.6, -1.0), 0.06),
        "{view:?}"
    );
    assert_eq!(view.yaw, 0.0);
    // The mouse moved up once, by 100 pixels: 0.2 radians up.
    assert!((view.pitch - 0.2).abs() < 1.0e-3, "{view:?}");

    // Turned half round, the eye turns with it.
    let scene = files(&[
        (GAME_FILE, "actor player.actor 0 0 0 180\n"),
        ("player.actor", "name player\ncamera 0.5 1.6 0\n"),
    ]);
    let mut play = Play::load(&scene, baked).unwrap();
    let view = play
        .frame(STEP_SECONDS, &mut Devices::new(), |_, _| {})
        .unwrap();
    assert!(
        view.position.abs_diff_eq(Vec3::new(-0.5, 1.6, 0.0), 1.0e-5),
        "{view:?}"
    );
    let expected = Quat::from_rotation_y(std::f32::consts::PI)
        .to_euler(EulerRot::YXZ)
        .0;
    assert!((view.yaw - expected).abs() < 1.0e-5);
}

#[test]
fn every_mistake_in_a_scenes_gameplay_is_reported_with_its_file() {
    let scene = files(&[
        (
            GAME_FILE,
            "actor gate.actor 0 0 0 0 colour=red locked=maybe\nactor door.actor 0 0 0 0\nactor bad.actor 0 0 0 0\nactor broken.actor 0 0 0 0 instances=8+4\nactor eye.actor 0 0 0 0\nactor eye.actor 0 0 0 0\nactor rock.actor 0 0 0 0 size=2\n",
        ),
        ("gate.actor", GATE_ACTOR),
        ("gate.star", GATE),
        ("bad.actor", "name bad\nscript bad.star\nproperty x\n"),
        ("broken.actor", "name broken\nscript broken.star\n"),
        ("broken.star", "def on_tick(self, dt):\n    x = = 1\n"),
        ("eye.actor", "name eye\ncamera 0 1 0\nscript nowhere.star\n"),
        ("rock.actor", "name rock\n"),
    ]);
    let errors = Play::load(&scene, baked).err().unwrap();
    let expect = [
        "gate.actor has no property `colour`",
        "`locked`: a bool is true or false",
        "actor 2 (door.actor): missing from the scene's gameplay",
        "bad.actor: line 3: expected `property NAME KIND VALUE`",
        "broken.star:2",
        "the scene has no instance 10",
        "its script nowhere.star is missing",
        "only one actor can have a camera",
        "rock.actor has no script, so no properties",
    ];
    for part in expect {
        assert!(
            errors.iter().any(|e| e.contains(part)),
            "{part} not in {errors:#?}"
        );
    }
    // Each script is loaded, and reported, once.
    assert_eq!(
        errors.iter().filter(|e| e.contains("nowhere.star")).count(),
        1
    );

    assert!(Play::load(&[], baked).err().unwrap()[0].contains("has no scene.game"));
    let errors = Play::load(&files(&[(GAME_FILE, "actor x\n")]), baked)
        .err()
        .unwrap();
    assert!(errors[0].starts_with("scene.game: line 1"), "{errors:?}");
    let errors = Play::load(
        &files(&[(GAME_FILE, "input a.input\n"), ("a.input", "jump key\n")]),
        baked,
    )
    .err()
    .unwrap();
    assert!(errors[0].starts_with("a.input: line 1"), "{errors:?}");
    let errors = Play::load(&files(&[(GAME_FILE, "input b.input\n")]), baked)
        .err()
        .unwrap();
    assert!(errors[0].contains("b.input: missing"), "{errors:?}");
}

#[test]
fn scripts_reload_while_playing_and_report_what_they_log() {
    let scene = files(&[
        (GAME_FILE, "actor talker.actor 0 0 0 0\n"),
        ("talker.actor", "name talker\nscript talk.star\n"),
        ("talk.star", "def on_begin(self):\n    log(\"hello\")\n"),
    ]);
    let mut play = Play::load(&scene, baked).unwrap();
    play.frame(STEP_SECONDS * 1.01, &mut Devices::new(), |_, _| {});
    let messages = play.take_messages();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text, "talker: hello");
    assert!(!messages[0].error);

    let error = play.reload("talk.star", "def on_tick(self, dt):\n    fail(\"no\")\n");
    assert!(error.is_ok());
    play.frame(STEP_SECONDS, &mut Devices::new(), |_, _| {});
    let messages = play.take_messages();
    assert!(messages[0].error);
    assert!(
        messages[0].text.starts_with("step 2, talker: on_tick:"),
        "{:?}",
        messages[0]
    );
    assert!(play.reload("talk.star", "def on_tick(self dt):\n").is_err());
}

#[test]
fn the_mouse_turns_as_far_however_fast_the_frames_come() {
    // The same 300 pixels of mouse, spread over a second at 60, 144 and 37 frames a second,
    // turns the player by the same amount: none is lost in frames with no step.
    let turned = |rate: f64| {
        let scene = files(&[
            (GAME_FILE, "actor player.actor 0 0 0 0\n"),
            (
                "player.actor",
                "name player\nscript walker.star\ncamera 0 1.6 0\n",
            ),
            ("walker.star", WALKER),
        ]);
        let mut play = Play::load(&scene, baked).unwrap();
        let mut devices = Devices::new();
        let frames = rate.round() as u32;
        let mut view = None;
        for _ in 0..frames {
            devices.mouse_motion(-300.0 / frames as f32, 30.0 / frames as f32);
            view = play.frame(1.0 / rate, &mut devices, |_, _| {});
        }
        // Let the last movement reach a step.
        view = play.frame(STEP_SECONDS, &mut devices, |_, _| {}).or(view);
        view.unwrap()
    };
    let at_60 = turned(60.0);
    // 300 pixels at 0.002 radians each; 30 down, looking down 0.06.
    assert!((at_60.yaw - 0.6).abs() < 1.0e-3, "{at_60:?}");
    assert!((at_60.pitch + 0.06).abs() < 1.0e-4, "{at_60:?}");
    for rate in [144.0, 37.0] {
        let view = turned(rate);
        assert!((view.yaw - at_60.yaw).abs() < 1.0e-3, "{rate} Hz: {view:?}");
        assert!(
            (view.pitch - at_60.pitch).abs() < 1.0e-4,
            "{rate} Hz: {view:?}"
        );
    }
}

#[test]
fn a_quarter_turn_looks_along_minus_x_as_the_renderers_camera_does() {
    let scene = files(&[
        (GAME_FILE, "actor cove/player.actor 0 0 0 90\n"),
        (
            "cove/player.actor",
            include_str!("../../../scenes/cove/player.actor"),
        ),
        (
            "cove/player.star",
            include_str!("../../../scenes/cove/player.star"),
        ),
    ]);
    let mut play = Play::load(&scene, baked).unwrap();
    let mut devices = Devices::new();
    devices.key("KeyW", true);
    // Two steps' walk, at 1.6 m/s.
    let view = play.frame(STEP_SECONDS * 2.0, &mut devices, |_, _| {});
    let view = view.unwrap();
    // The renderer's camera looks along (-sin yaw, 0, -cos yaw): -X at a quarter turn, and
    // walking forward goes that way.
    assert!(
        (view.yaw - std::f32::consts::FRAC_PI_2).abs() < 1.0e-5,
        "{view:?}"
    );
    // Drawn between the last two steps, and a frame ending just on a step is drawn at the one
    // before it: one step's walk.
    let walked = 1.6 * STEP_SECONDS as f32;
    assert!(
        view.position
            .abs_diff_eq(Vec3::new(-walked, 1.65, 0.0), 1.0e-4),
        "{view:?}"
    );
}

#[test]
fn in_the_cove_the_player_walks_to_the_gate_and_opens_it() {
    // The cove's own actors and scripts, as the bake carries them.
    let scene = files(&[
        (
            GAME_FILE,
            "actor cove/gate.actor 20.7 0 0 90 instances=0+1\nactor cove/player.actor 30 0 0 90\n",
        ),
        (
            "cove/gate.actor",
            include_str!("../../../scenes/cove/gate.actor"),
        ),
        (
            "cove/gate.star",
            include_str!("../../../scenes/cove/gate.star"),
        ),
        (
            "cove/player.actor",
            include_str!("../../../scenes/cove/player.actor"),
        ),
        (
            "cove/player.star",
            include_str!("../../../scenes/cove/player.star"),
        ),
    ]);
    let gate_at = |_: u32| Some(Mat4::from_translation(Vec3::new(20.7, 0.0, 0.0)));
    let mut play = Play::load(&scene, gate_at).unwrap();
    let gate_y = std::cell::Cell::new(0.0);
    let frame = |play: &mut Play, devices: &mut Devices| {
        play.frame(STEP_SECONDS, devices, |_, at| gate_y.set(at.w_axis.y));
    };
    // Too far to reach: pressing E does nothing.
    let mut devices = Devices::new();
    devices.key("KeyE", true);
    frame(&mut play, &mut devices);
    devices.key("KeyE", false);
    for _ in 0..30 {
        frame(&mut play, &mut devices);
    }
    assert_eq!(gate_y.get(), 0.0);
    // Walk towards it for 4 seconds, at 1.6 m/s: from 9.3 m to 2.9 m away.
    devices.key("KeyW", true);
    for _ in 0..240 {
        frame(&mut play, &mut devices);
    }
    devices.key("KeyW", false);
    devices.key("KeyE", true);
    frame(&mut play, &mut devices);
    devices.key("KeyE", false);
    for _ in 0..300 {
        frame(&mut play, &mut devices);
    }
    let messages: Vec<String> = play.take_messages().into_iter().map(|m| m.text).collect();
    assert_eq!(messages, ["gate: The gate rises."]);
    assert!(
        (gate_y.get() - 3.2).abs() < 1.0e-4,
        "the gate rose to {}",
        gate_y.get()
    );
}
