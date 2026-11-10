//! Gamepads, natively: read with `gilrs`, and passed on to the game under the names the
//! browser's standard layout gives a pad's buttons and sticks, so that one set of input
//! actions serves both.
//!
//! As in the browser, the pad used most recently is the one that is read. When it is unplugged,
//! or another takes over, whatever the last one held is let go.

use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};
use solstice_game::Devices;

/// A button's name in the standard layout: `south` is A on an Xbox pad and cross on a
/// PlayStation one, `l2` and `r2` the triggers.
pub fn button_name(button: Button) -> Option<&'static str> {
    Some(match button {
        Button::South => "south",
        Button::East => "east",
        Button::West => "west",
        Button::North => "north",
        Button::LeftTrigger => "l1",
        Button::RightTrigger => "r1",
        Button::LeftTrigger2 => "l2",
        Button::RightTrigger2 => "r2",
        Button::Select => "select",
        Button::Start => "start",
        Button::LeftThumb => "l3",
        Button::RightThumb => "r3",
        Button::DPadUp => "up",
        Button::DPadDown => "down",
        Button::DPadLeft => "left",
        Button::DPadRight => "right",
        _ => return None,
    })
}

/// A stick's axis by its name in the standard layout, and what to multiply its value by: there
/// a stick pulled back reads above zero, where `gilrs` has it below.
pub fn axis_name(axis: Axis) -> Option<(&'static str, f32)> {
    Some(match axis {
        Axis::LeftStickX => ("left_x", 1.0),
        Axis::LeftStickY => ("left_y", -1.0),
        Axis::RightStickX => ("right_x", 1.0),
        Axis::RightStickY => ("right_y", -1.0),
        _ => return None,
    })
}

/// Every name a pad's buttons and sticks go by.
pub const BUTTONS: [&str; 16] = [
    "south", "east", "west", "north", "l1", "r1", "l2", "r2", "select", "start", "l3", "r3", "up",
    "down", "left", "right",
];
pub const AXES: [&str; 4] = ["left_x", "left_y", "right_x", "right_y"];

/// Something a pad did: a button pressed so far, or a stick moved so far.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Did {
    Button(Button, f32),
    Axis(Axis, f32),
}

impl Did {
    fn from_event(event: EventType) -> Option<Self> {
        Some(match event {
            EventType::ButtonChanged(button, value, _) => Self::Button(button, value),
            EventType::ButtonPressed(button, _) => Self::Button(button, 1.0),
            EventType::ButtonReleased(button, _) => Self::Button(button, 0.0),
            EventType::AxisChanged(axis, value, _) => Self::Axis(axis, value),
            _ => return None,
        })
    }

    /// Passes it on to the game.
    pub fn pass(self, devices: &mut Devices) {
        match self {
            Self::Button(button, value) => {
                if let Some(name) = button_name(button) {
                    devices.gamepad_button(name, value);
                }
            }
            // Triggers that a pad gives as axes are buttons here, as they are in the browser.
            Self::Axis(Axis::LeftZ, value) => devices.gamepad_button("l2", value.clamp(0.0, 1.0)),
            Self::Axis(Axis::RightZ, value) => devices.gamepad_button("r2", value.clamp(0.0, 1.0)),
            // And so is a direction pad given as two.
            Self::Axis(Axis::DPadX, value) => {
                devices.gamepad_button("left", (-value).clamp(0.0, 1.0));
                devices.gamepad_button("right", value.clamp(0.0, 1.0));
            }
            Self::Axis(Axis::DPadY, value) => {
                devices.gamepad_button("down", (-value).clamp(0.0, 1.0));
                devices.gamepad_button("up", value.clamp(0.0, 1.0));
            }
            Self::Axis(axis, value) => {
                if let Some((name, sign)) = axis_name(axis) {
                    devices.gamepad_axis(name, value * sign);
                }
            }
        }
    }
}

/// Lets go of everything a pad held.
pub fn release(devices: &mut Devices) {
    for button in BUTTONS {
        devices.gamepad_button(button, 0.0);
    }
    for axis in AXES {
        devices.gamepad_axis(axis, 0.0);
    }
}

/// The gamepads plugged in.
pub struct Pads {
    gilrs: Gilrs,
    /// The pad last heard from, which is the one read.
    active: Option<GamepadId>,
}

impl Pads {
    /// Starts listening for gamepads, if this computer can.
    pub fn new() -> Option<Self> {
        match Gilrs::new() {
            Ok(gilrs) => {
                for (_, pad) in gilrs.gamepads() {
                    log::info!("gamepad: {}", pad.name());
                }
                Some(Self {
                    gilrs,
                    active: None,
                })
            }
            Err(error) => {
                log::warn!("no gamepads: {error}");
                None
            }
        }
    }

    /// Passes on what the pads have done since this was last called.
    pub fn poll(&mut self, devices: &mut Devices) {
        while let Some(event) = self.gilrs.next_event() {
            match event.event {
                EventType::Connected => {
                    log::info!("gamepad: {}", self.gilrs.gamepad(event.id).name());
                }
                EventType::Disconnected => {
                    if self.active == Some(event.id) {
                        release(devices);
                        self.active = None;
                    }
                }
                other => {
                    let Some(did) = Did::from_event(other) else {
                        continue;
                    };
                    // Another pad taking over: the one before lets go of what it held.
                    if self.active.is_some_and(|active| active != event.id) {
                        release(devices);
                    }
                    self.active = Some(event.id);
                    did.pass(devices);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use solstice_game::InputMap;

    /// What an action reads after a pad has done some things, with the built-in actions.
    fn after(done: &[Did], action: &str) -> f32 {
        let mut devices = Devices::new();
        for did in done {
            did.pass(&mut devices);
        }
        InputMap::default()
            .actions(&devices, 1.0 / 60.0)
            .value(action)
    }

    #[test]
    fn every_name_the_browser_uses_has_a_button_or_an_axis() {
        let buttons = [
            Button::South,
            Button::East,
            Button::West,
            Button::North,
            Button::LeftTrigger,
            Button::RightTrigger,
            Button::LeftTrigger2,
            Button::RightTrigger2,
            Button::Select,
            Button::Start,
            Button::LeftThumb,
            Button::RightThumb,
            Button::DPadUp,
            Button::DPadDown,
            Button::DPadLeft,
            Button::DPadRight,
        ];
        let named: Vec<&str> = buttons.iter().filter_map(|&b| button_name(b)).collect();
        assert_eq!(named, BUTTONS);
        let axes = [
            Axis::LeftStickX,
            Axis::LeftStickY,
            Axis::RightStickX,
            Axis::RightStickY,
        ];
        let named: Vec<&str> = axes
            .iter()
            .filter_map(|&a| axis_name(a))
            .map(|a| a.0)
            .collect();
        assert_eq!(named, AXES);
        assert_eq!(button_name(Button::Mode), None);
    }

    #[test]
    fn the_left_stick_walks_and_the_buttons_do_what_the_keys_do() {
        // Pushed away is forwards. It is above zero to gilrs and below it in the browser's
        // standard layout, which is what the actions are written for.
        assert_eq!(after(&[Did::Axis(Axis::LeftStickY, 1.0)], "forward"), 1.0);
        assert_eq!(after(&[Did::Axis(Axis::LeftStickY, -1.0)], "forward"), -1.0);
        // To the right is above zero; a little off centre is within the stick's dead zone.
        assert_eq!(after(&[Did::Axis(Axis::LeftStickX, 1.0)], "right"), 1.0);
        assert!(after(&[Did::Axis(Axis::LeftStickX, -0.5)], "right") < -0.3);
        assert_eq!(after(&[Did::Axis(Axis::LeftStickX, 0.05)], "right"), 0.0);
        assert_eq!(after(&[Did::Button(Button::South, 1.0)], "jump"), 1.0);
        assert_eq!(after(&[Did::Button(Button::West, 1.0)], "interact"), 1.0);
        assert_eq!(after(&[Did::Button(Button::LeftThumb, 1.0)], "run"), 1.0);
        // The right stick turns: to the right is a turn the other way round +Y.
        assert!(after(&[Did::Axis(Axis::RightStickX, 1.0)], "turn") < 0.0);
    }

    #[test]
    fn the_triggers_are_the_pedals_and_the_left_stick_steers() {
        let throttle = Did::Button(Button::RightTrigger2, 0.6);
        assert!((after(&[throttle], "throttle") - 0.6).abs() < 1.0e-6);
        assert_eq!(
            after(&[Did::Button(Button::LeftTrigger2, 1.0)], "brake"),
            1.0
        );
        assert_eq!(after(&[Did::Axis(Axis::LeftStickX, 1.0)], "steer"), 1.0);
        assert!(after(&[Did::Axis(Axis::LeftStickX, -0.5)], "steer") < -0.3);
        // A trigger that comes as an axis is the same pedal.
        assert!((after(&[Did::Axis(Axis::RightZ, 0.8)], "throttle") - 0.8).abs() < 1.0e-6);
        // The bumpers change gear, and the right stick looks round the car.
        assert_eq!(
            after(&[Did::Button(Button::RightTrigger, 1.0)], "shift_up"),
            1.0
        );
        assert_eq!(
            after(&[Did::Button(Button::LeftTrigger, 1.0)], "shift_down"),
            1.0
        );
        assert_eq!(after(&[Did::Button(Button::North, 1.0)], "reset"), 1.0);
        assert_eq!(
            after(&[Did::Axis(Axis::RightStickX, 1.0)], "look_round"),
            1.0
        );
        // Pulled back, which is below zero to gilrs, tips the camera up over the car.
        assert_eq!(
            after(&[Did::Axis(Axis::RightStickY, -1.0)], "look_over"),
            1.0
        );
    }

    #[test]
    fn a_trigger_is_a_button_however_the_pad_gives_it() {
        let map = InputMap::parse("fire gamepad r2\naim gamepad l2\nback gamepad down\n").unwrap();
        let read = |did: Did, action: &str| {
            let mut devices = Devices::new();
            did.pass(&mut devices);
            map.actions(&devices, 1.0 / 60.0).value(action)
        };
        assert!((read(Did::Button(Button::RightTrigger2, 0.6), "fire") - 0.6).abs() < 1.0e-6);
        // Some pads give their triggers as axes, and their direction pad as two.
        assert!((read(Did::Axis(Axis::RightZ, 0.8), "fire") - 0.8).abs() < 1.0e-6);
        assert_eq!(read(Did::Axis(Axis::LeftZ, 1.0), "aim"), 1.0);
        assert_eq!(read(Did::Axis(Axis::DPadY, -1.0), "back"), 1.0);
        assert_eq!(read(Did::Axis(Axis::DPadY, 1.0), "back"), 0.0);
    }

    #[test]
    fn letting_go_of_a_pad_lets_go_of_everything() {
        let mut devices = Devices::new();
        Did::Button(Button::South, 1.0).pass(&mut devices);
        Did::Axis(Axis::LeftStickX, 1.0).pass(&mut devices);
        release(&mut devices);
        let input = InputMap::default().actions(&devices, 1.0 / 60.0);
        assert_eq!((input.value("jump"), input.value("right")), (0.0, 0.0));
    }
}
