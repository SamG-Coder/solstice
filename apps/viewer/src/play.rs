//! Play mode in a viewer: a scene's game, driving the renderer's instances and camera.
//!
//! Built only with the `gameplay` feature, so the plain web viewer carries none of it.

use solstice_asset::GameFile;
pub use solstice_game::Message;
use solstice_game::{Devices, Play};
use solstice_render::{Camera, Renderer};

/// A scene's game, running in a viewer.
pub struct PlayMode {
    pub play: Play,
    /// What the input devices are doing, filled in from the window's events.
    pub devices: Devices,
    /// Actions to press in the next frame whatever the devices do, for scripted captures.
    pub presses: Vec<String>,
}

impl PlayMode {
    /// Starts the game in a scene's gameplay files, making every instance an actor draws with
    /// movable. Returns every mistake found in the files, each with its file and line.
    pub fn start(renderer: &mut Renderer, files: &[GameFile]) -> Result<Self, Vec<String>> {
        let play = Play::load(files, |id| renderer.instance_transform(id))?;
        for id in play.instances() {
            renderer.set_instance_movable(id, true);
        }
        Ok(Self {
            play,
            devices: Devices::new(),
            presses: Vec::new(),
        })
    }

    /// Runs a frame `seconds` long: moves the instances that moved, and puts the camera where
    /// the player's eye is, if there is a player. Returns what scripts logged and every error.
    pub fn frame(
        &mut self,
        renderer: &mut Renderer,
        camera: &mut Camera,
        seconds: f32,
    ) -> Vec<Message> {
        let mut input = self.play.actions(&self.devices, seconds);
        for action in self.presses.drain(..) {
            input.press(&action);
        }
        let view = self
            .play
            .frame_with(f64::from(seconds), &input, |id, world| {
                renderer.move_instance(id, world);
            });
        self.devices.end_frame();
        if let Some(view) = view {
            camera.position = view.position;
            camera.yaw = view.yaw;
            camera.pitch = view.pitch;
        }
        self.play.take_messages()
    }
}
