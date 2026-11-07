//! Shared parts of the viewer. The native window and the browser page are thin shells
//! around [`FlyCamera`] and the renderer.

#[cfg(target_arch = "wasm32")]
mod web;

use glam::Vec3;
use solstice_render::{Camera, Environment};

/// Keyboard and mouse state, in a form both front ends can fill in.
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraInput {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
    pub fast: bool,
}

/// A free-flying first-person camera.
#[derive(Clone, Copy, Debug)]
pub struct FlyCamera {
    pub camera: Camera,
    pub speed: f32,
}

impl FlyCamera {
    pub fn new(camera: Camera) -> Self {
        Self { camera, speed: 8.0 }
    }

    /// Applies input accumulated over `delta` seconds.
    pub fn update(&mut self, input: &CameraInput, delta: f32) {
        self.camera.yaw += input.look_yaw;
        self.camera.pitch = (self.camera.pitch + input.look_pitch).clamp(-1.5, 1.5);
        let speed = self.speed * if input.fast { 5.0 } else { 1.0 } * delta;
        self.camera.position += self.camera.forward() * input.forward * speed;
        self.camera.position += self.camera.right() * input.right * speed;
        self.camera.position += Vec3::Y * input.up * speed;
    }
}

/// Where the demo scene opens.
pub fn default_camera() -> Camera {
    Camera {
        position: Vec3::new(20.0, 4.2, 34.0),
        yaw: 0.6,
        pitch: -0.04,
        ..Default::default()
    }
}

/// Late-afternoon sun, low enough for long shadows.
pub fn default_environment() -> Environment {
    Environment {
        sun_direction: Vec3::new(0.62, 0.42, 0.5).normalize(),
        sun_illuminance: 14.0,
        // Above the physical ratio: like a sky light in a game engine, this holds shadow
        // detail that a real camera would need several stops of range to capture.
        sky_intensity: 3.0,
        gi_intensity: 0.85,
        ..Default::default()
    }
}
