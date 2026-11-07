//! Shared parts of the viewer. The native window and the browser page are thin shells
//! around [`FlyCamera`] and the renderer.

pub mod camera_path;
#[cfg(target_arch = "wasm32")]
mod web;

use glam::Vec3;
use solstice_asset::scene::Viewpoint;
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
    velocity: Vec3,
}

impl FlyCamera {
    pub fn new(camera: Camera) -> Self {
        Self {
            camera,
            speed: 8.0,
            velocity: Vec3::ZERO,
        }
    }

    /// Applies input accumulated over `delta` seconds.
    pub fn update(&mut self, input: &CameraInput, delta: f32) {
        self.camera.yaw += input.look_yaw;
        self.camera.pitch = (self.camera.pitch + input.look_pitch).clamp(-1.5, 1.5);
        let speed = self.speed * if input.fast { 5.0 } else { 1.0 };
        let wanted = (self.camera.forward() * input.forward
            + self.camera.right() * input.right
            + Vec3::Y * input.up)
            * speed;
        // Ease towards the wanted velocity, independent of frame rate. Instant starts and
        // stops read as stutter, whatever the frame rate. Looking is left direct, since any
        // smoothing there would be felt as input lag.
        const RESPONSE_PER_SECOND: f32 = 10.0;
        self.velocity += (wanted - self.velocity) * (1.0 - (-delta * RESPONSE_PER_SECOND).exp());
        self.camera.position += self.velocity * delta;
    }
}

/// Where a scene opens: its own viewpoint, if it has one.
pub fn initial_camera(viewpoint: Option<Viewpoint>) -> Camera {
    let mut camera = Camera::default();
    if let Some(viewpoint) = viewpoint {
        camera.position = Vec3::from_array(viewpoint.position);
        camera.yaw = viewpoint.yaw;
        camera.pitch = viewpoint.pitch;
    }
    camera
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fly(frames: u32, delta: f32, forward: f32, camera: &mut FlyCamera) {
        let input = CameraInput {
            forward,
            ..Default::default()
        };
        for _ in 0..frames {
            camera.update(&input, delta);
        }
    }

    #[test]
    fn reaches_full_speed_and_coasts_to_a_stop() {
        let mut camera = FlyCamera::new(Camera::default());
        fly(1, 1.0 / 60.0, 1.0, &mut camera);
        assert!(
            camera.velocity.length() < camera.speed * 0.3,
            "starts gradually"
        );
        fly(120, 1.0 / 60.0, 1.0, &mut camera);
        assert!(
            (camera.velocity.length() - camera.speed).abs() < 0.01,
            "reaches full speed"
        );
        fly(120, 1.0 / 60.0, 0.0, &mut camera);
        assert!(camera.velocity.length() < 0.01, "stops");
    }

    #[test]
    fn travel_does_not_depend_on_the_frame_rate() {
        let mut fast = FlyCamera::new(Camera::default());
        let mut slow = FlyCamera::new(Camera::default());
        fly(144, 1.0 / 144.0, 1.0, &mut fast);
        fly(30, 1.0 / 30.0, 1.0, &mut slow);
        let difference = fast.camera.position.distance(slow.camera.position);
        assert!(
            difference < 0.5,
            "travelled {difference} m apart in one second"
        );
    }
}
