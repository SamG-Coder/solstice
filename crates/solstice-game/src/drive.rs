//! Driving: the player's car, what is drawn of it, the camera that follows it, and its
//! instruments.
//!
//! * The input actions `throttle`, `brake`, `steer` (above zero to the right), `handbrake`,
//!   `shift_up` and `shift_down` drive the player's car. `reset` puts it back on the track, and
//!   `camera` changes the view; `look_back` looks behind while held.
//! * A car's wheels are instances of its mesh, found by where they are: an instance whose own
//!   origin is at one of the hubs its `.vehicle` file gives is that wheel, and is drawn where
//!   the suspension has it, steered and rolled. One whose origin is a few centimetres from a
//!   hub is on that wheel's upright, as a brake caliper is: it rides and steers with the wheel,
//!   and does not turn with it.

use crate::race::Lap;
use crate::track::Track;
use glam::{EulerRot, Mat4, Quat, Vec3};
use solstice_physics::vehicle::{Asked, Telemetry, Vehicle};
use solstice_physics::Car;
use solstice_world::{Entity, Input, Transform, World, STEP_SECONDS};

/// How near a hub an instance's origin must be, in metres, to be carried by that wheel.
pub const HUB_REACH: f32 = 0.06;
/// How near it must be to turn with the wheel. Further out than this, and within
/// [`HUB_REACH`], it is on the upright: it rides and steers, and stays the right way up.
pub const SPIN_REACH: f32 = 0.012;

/// One wheel, as it is drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DrawnWheel {
    /// Where its middle is, from the car's origin.
    pub position: Vec3,
    /// How far it is steered, in radians, above zero to the right.
    pub steer: f32,
    /// How far it has rolled, in radians, and how fast it is rolling, in radians a second.
    pub angle: f32,
    pub speed: f32,
}

impl DrawnWheel {
    /// Where the wheel is drawn, relative to the car, `alpha` of the way from the step before
    /// to the latest.
    pub fn between(before: &Self, now: &Self, alpha: f32) -> Mat4 {
        let position = before.position.lerp(now.position, alpha);
        let steer = before.steer + (now.steer - before.steer) * alpha;
        // Rolled back from where it is now by how fast it is rolling: a wheel turns more than
        // half way round in a step at speed, and the short way between two angles is then the
        // wrong way.
        let angle = now.angle - now.speed * STEP_SECONDS as f32 * (1.0 - alpha);
        Mat4::from_rotation_translation(
            Quat::from_rotation_y(-steer) * Quat::from_rotation_x(-angle),
            position,
        )
    }

    /// Where the wheel's upright is drawn: where the wheel is, steered, and not rolled.
    pub fn upright(before: &Self, now: &Self, alpha: f32) -> Mat4 {
        let position = before.position.lerp(now.position, alpha);
        let steer = before.steer + (now.steer - before.steer) * alpha;
        Mat4::from_rotation_translation(Quat::from_rotation_y(-steer), position)
    }
}

/// A car's wheels as of the latest step and the one before, for drawing between them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wheels {
    pub now: [DrawnWheel; 4],
    pub before: [DrawnWheel; 4],
}

impl Wheels {
    /// Takes the wheels as the vehicle has them after a step.
    pub fn follow(&mut self, vehicle: &Vehicle) {
        self.before = self.now;
        for (index, drawn) in self.now.iter_mut().enumerate() {
            let (position, _) = vehicle.wheel_pose(index);
            let wheel = &vehicle.wheels[index];
            *drawn = DrawnWheel {
                position,
                steer: wheel.steer,
                angle: wheel.angle,
                speed: wheel.speed,
            };
        }
    }

    /// The wheels of a car standing still, both steps the same.
    pub fn at_rest(vehicle: &Vehicle) -> Self {
        let mut wheels = Self::default();
        wheels.follow(vehicle);
        wheels.before = wheels.now;
        wheels
    }
}

/// What the player asks of the car this step, from the input actions.
pub fn asked(input: &Input) -> Asked {
    Asked {
        throttle: input.value("throttle"),
        brake: input.value("brake"),
        steer: input.value("steer"),
        handbrake: input.value("handbrake"),
        shift_up: input.pressed("shift_up"),
        shift_down: input.pressed("shift_down"),
    }
}

/// How hard the car that drives itself corners and brakes, in metres a second squared: within
/// what a racing car on slicks does, with room to spare.
const SELF_CORNERING: f32 = 13.0;
const SELF_BRAKING: f32 = 12.0;
const SELF_TOP_SPEED: f32 = 80.0;

/// What a car driving itself round a track asks for, for showing a circuit off and for trying
/// one: it steers for the middle of the road a way ahead, further the faster it goes, and
/// keeps to a speed the bends coming can be taken at, braking for them in good time. `forward`
/// is the way the car points and `speed` how fast it goes along that. Gives back the sample
/// of the track it is at too, to find it by next time.
pub fn autopilot(
    track: &Track,
    position: Vec3,
    forward: Vec3,
    speed: f32,
    hint: Option<usize>,
) -> (Asked, usize) {
    let place = track.locate(position, hint);
    let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let (target, _, _) = track.place(place.along + 7.0 + speed.abs() * 0.55);
    let to = flat(target - position).normalize_or_zero();
    // Above zero when the target is to the right, which is the way steering above zero turns.
    let turn = -flat(forward).normalize_or_zero().cross(to).y;

    // How fast each stretch of road ahead can be taken, and so how fast the car may be going
    // now and still slow for it in the distance there is.
    let mut limit = SELF_TOP_SPEED;
    let mut ahead = 0.0_f32;
    while ahead < 260.0 {
        let at = |d: f32| flat(track.place(place.along + ahead + d).0);
        let (a, b, c) = (at(-6.0), at(0.0), at(6.0));
        let bend = 2.0 * (b - a).cross(c - b).length()
            / ((b - a).length() * (c - b).length() * (c - a).length()).max(1.0e-6);
        let through = (SELF_CORNERING / bend.max(1.0e-4)).sqrt();
        let room = (ahead - 10.0).max(0.0);
        limit = limit.min((through * through + 2.0 * SELF_BRAKING * room).sqrt());
        ahead += 8.0;
    }
    let asked = Asked {
        throttle: ((limit - speed) * 0.6).clamp(0.0, 1.0),
        brake: ((speed - limit) * 0.35).clamp(0.0, 1.0),
        steer: (turn * 3.2).clamp(-1.0, 1.0),
        ..Asked::default()
    };
    (asked, place.sample)
}

/// How far below the nearest stretch of its track a car is before it counts as lost, in
/// metres, and is put back.
pub const FALLEN: f32 = 14.0;

/// Puts a car back on the track: on the centre line nearest where it is, a little back, facing
/// the way the lap runs, at rest.
pub fn put_back(world: &mut World, entity: Entity, track: &Track) {
    let Ok(mut transform) = world.entities.get::<&mut Transform>(entity) else {
        return;
    };
    let place = track.locate(transform.translation, None);
    let (centre, forward, left) = track.place(place.along - 8.0);
    let up = forward.cross(left).normalize_or(Vec3::Y);
    // The car's -Z along the track, its +X to the track's right, and standing on its slope.
    transform.rotation = Quat::from_mat3(&glam::Mat3::from_cols(-left, up, -forward)).normalize();
    transform.translation = centre + up * 0.05;
    drop(transform);
    if let Ok(mut car) = world.entities.get::<&mut Car>(entity) {
        car.vehicle.settle();
        car.asked = Asked::default();
    }
    if let Ok(mut lap) = world.entities.get::<&mut Lap>(entity) {
        lap.interrupt();
    }
}

/// The views of a car the `camera` action steps through: from close behind it out to high
/// above, then from on the car itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CameraView {
    /// Behind and above, following the car round.
    #[default]
    Chase,
    /// The same, from further back and higher.
    Far,
    /// Further again: the car and the road round it.
    Wide,
    /// From high above and well behind: the corner coming, and the one after.
    High,
    /// From the bonnet, looking down the road.
    Bonnet,
    /// From the nose, low, with nothing of the car in view.
    Bumper,
}

impl CameraView {
    pub fn next(self) -> Self {
        match self {
            Self::Chase => Self::Far,
            Self::Far => Self::Wide,
            Self::Wide => Self::High,
            Self::High => Self::Bonnet,
            Self::Bonnet => Self::Bumper,
            Self::Bumper => Self::Chase,
        }
    }

    /// For a view that follows the car: how far behind it the camera is, how far above, and
    /// how many seconds it takes to come round behind the car when the car turns.
    fn following(self) -> Option<(f32, f32, f32)> {
        match self {
            Self::Chase => Some((5.2, 1.5, 0.11)),
            Self::Far => Some((7.8, 2.3, 0.16)),
            Self::Wide => Some((12.5, 4.0, 0.2)),
            Self::High => Some((22.0, 11.0, 0.26)),
            Self::Bonnet | Self::Bumper => None,
        }
    }
}

/// Where the player is looking, besides the way the car is going.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Look {
    /// Behind: what the mirror shows.
    pub back: bool,
    /// Round the car: from -1, to the left, to 1, to the right. A stick held right over looks
    /// all but behind.
    pub around: f32,
    /// Over it: at 1 the camera is tipped up high above the car, at -1 down low behind it.
    pub over: f32,
}

/// How far round the car a stick held right over looks, in radians, and how far up and down
/// one pulled right back or pushed right away tips the camera.
const LOOK_AROUND: f32 = 2.9;
const LOOK_OVER: f32 = 0.55;
const LOOK_UNDER: f32 = 0.2;
/// How long, in seconds, the camera takes to follow the stick.
const LOOK_EASE: f32 = 0.09;

/// The camera that follows a car.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Chase {
    pub view: CameraView,
    /// The way the camera looks round the car, eased behind the car's own heading: radians
    /// about +Y, zero looking along -Z.
    yaw: f32,
    /// The height it follows the car at, eased, so that kerbs don't shake the picture.
    height: f32,
    /// How far round the car the player is looking, and how far over it, eased: radians.
    around: f32,
    over: f32,
    started: bool,
}

/// Where a camera is, and the way it looks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// The angle from `from` to `to`, the short way round.
fn turn(from: f32, to: f32) -> f32 {
    (to - from + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

impl Chase {
    /// The camera for a frame `seconds` long, for a car drawn at `car`, going at `speed` metres
    /// a second, with the player looking as `look` says.
    pub fn frame(&mut self, car: Mat4, speed: f32, look: Look, seconds: f32) -> Shot {
        let (_, rotation, origin) = car.to_scale_rotation_translation();
        let forward = rotation * Vec3::NEG_Z;
        let heading = (-forward.x).atan2(-forward.z);
        let (yaw, pitch, _) = rotation.to_euler(EulerRot::YXZ);
        let behind = if look.back { std::f32::consts::PI } else { 0.0 };
        // The stick, eased: a little of it looks a little way, and more of it a lot more.
        let asked = look.around.clamp(-1.0, 1.0);
        let around = asked.abs().powf(1.4) * asked.signum() * LOOK_AROUND;
        let tipped = look.over.clamp(-1.0, 1.0);
        let over = tipped * if tipped > 0.0 { LOOK_OVER } else { LOOK_UNDER };
        let follow = 1.0 - (-seconds / LOOK_EASE).exp();
        self.around += (around - self.around) * follow;
        self.over += (over - self.over) * follow;
        let Some((back, up, ease)) = self.view.following() else {
            self.started = false;
            let eye = if self.view == CameraView::Bonnet {
                Vec3::new(0.0, 1.02, -0.45)
            } else {
                Vec3::new(0.0, 0.52, -2.05)
            };
            // From on the car, the stick turns the driver's head: to the side, and no further.
            let head = self.around.clamp(-1.9, 1.9);
            return Shot {
                position: origin + rotation * eye,
                yaw: yaw + behind - head,
                pitch: if look.back { -pitch } else { pitch } - self.over * 0.5,
            };
        };
        if !self.started {
            self.yaw = heading;
            self.height = origin.y;
            self.started = true;
        }
        // Eased round behind the car, so that a slide shows as the car turning in the
        // picture; and a little further back the faster it goes.
        let follow = 1.0 - (-seconds / ease).exp();
        self.yaw += turn(self.yaw, heading) * follow;
        self.height += (origin.y - self.height) * (1.0 - (-seconds / 0.09).exp());
        // Looking to the right is from over the car's left shoulder.
        let round = self.yaw + behind - self.around;
        let away = Vec3::new(round.sin(), 0.0, round.cos());
        let back = back + (speed.abs() / 85.0).min(1.0) * 0.9;
        // Tipped up over the car, or down behind it: the camera's arm swings about the car.
        let (sin, cos) = self.over.sin_cos();
        let (back, up) = (back * cos - up * sin, (up * cos + back * sin).max(0.35));
        let position = Vec3::new(origin.x, self.height, origin.z) + away * back + Vec3::Y * up;
        // Looking at a point above the car and a little ahead of it.
        let target = origin + Vec3::Y * 0.85 - away * 1.2;
        let to = target - position;
        Shot {
            position,
            yaw: (-to.x).atan2(-to.z),
            pitch: to.y.atan2(to.x.hypot(to.z)),
        }
    }
}

/// The player's car's instruments, for a viewer to show.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hud {
    pub telemetry: Telemetry,
    /// Where the engine's limiter is, in rpm.
    pub limit: f32,
    /// The pedals and wheel as they are after easing: from 0 to 1, and from -1 to 1.
    pub throttle: f32,
    pub brake: f32,
    pub steer: f32,
    pub lap: Option<Lap>,
    pub track: String,
    pub view: CameraView,
}

impl Hud {
    /// One line of it, for a window's title or a log.
    pub fn line(&self) -> String {
        use crate::race::format_time;
        let t = &self.telemetry;
        let gear = match t.gear {
            -1 => "R".to_owned(),
            0 => "N".to_owned(),
            gear => gear.to_string(),
        };
        let mut line = format!(
            "{:3.0} km/h   {gear}   {:4.0} rpm",
            t.speed.abs() * 3.6,
            t.rpm
        );
        if t.abs_active {
            line.push_str("   ABS");
        }
        if t.traction_active {
            line.push_str("   TC");
        }
        if t.stability_active {
            line.push_str("   ESC");
        }
        if let Some(lap) = &self.lap {
            let time = |seconds: Option<f64>| seconds.map_or("-:--.---".to_owned(), format_time);
            line.push_str(&format!(
                "   |   lap {}  {}{}   last {}   best {}",
                lap.count + 1,
                time(lap.current),
                if lap.valid { "" } else { " (off track)" },
                time(lap.last),
                time(lap.best)
            ));
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wheel_is_drawn_rolled_back_from_where_it_is() {
        let before = DrawnWheel {
            position: Vec3::new(0.8, 0.33, -1.3),
            steer: 0.0,
            angle: 1.0,
            speed: 240.0,
        };
        let now = DrawnWheel {
            position: Vec3::new(0.8, 0.35, -1.3),
            steer: 0.2,
            angle: 1.0 + 240.0 / 60.0,
            speed: 240.0,
        };
        // At the latest step, as it is; at the one before, as it was then, though that is
        // more than half a turn back.
        let at = |alpha: f32| DrawnWheel::between(&before, &now, alpha);
        let turned = |angle: f32, steer: f32, position: Vec3| {
            Mat4::from_rotation_translation(
                Quat::from_rotation_y(-steer) * Quat::from_rotation_x(-angle),
                position,
            )
        };
        assert!(at(1.0).abs_diff_eq(turned(now.angle, 0.2, now.position), 1.0e-5));
        assert!(at(0.0).abs_diff_eq(turned(before.angle, 0.0, before.position), 1.0e-4));
        let half = turned(3.0, 0.1, Vec3::new(0.8, 0.34, -1.3));
        assert!(at(0.5).abs_diff_eq(half, 1.0e-4));
        // Its upright goes where it goes and steers with it, and doesn't roll.
        let held = DrawnWheel::upright(&before, &now, 0.5);
        assert!(held.abs_diff_eq(turned(0.0, 0.1, Vec3::new(0.8, 0.34, -1.3)), 1.0e-5));
    }

    #[test]
    fn the_chase_camera_sits_behind_the_car_and_looks_at_it() {
        let mut chase = Chase::default();
        // A car at (10, 2, 5) heading +X: yaw of -90 degrees.
        let car = Mat4::from_rotation_translation(
            Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
            Vec3::new(10.0, 2.0, 5.0),
        );
        let ahead = Look::default();
        let shot = chase.frame(car, 0.0, ahead, 1.0 / 60.0);
        assert!(shot.position.x < 10.0 - 4.0, "behind: {}", shot.position);
        assert!((shot.position.z - 5.0).abs() < 1.0e-3 && shot.position.y > 3.0);
        assert!(
            (shot.yaw + std::f32::consts::FRAC_PI_2).abs() < 1.0e-3,
            "{}",
            shot.yaw
        );
        assert!(
            shot.pitch < 0.0 && shot.pitch > -0.4,
            "looking down a little"
        );
        // Looking back, it is ahead of the car, looking the other way.
        let back = Look {
            back: true,
            ..Look::default()
        };
        let shot = chase.frame(car, 0.0, back, 1.0 / 60.0);
        assert!(shot.position.x > 14.0);
        assert!((turn(shot.yaw, std::f32::consts::FRAC_PI_2)).abs() < 1.0e-3);
        // When the car turns, the camera comes round after it, not at once.
        let turned = Mat4::from_rotation_translation(Quat::IDENTITY, Vec3::new(10.0, 2.0, 5.0));
        let first = chase.frame(turned, 30.0, ahead, 1.0 / 60.0);
        assert!(
            first.position.x < 9.0 && first.position.z < 6.5,
            "{}",
            first.position
        );
        let mut shot = first;
        for _ in 0..120 {
            shot = chase.frame(turned, 30.0, ahead, 1.0 / 60.0);
        }
        assert!((shot.position.x - 10.0).abs() < 0.05 && shot.position.z > 9.0);
        assert!(shot.yaw.abs() < 0.01);
    }

    #[test]
    fn the_bonnet_camera_rides_with_the_car() {
        let mut chase = Chase {
            view: CameraView::Bonnet,
            ..Chase::default()
        };
        let car = Mat4::from_rotation_translation(
            Quat::from_rotation_y(0.5) * Quat::from_rotation_x(0.1),
            Vec3::new(1.0, 2.0, 3.0),
        );
        let shot = chase.frame(car, 20.0, Look::default(), 1.0 / 60.0);
        assert!((shot.yaw - 0.5).abs() < 1.0e-4 && (shot.pitch - 0.1).abs() < 1.0e-4);
        assert!(shot.position.distance(Vec3::new(1.0, 2.0, 3.0)) < 1.3);
        // The stick turns the driver's head: to the right is a lower yaw.
        let right = Look {
            around: 1.0,
            ..Look::default()
        };
        let mut turned = shot;
        for _ in 0..60 {
            turned = chase.frame(car, 20.0, right, 1.0 / 60.0);
        }
        assert!(
            turned.yaw < 0.5 - 1.5 && turned.yaw > 0.5 - 2.0,
            "{}",
            turned.yaw
        );
        assert_eq!(turned.position, shot.position);
    }

    #[test]
    fn the_views_step_out_from_the_car_and_then_onto_it() {
        let car = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0));
        let mut view = CameraView::Chase;
        let mut last = 0.0;
        // Each view that follows the car is further from it than the one before.
        for _ in 0..4 {
            let mut chase = Chase {
                view,
                ..Chase::default()
            };
            let shot = chase.frame(car, 0.0, Look::default(), 1.0 / 60.0);
            let away = shot.position.length();
            assert!(away > last * 1.3, "{view:?} is {away} m away, after {last}");
            // And looks down at the car from behind it.
            assert!(shot.position.z > 0.0 && shot.pitch < 0.0 && shot.yaw.abs() < 1.0e-4);
            last = away;
            view = view.next();
        }
        assert!(last > 20.0, "the last is well back: {last} m");
        assert_eq!(view, CameraView::Bonnet);
        assert_eq!(view.next().next(), CameraView::Chase);
    }

    #[test]
    fn the_stick_swings_the_camera_round_the_car() {
        // A car at the origin heading -Z: the camera starts behind it, at +Z.
        let car = Mat4::IDENTITY;
        let mut chase = Chase::default();
        let settle = |chase: &mut Chase, look: Look| {
            let mut shot = chase.frame(car, 0.0, look, 1.0 / 60.0);
            for _ in 0..90 {
                shot = chase.frame(car, 0.0, look, 1.0 / 60.0);
            }
            shot
        };
        let ahead = settle(&mut chase, Look::default());
        assert!(ahead.position.x.abs() < 1.0e-3 && ahead.position.z > 4.0);
        // Half-way to the right: from over the car's left shoulder, looking across it.
        let right = settle(
            &mut chase,
            Look {
                around: 0.5,
                ..Look::default()
            },
        );
        assert!(
            right.position.x < -2.0,
            "to the left of the car: {}",
            right.position
        );
        assert!(right.yaw < -0.3, "looking to the right: {}", right.yaw);
        // Right over: all but in front of the car, looking back at it.
        let over = settle(
            &mut chase,
            Look {
                around: 1.0,
                ..Look::default()
            },
        );
        assert!(
            over.position.z < -4.0,
            "ahead of the car: {}",
            over.position
        );
        // It does not jump there: the first frame of a push is a small move.
        let mut fresh = Chase::default();
        settle(&mut fresh, Look::default());
        let first = fresh.frame(
            car,
            0.0,
            Look {
                around: 1.0,
                ..Look::default()
            },
            1.0 / 60.0,
        );
        assert!(first.position.distance(ahead.position) < 3.5);
        // Let go, and it comes back behind.
        let back = settle(&mut chase, Look::default());
        assert!(back.position.distance(ahead.position) < 0.01);
        // Pulled back, it rises over the car; pushed away, it drops, but not into the road.
        let high = settle(
            &mut chase,
            Look {
                over: 1.0,
                ..Look::default()
            },
        );
        assert!(high.position.y > ahead.position.y + 2.0 && high.pitch < ahead.pitch - 0.3);
        let low = settle(
            &mut chase,
            Look {
                over: -1.0,
                ..Look::default()
            },
        );
        assert!(low.position.y < ahead.position.y && low.position.y > 0.3);
    }

    #[test]
    fn the_instruments_read_as_one_line() {
        let hud = Hud {
            telemetry: Telemetry {
                speed: 50.0,
                rpm: 6543.0,
                gear: 3,
                abs_active: true,
                ..Telemetry::default()
            },
            ..Hud::default()
        };
        assert_eq!(hud.line(), "180 km/h   3   6543 rpm   ABS");
    }
}
