//! A car, stepped: what the ground, the air and its own engine do to it.
//!
//! Each fixed step is cut into [`SUBSTEPS`] shorter ones, as a tyre and a damper answer in
//! milliseconds. Each of those:
//!
//! 1. **Finds the ground** under each wheel with one ray down the suspension's axis, and from
//!    how far the wheel has risen, the spring, damper, anti-roll bar and bump stop give the load
//!    on its tyre.
//! 2. **Solves the drivetrain**: the engine, clutch, gearbox, differential, brakes and the two
//!    driven wheels, with the tyres' grip on the road, all at once (see [`drivetrain`]).
//! 3. **Sums the forces** on the body: each tyre's, gravity, drag and downforce.
//! 4. **Moves the body**: its velocity and spin, then its place and turn.
//!
//! The body is a rigid one with its own mass and inertia. The game's physics engine moves the
//! same body and deals with what it hits: it is handed where this step ended (see
//! [`Body`]), and adds what the collision did.

use crate::drivetrain::{self, Coupling, DrivenAxle};
use crate::spec::{VehicleSpec, AIR_DENSITY, RPM_TO_RADIANS};
use crate::surface::{bump, Surface};
use crate::tyre::TyreCurves;
use glam::{Quat, Vec3};

/// How many shorter steps one fixed step is cut into.
pub const SUBSTEPS: u32 = 10;
/// Below this speed, in metres a second, slip is measured against this speed instead of the
/// wheel's own: slip is a ratio of speeds, and has no meaning at a standstill.
const LOW_SPEED: f32 = 2.0;
/// How far above its highest point a wheel's ray starts, so that a wheel pushed past its bump
/// stop still finds the ground.
const RAY_MARGIN: f32 = 0.15;
/// The bump stop's rate, in newtons a metre: what the wheel meets at the top of its travel.
const BUMP_STOP: f32 = 600_000.0;
/// The damper's knee: above this speed, in metres a second, it stiffens only this much as fast.
const DAMPER_KNEE: f32 = 0.13;
const DAMPER_FAST: f32 = 0.45;
/// The most a tyre can be loaded, as a multiple of its load at rest: a landing, not a number.
const MAX_LOAD: f32 = 9.0;
/// The slip ratio at which the anti-lock brakes let go, and at which traction control cuts in:
/// just past where the tyre's grip peaks.
const ABS_SLIP: f32 = 0.13;
const TRACTION_SLIP: f32 = 0.11;
/// How much more slip than that takes the throttle away altogether.
const TRACTION_BAND: f32 = 0.12;
/// Stability control: how far the rear tyres' slip angle may pass the angle they grip best at
/// before it steps in, as a multiple of that angle, and at how much it is doing all it can.
const STABILITY_FROM: f32 = 0.95;
const STABILITY_FULL: f32 = 1.9;
/// How far ahead it looks, in seconds, at the rate the tail is stepping out: a tail that is
/// swinging fast is caught before it has gone far.
const STABILITY_AHEAD: f32 = 0.16;
/// The most of the front brake it puts on the outside front wheel to pull the nose back.
const STABILITY_BRAKE: f32 = 0.6;
/// Under braking, the rear brakes begin to be let go at this multiple of that angle: a tail
/// that the brakes are holding at the edge of its grip has none left to hold the car straight.
const STABILITY_RELEASE: f32 = 0.55;
/// Below this speed, in metres a second, there is nothing for it to do.
const STABILITY_SPEED: f32 = 7.0;
/// How far above idle, in rpm, the clutch starts to bite when pulling away flat out.
const LAUNCH_REVS: f32 = 2600.0;
/// Within this many rpm of the gearbox's speed, the engine counts as turning with it.
const CLUTCH_MATCHED: f32 = 150.0;

/// What the ground is under a wheel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// How far along the ray it was met, in metres.
    pub distance: f32,
    /// Which way the ground faces there: a unit vector.
    pub normal: Vec3,
    pub surface: Surface,
}

/// The world, as a wheel finds it.
pub trait Ground {
    /// What a ray from `from` along the unit vector `direction` meets first within `reach`.
    fn cast(&self, from: Vec3, direction: Vec3, reach: f32) -> Option<Contact>;
}

/// Flat ground at a height, of one surface: for tests, and for tuning a car.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flat {
    pub height: f32,
    pub surface: Surface,
}

impl Ground for Flat {
    fn cast(&self, from: Vec3, direction: Vec3, reach: f32) -> Option<Contact> {
        if direction.y >= -1.0e-6 {
            return None;
        }
        let distance = (self.height - from.y) / direction.y;
        (0.0..=reach).contains(&distance).then_some(Contact {
            distance,
            normal: Vec3::Y,
            surface: self.surface,
        })
    }
}

/// Where the car's body is and how it is moving.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    /// The car's origin: on the ground under its middle, when it stands level.
    pub position: Vec3,
    pub rotation: Quat,
    /// Of its centre of mass, in metres a second.
    pub velocity: Vec3,
    /// How it is turning, in radians a second about each of the world's axes.
    pub spin: Vec3,
}

impl Body {
    /// Standing still at a place, turned by `yaw` radians about +Y. Zero looks along -Z.
    pub fn at_rest(position: Vec3, yaw: f32) -> Self {
        Self {
            position,
            rotation: Quat::from_rotation_y(yaw),
            velocity: Vec3::ZERO,
            spin: Vec3::ZERO,
        }
    }

    /// The way the car points.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }
}

/// What the driver is doing with the car, after any help (see [`crate::driver`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Controls {
    /// From 0 to 1.
    pub throttle: f32,
    pub brake: f32,
    /// From -1, full left, to 1, full right.
    pub steer: f32,
    /// From 0 to 1: locks the rear wheels.
    pub handbrake: f32,
    /// Asks for the next gear up or down, once: set for one step.
    pub shift_up: bool,
    pub shift_down: bool,
}

/// One wheel, as it is now.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wheel {
    /// How fast it turns, in radians a second: above zero rolling forward.
    pub speed: f32,
    /// How far it has turned, in radians, kept within one turn.
    pub angle: f32,
    /// How far it has risen from where it rests, in metres: below zero, hanging.
    pub travel: f32,
    /// How far it is steered, in radians: above zero to the right.
    pub steer: f32,
    /// How hard its tyre is pressed to the ground, in newtons.
    pub load: f32,
    /// Its slip ratio, and the tangent of its slip angle, as the tyre feels them.
    pub slip_ratio: f32,
    pub slide: f32,
    /// The force the ground puts on it: along the way it points, and to its right.
    pub force: (f32, f32),
    pub on_ground: bool,
    pub surface: Surface,
    /// The share of the brake's pressure the anti-lock system is letting through.
    pub brake_share: f32,
}

/// What the instruments show, and what a game wants to know of how a car is doing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Telemetry {
    /// Along the car, in metres a second: below zero going backwards.
    pub speed: f32,
    pub rpm: f32,
    /// -1 is reverse, 0 neutral.
    pub gear: i32,
    /// What the engine is being asked for, from 0 to 1, after the limiter and traction control.
    pub throttle: f32,
    /// The body's acceleration over the last step, in g: forward, to the right, and up.
    pub g: Vec3,
    /// Whether the anti-lock brakes, traction control or stability control are at work.
    pub abs_active: bool,
    pub traction_active: bool,
    pub stability_active: bool,
    /// Whether the engine is at its limiter.
    pub limiter: bool,
}

/// A gear change under way.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shift {
    seconds_left: f32,
    /// Whether it is to a lower gear, where the engine is blipped up to the wheels' speed.
    down: bool,
}

/// A car: its numbers, and the state of everything that turns.
#[derive(Clone, Debug)]
pub struct Vehicle {
    spec: VehicleSpec,
    curves: TyreCurves,
    gravity: f32,
    /// The load on a front wheel and on a rear one, standing still.
    static_loads: (f32, f32),
    pub wheels: [Wheel; 4],
    /// The engine's speed, in radians a second.
    engine_speed: f32,
    gear: i32,
    shift: Option<Shift>,
    /// The torque through the clutch in the last short step, which sets how hard the
    /// differential locks.
    clutch_torque: f32,
    /// The share of the throttle traction control is letting through.
    traction_share: f32,
    /// How far the tail was stepping out a short step ago: the rear axle's slip angle, in
    /// radians, above zero when it slides to the right.
    rear_slide: f32,
    limiting: bool,
    telemetry: Telemetry,
}

impl Vehicle {
    /// A car of these numbers, standing still with its engine idling, in neutral.
    ///
    /// # Panics
    /// If the numbers fail [`VehicleSpec::check`]: check what comes from a file first.
    pub fn new(spec: VehicleSpec, gravity: f32) -> Self {
        if let Err(message) = spec.check() {
            panic!("an unsound vehicle: {message}");
        }
        let wheel = Wheel {
            brake_share: 1.0,
            ..Wheel::default()
        };
        Self {
            curves: spec.tyre.curves(),
            static_loads: spec.static_loads(gravity),
            engine_speed: spec.engine.idle * RPM_TO_RADIANS,
            gravity,
            spec,
            wheels: [wheel; 4],
            gear: 0,
            shift: None,
            clutch_torque: 0.0,
            traction_share: 1.0,
            rear_slide: 0.0,
            limiting: false,
            telemetry: Telemetry::default(),
        }
    }

    pub fn spec(&self) -> &VehicleSpec {
        &self.spec
    }

    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    pub fn gear(&self) -> i32 {
        self.gear
    }

    pub fn rpm(&self) -> f32 {
        self.engine_speed / RPM_TO_RADIANS
    }

    /// Whether a gear change is under way.
    pub fn shifting(&self) -> bool {
        self.shift.is_some()
    }

    /// Puts the car in a gear at once: -1 for reverse, 0 for neutral. For starting a race, and
    /// for tests.
    pub fn set_gear(&mut self, gear: i32) {
        self.gear = gear.clamp(-1, self.spec.gearbox.gears.len() as i32);
        self.shift = None;
    }

    /// Stops every wheel and sets the engine idling: for putting a car back on the track.
    pub fn settle(&mut self) {
        for wheel in &mut self.wheels {
            *wheel = Wheel {
                angle: wheel.angle,
                brake_share: 1.0,
                ..Wheel::default()
            };
        }
        self.engine_speed = self.spec.engine.idle * RPM_TO_RADIANS;
        self.gear = 0;
        self.shift = None;
        self.clutch_torque = 0.0;
        self.traction_share = 1.0;
        self.rear_slide = 0.0;
    }

    /// Sets every wheel rolling at the speed the car is going: for starting a car that is
    /// already moving, with the engine turning to match in the gear it is in.
    pub fn roll_with(&mut self, body: &Body) {
        let speed = body.velocity.dot(body.forward());
        for (index, wheel) in self.wheels.iter_mut().enumerate() {
            wheel.speed = speed / self.spec.axle(index).radius;
        }
        if let Some(ratio) = self.ratio() {
            let driven = self.driven_wheels();
            let wheels = 0.5 * (self.wheels[driven[0]].speed + self.wheels[driven[1]].speed);
            self.engine_speed = (wheels * ratio).max(self.spec.engine.idle * RPM_TO_RADIANS);
        }
    }

    /// The gear's ratio with the final drive, from the engine to the wheels: below zero in
    /// reverse, and none in neutral.
    pub fn ratio(&self) -> Option<f32> {
        let gearbox = &self.spec.gearbox;
        match self.gear {
            0 => None,
            gear if gear < 0 => Some(-gearbox.reverse * gearbox.final_drive),
            gear => Some(gearbox.gears[gear as usize - 1] * gearbox.final_drive),
        }
    }

    fn driven_wheels(&self) -> [usize; 2] {
        if self.spec.front.driven {
            [0, 1]
        } else {
            [2, 3]
        }
    }

    /// Where a wheel is drawn, from the car's origin, and how it is turned: steered, and rolled.
    pub fn wheel_pose(&self, index: usize) -> (Vec3, Quat) {
        let axle = self.spec.axle(index);
        let wheel = &self.wheels[index];
        let travel = wheel.travel.clamp(-axle.travel_down, axle.travel_up + 0.02);
        (
            self.spec.hubs()[index] + Vec3::Y * travel,
            Quat::from_rotation_y(-wheel.steer) * Quat::from_rotation_x(-wheel.angle),
        )
    }

    /// How far each front wheel is steered, in radians, for this much of the lock: left, right.
    ///
    /// The wheel on the inside of the turn follows a tighter circle, so it turns further.
    fn steer_angles(&self, steer: f32) -> (f32, f32) {
        let middle = steer.clamp(-1.0, 1.0) * self.spec.steering_lock;
        if middle.abs() < 1.0e-5 {
            return (middle, middle);
        }
        let wheelbase = self.spec.wheelbase();
        let half_track = self.spec.front.track * 0.5;
        // The turn's radius to the car's middle line: above zero turning right.
        let radius = wheelbase / middle.tan();
        let ideal = |offset: f32| (wheelbase / (radius - offset)).atan();
        let blend = |ideal: f32| middle + self.spec.ackermann * (ideal - middle);
        (blend(ideal(-half_track)), blend(ideal(half_track)))
    }

    /// Asks for a gear change, if one can be made: not while another is under way, not into a
    /// gear that would over-rev the engine, and into reverse only when nearly stopped.
    fn request_shift(&mut self, up: bool, speed: f32) {
        if self.shift.is_some() {
            return;
        }
        let top = self.spec.gearbox.gears.len() as i32;
        let target = self.gear + if up { 1 } else { -1 };
        if !(-1..=top).contains(&target) {
            return;
        }
        if target == -1 && speed > 1.5 {
            return;
        }
        if target >= 1 && speed < -1.5 {
            return;
        }
        if !up && target >= 1 {
            let gearbox = &self.spec.gearbox;
            let ratio = gearbox.gears[target as usize - 1] * gearbox.final_drive;
            let driven = self.driven_wheels();
            let wheels = 0.5 * (self.wheels[driven[0]].speed + self.wheels[driven[1]].speed);
            if wheels * ratio > (self.spec.engine.limit + 300.0) * RPM_TO_RADIANS {
                return;
            }
        }
        let seconds = self.spec.gearbox.shift_seconds * if up { 1.0 } else { 2.0 };
        self.gear = target;
        self.shift = (seconds > 0.0).then_some(Shift {
            seconds_left: seconds,
            down: !up,
        });
    }

    /// Moves the car through one fixed step of `seconds`.
    pub fn step(
        &mut self,
        body: &mut Body,
        controls: &Controls,
        ground: &dyn Ground,
        seconds: f32,
    ) {
        let speed = body.velocity.dot(body.forward());
        if controls.shift_up {
            self.request_shift(true, speed);
        } else if controls.shift_down {
            self.request_shift(false, speed);
        }
        let (left, right) = self.steer_angles(controls.steer);
        self.wheels[0].steer = left;
        self.wheels[1].steer = right;

        let before = body.velocity;
        let dt = seconds / SUBSTEPS as f32;
        let mut aids = (false, false, false);
        for _ in 0..SUBSTEPS {
            let (abs, traction, stability) = self.substep(body, controls, ground, dt);
            aids = (aids.0 | abs, aids.1 | traction, aids.2 | stability);
        }

        let acceleration = (body.velocity - before) / seconds;
        // What the driver feels: the acceleration, less gravity's.
        let felt = acceleration + Vec3::Y * self.gravity;
        self.telemetry = Telemetry {
            speed: body.velocity.dot(body.forward()),
            rpm: self.rpm(),
            gear: self.gear,
            throttle: self.telemetry.throttle,
            g: Vec3::new(
                felt.dot(body.forward()),
                felt.dot(body.right()),
                felt.dot(body.up()),
            ) / self.gravity,
            abs_active: aids.0,
            traction_active: aids.1,
            stability_active: aids.2,
            limiter: self.limiting,
        };
    }

    /// One short step. Returns whether the anti-lock brakes, traction control and stability
    /// control were at work.
    fn substep(
        &mut self,
        body: &mut Body,
        controls: &Controls,
        ground: &dyn Ground,
        dt: f32,
    ) -> (bool, bool, bool) {
        let spec = &self.spec;
        let hubs = spec.hubs();
        let (forward, up) = (body.forward(), body.up());
        let centre = body.position + body.rotation * spec.centre;

        // 1. The ground under each wheel, and how far each has risen.
        let mut found: [Option<Found>; 4] = [None; 4];
        let mut travel = [0.0_f32; 4];
        for index in 0..4 {
            let axle = spec.axle(index);
            let top_offset = axle.travel_up + RAY_MARGIN;
            let top = body.position + body.rotation * hubs[index] + up * top_offset;
            let reach = top_offset + axle.travel_down + axle.radius;
            travel[index] = -axle.travel_down;
            let Some(contact) = ground.cast(top, -up, reach) else {
                continue;
            };
            // A wall, or the underside of something: not ground to stand on.
            if contact.normal.dot(up) < 0.3 || !contact.normal.is_finite() {
                continue;
            }
            let grip = contact.surface.grip();
            let point = top - up * contact.distance;
            let rough = if grip.bump_height > 0.0 {
                grip.bump_height * bump(point.x, point.z, grip.bump_length)
            } else {
                0.0
            };
            let risen = top_offset - (contact.distance - rough - axle.radius);
            if risen <= -axle.travel_down {
                continue;
            }
            travel[index] = risen;
            found[index] = Some(Found {
                point,
                normal: contact.normal,
                surface: contact.surface,
            });
        }

        // The load on each tyre: spring, damper, anti-roll bar and bump stop.
        let mut loads = [0.0_f32; 4];
        for index in 0..4 {
            let Some(contact) = &found[index] else {
                continue;
            };
            let axle = spec.axle(index);
            let rest = if index < 2 {
                self.static_loads.0
            } else {
                self.static_loads.1
            };
            let arm = contact.point - centre;
            let moving = body.velocity + body.spin.cross(arm);
            let rising = -moving.dot(contact.normal);
            let damper = if rising > 0.0 {
                axle.bump
            } else {
                axle.rebound
            };
            let damped = damper
                * if rising.abs() < DAMPER_KNEE {
                    rising
                } else {
                    rising.signum() * (DAMPER_KNEE + DAMPER_FAST * (rising.abs() - DAMPER_KNEE))
                };
            // The bar pushes down on the wheel that has risen more than its neighbour.
            let other = index ^ 1;
            let bar = axle.bar * (travel[index] - travel[other]);
            let stop = BUMP_STOP * (travel[index] - axle.travel_up).max(0.0);
            let load = rest + axle.spring * travel[index] + damped + bar + stop;
            loads[index] = load.clamp(0.0, rest * MAX_LOAD);
        }

        // 2. Each tyre's slip, and the force it makes at the speed its wheel turns now.
        let mut patches: [Option<Patch>; 4] = [None; 4];
        for index in 0..4 {
            let wheel = &mut self.wheels[index];
            wheel.travel = travel[index];
            wheel.load = loads[index];
            wheel.on_ground = found[index].is_some();
            let Some(contact) = found[index].filter(|_| loads[index] > 0.0) else {
                wheel.slide = 0.0;
                wheel.slip_ratio = 0.0;
                wheel.force = (0.0, 0.0);
                continue;
            };
            wheel.surface = contact.surface;
            let axle = spec.axle(index);
            let heading = body.rotation * (Quat::from_rotation_y(-wheel.steer) * Vec3::NEG_Z);
            let along =
                (heading - contact.normal * heading.dot(contact.normal)).normalize_or_zero();
            if along == Vec3::ZERO {
                continue;
            }
            let across = along.cross(contact.normal);
            let arm = contact.point - centre;
            let moving = body.velocity + body.spin.cross(arm);
            let (rolling, sliding) = (moving.dot(along), moving.dot(across));
            let against = rolling.abs().max(LOW_SPEED);
            // The side force lags the slip angle: the tyre must roll a way to take up its set.
            let slide = -sliding / against;
            let catch_up = 1.0 - (-dt * (rolling.abs() + 3.0) / spec.tyre.relaxation).exp();
            wheel.slide += (slide - wheel.slide) * catch_up;
            let surface = contact.surface.grip();
            patches[index] = Some(Patch {
                point: contact.point,
                normal: contact.normal,
                along,
                across,
                rolling,
                against,
                grip: surface.grip * axle.grip,
                resistance: -surface.rolling_at(rolling) * loads[index] * (rolling / 0.5).tanh(),
            });
        }

        // The tyre's force along, as the drivetrain needs it: at the wheel's speed now, and how
        // it changes with that speed.
        let tyre = spec.tyre;
        let curves = self.curves;
        let grip_force = |patch: &Patch, load: f32, slide: f32, radius: f32, speed: f32| {
            let ratio = (speed * radius - patch.rolling) / patch.against;
            tyre.force(&curves, load, ratio, slide, patch.grip)
        };
        let mut along_now = [0.0_f32; 4];
        let mut stiffness = [0.0_f32; 4];
        for index in 0..4 {
            let Some(patch) = &patches[index] else {
                continue;
            };
            let (wheel, radius) = (&self.wheels[index], spec.axle(index).radius);
            const NUDGE: f32 = 0.25;
            let (now, _) = grip_force(patch, loads[index], wheel.slide, radius, wheel.speed);
            let (more, _) = grip_force(
                patch,
                loads[index],
                wheel.slide,
                radius,
                wheel.speed + NUDGE,
            );
            along_now[index] = now;
            // Past its peak a tyre grips less the more it slips. Solving ahead with that would
            // feed the slip; holding the force steady there is safe, and corrected next step.
            stiffness[index] = ((more - now) / NUDGE).max(0.0);
        }

        // The driver's aids, from the slip of the step before.
        let mut abs_active = false;
        for (wheel, patch) in self.wheels.iter_mut().zip(&patches) {
            // Locking: turning slower than the road passes, whichever way the car is going.
            let slipping = spec.abs
                && controls.brake > 0.0
                && wheel.on_ground
                && patch.is_some_and(|p| {
                    p.rolling.abs() > 2.5 && wheel.slip_ratio * p.rolling.signum() < -ABS_SLIP
                });
            wheel.brake_share = if slipping {
                abs_active = true;
                (wheel.brake_share - dt * 22.0).max(0.05)
            } else {
                (wheel.brake_share + dt * 9.0).min(1.0)
            };
        }
        // Traction control eases the throttle by how far past the tyre's peak the faster
        // driven wheel is spinning, quickly, and gives it back more slowly.
        let driven = self.driven_wheels();
        let wheelspin = driven
            .iter()
            .filter(|&&index| self.wheels[index].on_ground)
            .map(|&index| self.wheels[index].slip_ratio * self.gear.signum() as f32)
            .fold(0.0, f32::max);
        let spinning =
            spec.traction_control && controls.throttle > 0.0 && wheelspin > TRACTION_SLIP;
        // At once, no more throttle than the slip allows; and for as long as the wheel goes on
        // spinning, less and less, until it is back at the edge of its grip. Without the
        // second, it would settle part-way, spinning a little for good.
        self.traction_share = if spinning {
            let excess = (wheelspin - TRACTION_SLIP) / TRACTION_BAND;
            (self.traction_share - dt * 14.0 * excess)
                .min(1.0 - excess)
                .max(0.0)
        } else {
            (self.traction_share + dt * 3.5).min(1.0)
        };

        // Stability control watches the tail. The rear tyres' slip angle is how far the rear
        // axle's own path points from the way the car does; past the angle they grip best at,
        // the tail is on its way round. It looks a little ahead at the rate that is growing,
        // then eases the engine, lets go of the rear brakes, and brakes the outside front
        // wheel, which pulls the nose back into line.
        let behind = body.rotation * (Vec3::new(0.0, 0.0, spec.rear.at) - spec.centre);
        let passing = body.velocity + body.spin.cross(behind);
        let (sideways, onward) = (passing.dot(body.right()), passing.dot(forward));
        let rear_slide = if onward > STABILITY_SPEED {
            sideways.atan2(onward)
        } else {
            0.0
        };
        let growing = ((rear_slide.abs() - self.rear_slide.abs()) / dt).max(0.0);
        self.rear_slide = rear_slide;
        let peak = spec.tyre.peak_angle;
        let slide = rear_slide.abs() + growing * STABILITY_AHEAD;
        let over =
            |from: f32| ((slide - peak * from) / (peak * (STABILITY_FULL - from))).clamp(0.0, 1.0);
        let (catching, releasing) = if spec.stability {
            (over(STABILITY_FROM), over(STABILITY_RELEASE))
        } else {
            (0.0, 0.0)
        };
        // The front wheel on the side the tail is sliding towards: the outside one.
        let outside = if rear_slide > 0.0 { 1 } else { 0 };

        // The engine: what it is asked for, and the torque it makes.
        let engine = &spec.engine;
        let rpm = self.engine_speed / RPM_TO_RADIANS;
        self.limiting = if self.limiting {
            rpm > engine.limit - 150.0
        } else {
            rpm > engine.limit
        };
        let ratio = self.ratio();
        let wheels_speed = 0.5 * (self.wheels[driven[0]].speed + self.wheels[driven[1]].speed);
        // How fast the gearbox's input shaft turns: what the engine would turn at, in gear.
        let shaft = ratio.map_or(0.0, |ratio| wheels_speed * ratio);
        let mut throttle =
            controls.throttle.clamp(0.0, 1.0) * self.traction_share * (1.0 - catching);
        if let Some(shift) = &self.shift {
            // A lower gear wants the engine faster: blip it up to the wheels' speed.
            throttle = if shift.down && self.engine_speed < shaft * 0.97 {
                1.0
            } else {
                0.0
            };
        }
        if self.limiting {
            throttle = 0.0;
        }
        // Idling: enough throttle to keep turning.
        let idling = ((engine.idle * 1.03 - rpm) / 150.0).clamp(0.0, 1.0) * 0.6;
        throttle = throttle.max(idling);
        self.telemetry.throttle = throttle;
        let engine_torque =
            engine.full_torque(rpm) * throttle - engine.drag_torque(rpm) * (1.0 - throttle);

        // The clutch: open in a gear change, slipping to pull away, shut once rolling.
        let shaft_rpm = shaft / RPM_TO_RADIANS;
        let engaged = if self.shift.is_some() || ratio.is_none() {
            0.0
        } else if shaft_rpm > engine.idle * 1.15 && shaft_rpm >= rpm - CLUTCH_MATCHED {
            // Rolling, with the engine and the gearbox turning together, or the wheels
            // driving the engine: shut.
            1.0
        } else {
            // Pulling away, or the engine still coming down to the gearbox's speed: the clutch
            // bites as the revs rise past a point that goes up with the throttle. A gentle
            // start slips it just above idle; a hard one holds the engine up in its torque
            // while the car catches up. Traction control eases it with the throttle, as the
            // engine's own momentum would spin the wheels whatever the throttle did.
            let bite = engine.idle * 1.05 + controls.throttle.clamp(0.0, 1.0) * LAUNCH_REVS;
            ((rpm - bite) / 1500.0).clamp(0.0, 1.0) * self.traction_share
        };
        let diff = &spec.differential;
        let carried = self.clutch_torque * ratio.unwrap_or(0.0).abs() * spec.gearbox.efficiency;
        let lock = diff.preload
            + carried.abs()
                * if self.clutch_torque >= 0.0 {
                    diff.power
                } else {
                    diff.coast
                };

        // The brakes: each wheel's, as far as the anti-lock system lets it.
        let brake = controls.brake.clamp(0.0, 1.0);
        let handbrake = controls.handbrake.clamp(0.0, 1.0);
        let mut brakes = [0.0_f32; 4];
        for (index, torque) in brakes.iter_mut().enumerate() {
            let axle = spec.axle(index);
            let share = self.wheels[index].brake_share;
            *torque = brake * axle.brake * share;
            if index >= 2 {
                *torque *= 1.0 - 0.92 * releasing;
                *torque += handbrake * 3000.0;
            } else if index == outside {
                *torque =
                    (*torque + catching * STABILITY_BRAKE * axle.brake * share).min(axle.brake);
            }
        }

        // Solve the turning parts for their speeds at the end of this short step.
        let axle = spec.axle(driven[0]);
        let side = |index: usize| drivetrain::Wheel {
            speed: self.wheels[index].speed,
            inertia: spec.axle(index).inertia,
            radius: spec.axle(index).radius,
            force: along_now[index],
            stiffness: stiffness[index],
            brake: brakes[index],
        };
        let solved = drivetrain::solve(
            &DrivenAxle {
                engine_speed: self.engine_speed,
                engine_inertia: engine.inertia,
                engine_torque,
                ratio: ratio.unwrap_or(0.0),
                efficiency: spec.gearbox.efficiency,
                clutch: Coupling::new(spec.gearbox.clutch * engaged),
                differential: Coupling::new(lock),
                left: side(driven[0]),
                right: side(driven[1]),
            },
            dt,
        );
        debug_assert!(axle.driven);
        let mut speeds = [0.0_f32; 4];
        speeds[driven[0]] = solved.left;
        speeds[driven[1]] = solved.right;
        for index in (0..4).filter(|index| !driven.contains(index)) {
            speeds[index] = drivetrain::solve_free(&side(index), dt);
        }
        self.engine_speed = solved.engine.max(0.0);
        self.clutch_torque = solved.clutch_torque;

        // 3. The forces on the body.
        let mut force = Vec3::new(0.0, -spec.mass * self.gravity, 0.0);
        let mut torque = Vec3::ZERO;
        let mut push = |at: Vec3, pushed: Vec3| {
            force += pushed;
            torque += (at - centre).cross(pushed);
        };
        for index in 0..4 {
            let wheel = &mut self.wheels[index];
            let axle = spec.axle(index);
            let was = wheel.speed;
            wheel.speed = speeds[index];
            wheel.angle = (wheel.angle + wheel.speed * dt) % std::f32::consts::TAU;
            let Some(patch) = &patches[index] else {
                continue;
            };
            let load = loads[index];
            wheel.slip_ratio = (wheel.speed * axle.radius - patch.rolling) / patch.against;
            // Along: what the drivetrain was solved with, so that the wheel and the road agree.
            // Across: what the tyre gives at its new slip. Both within what the tyre can do.
            let limit = load * tyre.load_factor(load) * patch.grip;
            let mut along = along_now[index] + stiffness[index] * (wheel.speed - was);
            let (_, mut across) = grip_force(patch, load, wheel.slide, axle.radius, wheel.speed);
            let used = (along / (limit * tyre.grip_long)).hypot(across / (limit * tyre.grip_lat));
            if used > 1.0 {
                along /= used;
                across /= used;
            }
            wheel.force = (along, across);

            push(patch.point, patch.normal * load);
            // The suspension's links carry a share of the weight transfer straight to the
            // body: as if the tyre's force reached it higher up.
            let links = if index < 2 || along < 0.0 {
                spec.anti_dive
            } else {
                spec.anti_squat
            };
            push(
                patch.point + up * (links * spec.centre.y),
                patch.along * (along + patch.resistance),
            );
            push(patch.point + up * axle.roll_centre, patch.across * across);
        }

        // The air: drag against the way it goes, and downforce on each axle.
        let air_speed = body.velocity.length();
        push(
            centre,
            -body.velocity * (0.5 * AIR_DENSITY * spec.aero.drag * air_speed),
        );
        let ahead = body.velocity.dot(forward);
        let downforce = 0.5 * AIR_DENSITY * spec.aero.downforce * ahead * ahead;
        let axle_at = |z: f32| body.position + body.rotation * Vec3::new(0.0, spec.centre.y, z);
        push(
            axle_at(-spec.front.at),
            -up * (downforce * spec.aero.balance),
        );
        push(
            axle_at(spec.rear.at),
            -up * (downforce * (1.0 - spec.aero.balance)),
        );

        // 4. Move the body: velocity and spin first, then place and turn with the new ones.
        body.velocity += force / spec.mass * dt;
        let inverse = body.rotation.inverse();
        let mut spin = inverse * body.spin;
        let turning = inverse * torque - spin.cross(spec.inertia * spin);
        spin += turning / spec.inertia * dt;
        body.spin = body.rotation * spin;
        let centre = centre + body.velocity * dt;
        body.rotation = (Quat::from_scaled_axis(body.spin * dt) * body.rotation).normalize();
        body.position = centre - body.rotation * spec.centre;

        if let Some(shift) = &mut self.shift {
            shift.seconds_left -= dt;
            if shift.seconds_left <= 0.0 {
                self.shift = None;
            }
        }
        (
            abs_active,
            spinning,
            catching > 0.02 || (brake > 0.0 && releasing > 0.02),
        )
    }
}

/// Where a wheel's ray met the ground.
#[derive(Clone, Copy)]
struct Found {
    point: Vec3,
    normal: Vec3,
    surface: Surface,
}

/// A tyre's contact with the ground.
#[derive(Clone, Copy)]
struct Patch {
    point: Vec3,
    normal: Vec3,
    /// The way the tyre points, along the ground, and to its right.
    along: Vec3,
    across: Vec3,
    /// How fast the ground passes under it, along the way it points.
    rolling: f32,
    /// The speed slip is measured against.
    against: f32,
    /// The surface's grip, with the axle's.
    grip: f32,
    /// Rolling resistance, along the way it points.
    resistance: f32,
}
