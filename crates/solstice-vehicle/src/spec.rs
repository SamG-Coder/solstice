//! A car's numbers, and the `.vehicle` text file they are kept in.
//!
//! ```text
//! # A front-mid-engined GT racing car.
//! mass 1320                          # kilograms, with its driver and fuel
//! inertia 1780 1950 540              # about its pitch, yaw and roll axes, kg m²
//! centre 0 0.4 0.04                  # the centre of mass from the actor's origin: right, up, back
//! body 1.9 0.9 4.5 at 0 0.65 0       # the box it hits things with, and where its middle is
//! front at=1.32 track=1.66 radius=0.335 inertia=1.5
//! front spring=118000 bump=7600 rebound=10500 bar=34000 travel=0.065,0.075
//! rear  at=1.36 track=1.64 radius=0.345 inertia=1.9 driven=yes
//! tyre grip=1.72,1.68 peak=0.09,7.5 relaxation=0.28
//! engine idle=1000 limit=8500 inertia=0.13 friction=35,0.011
//! torque 1000 300                    # newton metres at full throttle, by rpm: one line a point
//! gears 3.15 2.29 1.78 1.45 1.21 1.03
//! final 3.7
//! ```
//!
//! Lengths are in metres, forces in newtons, torques in newton metres, angles in degrees. The
//! car's origin is on the ground under the middle of the car, at the height its wheels touch at
//! rest; it looks along -Z, with +X to its right. Every line may be left out: what it sets then
//! stays as [`VehicleSpec::gt`] has it.

use crate::tyre::Tyre;
use glam::Vec3;
use std::fmt::{self, Write as _};

/// A mistake in a `.vehicle` file, and the line it is on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for SpecError {}

/// The two wheels at one end of the car, and what holds them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axle {
    /// How far from the origin the axle is, in metres: ahead for the front axle, behind for the
    /// rear.
    pub at: f32,
    /// Between the middles of the two tyres.
    pub track: f32,
    /// The tyre's rolling radius.
    pub radius: f32,
    /// Of one wheel with its tyre, brake disc and half-shaft, about its axle, in kg m².
    pub inertia: f32,
    /// The spring's rate at the wheel, in newtons a metre.
    pub spring: f32,
    /// The damper's rates at the wheel, in newton seconds a metre: as the wheel rises, and as
    /// it falls.
    pub bump: f32,
    pub rebound: f32,
    /// The anti-roll bar: newtons for each metre one wheel is higher than the other.
    pub bar: f32,
    /// How far the wheel can rise from where it rests before the bump stop, and how far fall.
    pub travel_up: f32,
    pub travel_down: f32,
    /// The height of the roll centre: where the tyres' side force reaches the body.
    pub roll_centre: f32,
    /// The most torque one brake can make.
    pub brake: f32,
    /// Times the tyre's grip: for an axle with wider or narrower tyres.
    pub grip: f32,
    /// Whether the engine turns these wheels.
    pub driven: bool,
}

/// The engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Engine {
    /// Torque at full throttle, by revolutions a minute, rising in rpm.
    pub torque: Vec<(f32, f32)>,
    pub idle: f32,
    /// Where the limiter cuts in.
    pub limit: f32,
    /// Of the crankshaft, flywheel and clutch, in kg m².
    pub inertia: f32,
    /// What it takes to turn it with the throttle shut: newton metres, plus so many for each
    /// rpm. This is engine braking.
    pub friction: (f32, f32),
}

impl Engine {
    /// Torque at full throttle at this speed, between the points of its curve.
    pub fn full_torque(&self, rpm: f32) -> f32 {
        let curve = &self.torque;
        let (Some(first), Some(last)) = (curve.first(), curve.last()) else {
            return 0.0;
        };
        if rpm <= first.0 {
            return first.1;
        }
        if rpm >= last.0 {
            return last.1;
        }
        let above = curve.partition_point(|point| point.0 <= rpm);
        let (low, high) = (curve[above - 1], curve[above]);
        low.1 + (high.1 - low.1) * (rpm - low.0) / (high.0 - low.0)
    }

    /// What holds it back with the throttle shut, at this speed.
    pub fn drag_torque(&self, rpm: f32) -> f32 {
        self.friction.0 + self.friction.1 * rpm.max(0.0)
    }

    /// The most power it makes, in watts, and the rpm it makes it at.
    pub fn peak_power(&self) -> (f32, f32) {
        let mut best = (0.0, self.idle);
        let mut rpm = self.idle;
        while rpm <= self.limit {
            let power = self.full_torque(rpm) * rpm * RPM_TO_RADIANS;
            if power > best.0 {
                best = (power, rpm);
            }
            rpm += 25.0;
        }
        best
    }
}

/// Radians a second for each revolution a minute.
pub const RPM_TO_RADIANS: f32 = std::f32::consts::TAU / 60.0;

/// The gearbox and final drive.
#[derive(Clone, Debug, PartialEq)]
pub struct Gearbox {
    /// Each forward gear's ratio, first gear first.
    pub gears: Vec<f32>,
    pub reverse: f32,
    pub final_drive: f32,
    /// How long a shift cuts the drive for.
    pub shift_seconds: f32,
    /// The share of the engine's torque that reaches the wheels.
    pub efficiency: f32,
    /// The most torque the clutch can carry.
    pub clutch: f32,
}

/// A clutch-pack limited-slip differential: it resists its two wheels turning at different
/// speeds with a torque that grows with what it carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Differential {
    /// The locking torque it always has.
    pub preload: f32,
    /// More, as a share of the torque it carries, under power and on the overrun.
    pub power: f32,
    pub coast: f32,
}

/// The air.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aero {
    /// Drag coefficient times frontal area, in m².
    pub drag: f32,
    /// The same for downforce.
    pub downforce: f32,
    /// The share of the downforce on the front axle.
    pub balance: f32,
}

/// The density of air at sea level and 15 °C, in kg/m³.
pub const AIR_DENSITY: f32 = 1.225;

/// Everything about a car that its behaviour follows from.
#[derive(Clone, Debug, PartialEq)]
pub struct VehicleSpec {
    pub mass: f32,
    /// About the pitch (X), yaw (Y) and roll (Z) axes through the centre of mass, in kg m².
    pub inertia: Vec3,
    /// The centre of mass, from the origin.
    pub centre: Vec3,
    /// The box it hits walls and other cars with: its full size, and where its middle is.
    pub body_size: Vec3,
    pub body_at: Vec3,
    pub front: Axle,
    pub rear: Axle,
    pub tyre: Tyre,
    pub engine: Engine,
    pub gearbox: Gearbox,
    pub differential: Differential,
    pub aero: Aero,
    /// The most the front wheels turn, in radians, with the inner wheel at this.
    pub steering_lock: f32,
    /// How much more the inner wheel turns than the outer: 1 is true Ackermann, where every
    /// wheel rolls about one point, and 0 is parallel.
    pub ackermann: f32,
    /// The share of weight transfer under braking and under power that the suspension's links
    /// carry rather than its springs, which is what stops a car diving and squatting.
    pub anti_dive: f32,
    pub anti_squat: f32,
    /// Whether the brakes are kept from locking, and the driven wheels from spinning.
    pub abs: bool,
    pub traction_control: bool,
    /// Whether a tail that steps out is caught: the engine eased, the rear brakes let go,
    /// and the outside front wheel braked, as a road car's stability control does.
    pub stability: bool,
}

impl VehicleSpec {
    /// A front-mid-engined GT racing car, in the mould of the GT3 class.
    ///
    /// * 1320 kg with driver and fuel, 48% of it on the front axle, its centre of mass 40 cm up.
    /// * A naturally aspirated 4-litre V8: about 520 hp at 7500 rpm, 525 N m at 6000.
    /// * A six-speed sequential gearbox behind a limited-slip differential, driving the rear.
    /// * Slicks, on suspension with a ride frequency near 3 Hz, as cars with downforce run.
    /// * About 550 kg of downforce at 250 km/h, 42% of it on the front.
    pub fn gt() -> Self {
        Self {
            mass: 1320.0,
            inertia: Vec3::new(1780.0, 1950.0, 540.0),
            centre: Vec3::new(0.0, 0.4, 0.04),
            body_size: Vec3::new(1.9, 0.9, 4.5),
            body_at: Vec3::new(0.0, 0.65, 0.0),
            front: Axle {
                at: 1.32,
                track: 1.66,
                radius: 0.335,
                inertia: 1.5,
                spring: 118_000.0,
                bump: 7600.0,
                rebound: 10_500.0,
                bar: 34_000.0,
                travel_up: 0.065,
                travel_down: 0.075,
                roll_centre: 0.03,
                brake: 3300.0,
                grip: 1.0,
                driven: false,
            },
            rear: Axle {
                at: 1.36,
                track: 1.64,
                radius: 0.345,
                inertia: 1.9,
                spring: 128_000.0,
                bump: 8200.0,
                rebound: 11_500.0,
                bar: 20_000.0,
                travel_up: 0.07,
                travel_down: 0.08,
                roll_centre: 0.07,
                brake: 2300.0,
                grip: 1.04,
                driven: true,
            },
            tyre: Tyre::slick(),
            engine: Engine {
                torque: vec![
                    (1000.0, 300.0),
                    (2000.0, 385.0),
                    (3000.0, 445.0),
                    (4000.0, 490.0),
                    (5000.0, 518.0),
                    (6000.0, 525.0),
                    (6500.0, 520.0),
                    (7000.0, 507.0),
                    (7500.0, 488.0),
                    (8000.0, 452.0),
                    (8500.0, 405.0),
                    (9000.0, 330.0),
                ],
                idle: 1000.0,
                limit: 8500.0,
                inertia: 0.13,
                friction: (35.0, 0.011),
            },
            gearbox: Gearbox {
                gears: vec![3.15, 2.29, 1.78, 1.45, 1.21, 1.03],
                reverse: 2.9,
                final_drive: 3.7,
                shift_seconds: 0.06,
                efficiency: 0.93,
                clutch: 900.0,
            },
            differential: Differential {
                preload: 60.0,
                power: 0.45,
                coast: 0.3,
            },
            aero: Aero {
                drag: 0.92,
                downforce: 1.9,
                balance: 0.42,
            },
            steering_lock: 26.0_f32.to_radians(),
            ackermann: 0.6,
            anti_dive: 0.25,
            anti_squat: 0.3,
            abs: true,
            traction_control: true,
            stability: true,
        }
    }

    /// From the front axle to the rear.
    pub fn wheelbase(&self) -> f32 {
        self.front.at + self.rear.at
    }

    /// The share of the car's weight on the front axle, standing still.
    pub fn front_weight(&self) -> f32 {
        (self.rear.at - self.centre.z) / self.wheelbase()
    }

    /// The front axle's share of the braking, with the pedal hard down.
    pub fn brake_bias(&self) -> f32 {
        self.front.brake / (self.front.brake + self.rear.brake)
    }

    /// The load on one wheel of each axle, standing still, in newtons: front, then rear.
    pub fn static_loads(&self, gravity: f32) -> (f32, f32) {
        let weight = self.mass * gravity;
        let front = self.front_weight();
        (weight * front * 0.5, weight * (1.0 - front) * 0.5)
    }

    /// Where each wheel's middle is at rest, from the origin: front left, front right, rear
    /// left, rear right.
    pub fn hubs(&self) -> [Vec3; 4] {
        let (f, r) = (&self.front, &self.rear);
        [
            Vec3::new(-f.track * 0.5, f.radius, -f.at),
            Vec3::new(f.track * 0.5, f.radius, -f.at),
            Vec3::new(-r.track * 0.5, r.radius, r.at),
            Vec3::new(r.track * 0.5, r.radius, r.at),
        ]
    }

    /// The axle a wheel is on: the first two are the front's.
    pub fn axle(&self, wheel: usize) -> &Axle {
        if wheel < 2 {
            &self.front
        } else {
            &self.rear
        }
    }

    /// Reads a `.vehicle` file. What it leaves out stays as [`VehicleSpec::gt`] has it, but for
    /// the torque curve and the gears, which a file gives whole or not at all.
    pub fn parse(text: &str) -> Result<Self, SpecError> {
        let mut spec = Self::gt();
        let mut torque: Vec<(f32, f32)> = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let words: Vec<&str> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            let Some((&keyword, values)) = words.split_first() else {
                continue;
            };
            let error = |message: String| SpecError {
                line: index + 1,
                message,
            };
            let numbers = |count: usize, form: &str| -> Result<Vec<f32>, SpecError> {
                if values.len() != count {
                    return Err(error(format!("expected `{keyword} {form}`")));
                }
                values.iter().map(|w| number(w).map_err(&error)).collect()
            };
            match keyword {
                "mass" => spec.mass = numbers(1, "KILOGRAMS")?[0],
                "inertia" => spec.inertia = Vec3::from_slice(&numbers(3, "PITCH YAW ROLL")?),
                "centre" => spec.centre = Vec3::from_slice(&numbers(3, "X Y Z")?),
                "body" => {
                    let form = "expected `body SX SY SZ at X Y Z`";
                    let [sx, sy, sz, "at", x, y, z] = values else {
                        return Err(error(form.to_owned()));
                    };
                    let read = |w: &str| number(w).map_err(&error);
                    spec.body_size = Vec3::new(read(sx)?, read(sy)?, read(sz)?);
                    spec.body_at = Vec3::new(read(x)?, read(y)?, read(z)?);
                }
                "front" | "rear" => {
                    let axle = if keyword == "front" {
                        &mut spec.front
                    } else {
                        &mut spec.rear
                    };
                    for (key, value) in pairs(values).map_err(&error)? {
                        set_axle(axle, key, value).map_err(&error)?;
                    }
                }
                "tyre" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        set_tyre(&mut spec.tyre, key, value).map_err(&error)?;
                    }
                }
                "engine" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        let engine = &mut spec.engine;
                        match key {
                            "idle" => engine.idle = number(value).map_err(&error)?,
                            "limit" => engine.limit = number(value).map_err(&error)?,
                            "inertia" => engine.inertia = number(value).map_err(&error)?,
                            "friction" => engine.friction = pair(value).map_err(&error)?,
                            other => return Err(error(unknown("engine", other))),
                        }
                    }
                }
                "torque" => {
                    let point = numbers(2, "RPM NEWTON_METRES")?;
                    torque.push((point[0], point[1]));
                }
                "gears" => {
                    if values.is_empty() {
                        return Err(error("expected `gears RATIO...`, first gear first".into()));
                    }
                    spec.gearbox.gears = values
                        .iter()
                        .map(|w| number(w).map_err(&error))
                        .collect::<Result<_, _>>()?;
                }
                "final" => spec.gearbox.final_drive = numbers(1, "RATIO")?[0],
                "reverse" => spec.gearbox.reverse = numbers(1, "RATIO")?[0],
                "gearbox" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        let read = number(value).map_err(&error)?;
                        match key {
                            "shift" => spec.gearbox.shift_seconds = read,
                            "efficiency" => spec.gearbox.efficiency = read,
                            "clutch" => spec.gearbox.clutch = read,
                            other => return Err(error(unknown("gearbox", other))),
                        }
                    }
                }
                "diff" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        let read = number(value).map_err(&error)?;
                        match key {
                            "preload" => spec.differential.preload = read,
                            "power" => spec.differential.power = read,
                            "coast" => spec.differential.coast = read,
                            other => return Err(error(unknown("diff", other))),
                        }
                    }
                }
                "aero" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        let read = number(value).map_err(&error)?;
                        match key {
                            "drag" => spec.aero.drag = read,
                            "downforce" => spec.aero.downforce = read,
                            "balance" => spec.aero.balance = read,
                            other => return Err(error(unknown("aero", other))),
                        }
                    }
                }
                "steering" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        let read = number(value).map_err(&error)?;
                        match key {
                            "lock" => spec.steering_lock = read.to_radians(),
                            "ackermann" => spec.ackermann = read,
                            other => return Err(error(unknown("steering", other))),
                        }
                    }
                }
                "geometry" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        let read = number(value).map_err(&error)?;
                        match key {
                            "anti_dive" => spec.anti_dive = read,
                            "anti_squat" => spec.anti_squat = read,
                            other => return Err(error(unknown("geometry", other))),
                        }
                    }
                }
                "assists" => {
                    for (key, value) in pairs(values).map_err(&error)? {
                        let on = switch(value).map_err(&error)?;
                        match key {
                            "abs" => spec.abs = on,
                            "traction" => spec.traction_control = on,
                            "stability" => spec.stability = on,
                            other => return Err(error(unknown("assists", other))),
                        }
                    }
                }
                other => {
                    return Err(error(format!(
                        "unknown line `{other}`: a vehicle file has mass, inertia, centre, body, \
                         front, rear, tyre, engine, torque, gears, final, reverse, gearbox, diff, \
                         aero, steering, geometry and assists lines"
                    )))
                }
            }
        }
        if !torque.is_empty() {
            spec.engine.torque = torque;
        }
        spec.check()
            .map_err(|message| SpecError { line: 1, message })?;
        Ok(spec)
    }

    /// Whether a car can be made of these numbers, or why not. Numbers come from files, and a
    /// car with no mass, or wheels inside each other, would send the simulation to infinity.
    pub fn check(&self) -> Result<(), String> {
        let positive = |name: &str, value: f32| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("{name} must be above 0, not {value}"))
            }
        };
        let share = |name: &str, value: f32| {
            if (0.0..=1.0).contains(&value) {
                Ok(())
            } else {
                Err(format!("{name} is a share, between 0 and 1, not {value}"))
            }
        };
        positive("mass", self.mass)?;
        for (name, value) in [
            ("pitch inertia", self.inertia.x),
            ("yaw inertia", self.inertia.y),
            ("roll inertia", self.inertia.z),
            ("the body's width", self.body_size.x),
            ("the body's height", self.body_size.y),
            ("the body's length", self.body_size.z),
        ] {
            positive(name, value)?;
        }
        if !self.centre.is_finite() || !self.body_at.is_finite() {
            return Err("the centre of mass and the body's middle must be numbers".into());
        }
        for (name, axle) in [("front", &self.front), ("rear", &self.rear)] {
            for (what, value) in [
                ("at", axle.at),
                ("track", axle.track),
                ("radius", axle.radius),
                ("inertia", axle.inertia),
                ("spring", axle.spring),
                ("travel up", axle.travel_up),
                ("travel down", axle.travel_down),
                ("grip", axle.grip),
            ] {
                positive(&format!("{name} {what}"), value)?;
            }
            for (what, value) in [
                ("bump", axle.bump),
                ("rebound", axle.rebound),
                ("bar", axle.bar),
                ("brake", axle.brake),
                ("roll_centre", axle.roll_centre),
            ] {
                if !value.is_finite() || value < 0.0 {
                    return Err(format!("{name} {what} can't be below 0, not {value}"));
                }
            }
        }
        if self.centre.z <= -self.front.at || self.centre.z >= self.rear.at {
            return Err("the centre of mass must be between the axles".into());
        }
        if !self.front.driven && !self.rear.driven {
            return Err("one axle must be driven".into());
        }
        if self.front.driven && self.rear.driven {
            return Err("four-wheel drive isn't built yet: drive one axle".into());
        }
        let tyre = &self.tyre;
        for (name, value) in [
            ("tyre grip along", tyre.grip_long),
            ("tyre grip across", tyre.grip_lat),
            ("the tyre's peak slip ratio", tyre.peak_ratio),
            ("the tyre's peak slip angle", tyre.peak_angle),
            ("the tyre's nominal load", tyre.nominal_load),
            ("the tyre's relaxation length", tyre.relaxation),
        ] {
            positive(name, value)?;
        }
        for (name, (c, e)) in [("along", tyre.shape_long), ("across", tyre.shape_lat)] {
            if !(c > 1.0 && c < 2.0 && e < 1.0 && e.is_finite()) {
                return Err(format!(
                    "the tyre's shape {name} needs C between 1 and 2 and E below 1, not {c} and {e}"
                ));
            }
        }
        if !(0.0..1.0).contains(&tyre.load_sensitivity) {
            return Err("the tyre's load sensitivity is between 0 and 1".into());
        }
        let engine = &self.engine;
        if engine.torque.len() < 2 {
            return Err("the engine needs at least two `torque` points".into());
        }
        if engine.torque.windows(2).any(|pair| pair[1].0 <= pair[0].0) {
            return Err("the `torque` points must rise in rpm".into());
        }
        if engine.torque.iter().any(|p| !p.1.is_finite() || p.1 < 0.0) {
            return Err("a `torque` point can't be below 0".into());
        }
        positive("engine idle", engine.idle)?;
        positive("engine inertia", engine.inertia)?;
        if engine.limit <= engine.idle * 1.5 {
            return Err("the engine's limit must be well above its idle".into());
        }
        if engine.friction.0 < 0.0 || engine.friction.1 < 0.0 {
            return Err("engine friction can't be below 0".into());
        }
        let gearbox = &self.gearbox;
        if gearbox.gears.is_empty() || gearbox.gears.len() > 10 {
            return Err("a gearbox has one to ten gears".into());
        }
        for &ratio in &gearbox.gears {
            positive("a gear's ratio", ratio)?;
        }
        if gearbox.gears.windows(2).any(|pair| pair[1] >= pair[0]) {
            return Err("each gear must be taller than the one below: a smaller ratio".into());
        }
        positive("the final drive", gearbox.final_drive)?;
        positive("the reverse ratio", gearbox.reverse)?;
        positive("the clutch's torque", gearbox.clutch)?;
        if !(0.0..=1.0).contains(&gearbox.shift_seconds) {
            return Err("a shift takes between 0 and 1 seconds".into());
        }
        if !(0.5..=1.0).contains(&gearbox.efficiency) {
            return Err("the gearbox's efficiency is between 0.5 and 1".into());
        }
        let diff = &self.differential;
        if diff.preload < 0.0 || !(0.0..=1.0).contains(&diff.power) {
            return Err("the diff's preload can't be below 0, nor its power lock above 1".into());
        }
        share("the diff's coast lock", diff.coast)?;
        if self.aero.drag < 0.0 || self.aero.downforce < 0.0 {
            return Err("drag and downforce can't be below 0".into());
        }
        share("aero balance", self.aero.balance)?;
        if !(self.steering_lock > 0.0 && self.steering_lock < 1.2) {
            return Err("the steering lock is above 0 and below 68 degrees".into());
        }
        share("ackermann", self.ackermann)?;
        share("anti_dive", self.anti_dive)?;
        share("anti_squat", self.anti_squat)?;
        Ok(())
    }

    /// The file's text, which [`VehicleSpec::parse`] reads back as it is.
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        let yes = |on: bool| if on { "yes" } else { "no" };
        let _ = writeln!(text, "mass {}", self.mass);
        let i = self.inertia;
        let _ = writeln!(text, "inertia {} {} {}", i.x, i.y, i.z);
        let c = self.centre;
        let _ = writeln!(text, "centre {} {} {}", c.x, c.y, c.z);
        let (s, a) = (self.body_size, self.body_at);
        let _ = writeln!(
            text,
            "body {} {} {} at {} {} {}",
            s.x, s.y, s.z, a.x, a.y, a.z
        );
        for (name, axle) in [("front", &self.front), ("rear", &self.rear)] {
            let _ = writeln!(
                text,
                "{name} at={} track={} radius={} inertia={} driven={}",
                axle.at,
                axle.track,
                axle.radius,
                axle.inertia,
                yes(axle.driven)
            );
            let _ = writeln!(
                text,
                "{name} spring={} bump={} rebound={} bar={} travel={},{} roll_centre={}",
                axle.spring,
                axle.bump,
                axle.rebound,
                axle.bar,
                axle.travel_up,
                axle.travel_down,
                axle.roll_centre
            );
            let _ = writeln!(text, "{name} brake={} grip={}", axle.brake, axle.grip);
        }
        let t = &self.tyre;
        let _ = writeln!(
            text,
            "tyre grip={},{} peak={},{} along={},{} across={},{}",
            t.grip_long,
            t.grip_lat,
            t.peak_ratio,
            t.peak_angle.to_degrees(),
            t.shape_long.0,
            t.shape_long.1,
            t.shape_lat.0,
            t.shape_lat.1
        );
        let _ = writeln!(
            text,
            "tyre load={} sensitivity={} relaxation={}",
            t.nominal_load, t.load_sensitivity, t.relaxation
        );
        let e = &self.engine;
        let _ = writeln!(
            text,
            "engine idle={} limit={} inertia={} friction={},{}",
            e.idle, e.limit, e.inertia, e.friction.0, e.friction.1
        );
        for (rpm, torque) in &e.torque {
            let _ = writeln!(text, "torque {rpm} {torque}");
        }
        let g = &self.gearbox;
        let gears: Vec<String> = g.gears.iter().map(f32::to_string).collect();
        let _ = writeln!(text, "gears {}", gears.join(" "));
        let _ = writeln!(text, "final {}", g.final_drive);
        let _ = writeln!(text, "reverse {}", g.reverse);
        let _ = writeln!(
            text,
            "gearbox shift={} efficiency={} clutch={}",
            g.shift_seconds, g.efficiency, g.clutch
        );
        let d = &self.differential;
        let _ = writeln!(
            text,
            "diff preload={} power={} coast={}",
            d.preload, d.power, d.coast
        );
        let _ = writeln!(
            text,
            "aero drag={} downforce={} balance={}",
            self.aero.drag, self.aero.downforce, self.aero.balance
        );
        let _ = writeln!(
            text,
            "steering lock={} ackermann={}",
            self.steering_lock.to_degrees(),
            self.ackermann
        );
        let _ = writeln!(
            text,
            "geometry anti_dive={} anti_squat={}",
            self.anti_dive, self.anti_squat
        );
        let _ = writeln!(
            text,
            "assists abs={} traction={} stability={}",
            yes(self.abs),
            yes(self.traction_control),
            yes(self.stability)
        );
        text
    }
}

fn unknown(line: &str, key: &str) -> String {
    format!("a `{line}` line has no `{key}`")
}

fn number(word: &str) -> Result<f32, String> {
    word.parse::<f32>()
        .ok()
        .filter(|n| n.is_finite())
        .ok_or_else(|| format!("`{word}` is not a number"))
}

/// Two numbers with a comma between: `0.065,0.075`.
fn pair(word: &str) -> Result<(f32, f32), String> {
    let (first, second) = word
        .split_once(',')
        .ok_or_else(|| format!("`{word}` should be two numbers with a comma between"))?;
    Ok((number(first)?, number(second)?))
}

fn switch(word: &str) -> Result<bool, String> {
    match word {
        "yes" | "on" | "true" => Ok(true),
        "no" | "off" | "false" => Ok(false),
        other => Err(format!("`{other}` should be yes or no")),
    }
}

/// Splits each word into its key and value: `spring=118000`.
fn pairs<'a>(words: &[&'a str]) -> Result<Vec<(&'a str, &'a str)>, String> {
    if words.is_empty() {
        return Err("expected `KEY=VALUE` after it".to_owned());
    }
    words
        .iter()
        .map(|word| {
            word.split_once('=')
                .filter(|(key, value)| !key.is_empty() && !value.is_empty())
                .ok_or_else(|| format!("`{word}` should be `KEY=VALUE`"))
        })
        .collect()
}

fn set_axle(axle: &mut Axle, key: &str, value: &str) -> Result<(), String> {
    match key {
        "at" => axle.at = number(value)?,
        "track" => axle.track = number(value)?,
        "radius" => axle.radius = number(value)?,
        "inertia" => axle.inertia = number(value)?,
        "spring" => axle.spring = number(value)?,
        "bump" => axle.bump = number(value)?,
        "rebound" => axle.rebound = number(value)?,
        "bar" => axle.bar = number(value)?,
        "travel" => (axle.travel_up, axle.travel_down) = pair(value)?,
        "roll_centre" => axle.roll_centre = number(value)?,
        "brake" => axle.brake = number(value)?,
        "grip" => axle.grip = number(value)?,
        "driven" => axle.driven = switch(value)?,
        other => return Err(unknown("front or rear", other)),
    }
    Ok(())
}

fn set_tyre(tyre: &mut Tyre, key: &str, value: &str) -> Result<(), String> {
    match key {
        "grip" => (tyre.grip_long, tyre.grip_lat) = pair(value)?,
        "peak" => {
            let (ratio, degrees) = pair(value)?;
            tyre.peak_ratio = ratio;
            tyre.peak_angle = degrees.to_radians();
        }
        "along" => tyre.shape_long = pair(value)?,
        "across" => tyre.shape_lat = pair(value)?,
        "load" => tyre.nominal_load = number(value)?,
        "sensitivity" => tyre.load_sensitivity = number(value)?,
        "relaxation" => tyre.relaxation = number(value)?,
        other => return Err(unknown("tyre", other)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gt_is_sound_and_is_what_it_says() {
        let gt = VehicleSpec::gt();
        gt.check().unwrap();
        assert!((gt.wheelbase() - 2.68).abs() < 1.0e-5);
        let front = gt.front_weight();
        assert!((0.47..0.50).contains(&front), "{front}");
        let (power, rpm) = gt.engine.peak_power();
        let horsepower = power / 745.7;
        assert!((505.0..530.0).contains(&horsepower), "{horsepower} hp");
        assert!((7000.0..8000.0).contains(&rpm), "{rpm}");
        // Geared for a top speed near 290 km/h at the limiter in top.
        let top = gt.engine.limit * RPM_TO_RADIANS
            / (gt.gearbox.gears.last().unwrap() * gt.gearbox.final_drive)
            * gt.rear.radius;
        assert!((78.0..84.0).contains(&top), "{} km/h", top * 3.6);
        // About 550 kg of downforce at 250 km/h.
        let speed = 250.0 / 3.6_f32;
        let downforce = 0.5 * AIR_DENSITY * gt.aero.downforce * speed * speed / 9.81;
        assert!((500.0..620.0).contains(&downforce), "{downforce} kg");
    }

    #[test]
    fn torque_is_read_between_its_points() {
        let engine = VehicleSpec::gt().engine;
        assert_eq!(engine.full_torque(500.0), 300.0);
        assert_eq!(engine.full_torque(6000.0), 525.0);
        assert!((engine.full_torque(5500.0) - 521.5).abs() < 1.0e-3);
        assert_eq!(engine.full_torque(20_000.0), 330.0);
        assert!((engine.drag_torque(7000.0) - 112.0).abs() < 1.0e-3);
    }

    #[test]
    fn a_file_reads_back_what_it_wrote() {
        let mut spec = VehicleSpec::gt();
        spec.mass = 1405.5;
        spec.front.bar = 12_345.0;
        spec.rear.travel_down = 0.11;
        spec.abs = false;
        spec.gearbox.gears = vec![3.0, 2.0, 1.5, 1.0];
        spec.engine.torque = vec![(900.0, 200.0), (4000.0, 410.5), (7000.0, 380.0)];
        spec.tyre.peak_angle = 6.0_f32.to_radians();
        let text = spec.to_text();
        let back = VehicleSpec::parse(&text).unwrap();
        // Degrees and back lose a last bit.
        assert!((back.tyre.peak_angle - spec.tyre.peak_angle).abs() < 1.0e-6);
        assert!((back.steering_lock - spec.steering_lock).abs() < 1.0e-6);
        let mut same = back.clone();
        same.tyre.peak_angle = spec.tyre.peak_angle;
        same.steering_lock = spec.steering_lock;
        assert_eq!(same, spec);
        assert_eq!(back.to_text(), text);
    }

    #[test]
    fn an_empty_file_is_the_gt() {
        assert_eq!(
            VehicleSpec::parse("# nothing\n\n").unwrap(),
            VehicleSpec::gt()
        );
    }

    #[test]
    fn mistakes_are_reported_with_their_line() {
        let cases = [
            ("mass heavy", 1, "not a number"),
            ("mass 1200\nwings 2", 2, "unknown line `wings`"),
            ("front spring", 1, "KEY=VALUE"),
            ("front springs=3", 1, "no `springs`"),
            ("front travel=0.1", 1, "comma"),
            ("mass 0", 1, "mass must be above 0"),
            ("gears 3 2 2.5", 1, "taller"),
            ("torque 1000 300", 1, "two `torque` points"),
            ("torque 2000 300\ntorque 1000 300", 1, "rise in rpm"),
            ("centre 0 0.4 3", 1, "between the axles"),
            ("front driven=yes", 1, "four-wheel drive"),
            ("assists abs=maybe", 1, "yes or no"),
            ("body 1 1 1", 1, "body SX SY SZ at"),
            ("tyre along=2.5,0", 1, "C between 1 and 2"),
            ("aero balance=1.4", 1, "share"),
        ];
        for (text, line, part) in cases {
            let error = VehicleSpec::parse(text).unwrap_err();
            assert_eq!(error.line, line, "{text}: {error}");
            assert!(error.message.contains(part), "{text}: {error}");
        }
    }
}
