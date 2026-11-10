//! Cars: `solstice-vehicle`'s dynamics, on the scene's collision, among everything else that
//! collides.
//!
//! A car is two things at once. Its wheels, springs, engine and tyres are `solstice-vehicle`'s,
//! which steps them many times a step against rays cast at the ground. Its body is also a rigid
//! body here, a box, so that it hits walls and other cars as anything else does.
//!
//! Each step, the car's own dynamics run first, from where the body is, and say where the step
//! ends. The body is then given the velocity that carries it exactly there, and the physics
//! step moves it and deals with whatever it hits. What that changed in its velocity, a wall
//! pushing back or another car, is added to the velocity the car's dynamics ended with.

use crate::{from_vector, to_pose, to_vector, Physics, GRAVITY};
use glam::{Quat, Vec3};
use rapier3d::prelude::*;
use solstice_vehicle::{Asked, Contact, Controls, Driver, Ground, Surface, Vehicle};
use solstice_world::{Entity, Transform, World, STEP_SECONDS};

/// Makes an actor a car. Its `Transform` is the car's origin: on the ground under its middle.
#[derive(Clone, Debug)]
pub struct Car {
    pub vehicle: Vehicle,
    /// What turns what is asked into the wheel and pedals: easing, and the automatic gearbox.
    pub driver: Driver,
    /// What its driver is asking for, set each step by whatever drives it.
    pub asked: Asked,
    /// What the wheel and pedals did in the latest step, for instruments to show.
    pub controls: Controls,
}

impl Car {
    pub fn new(vehicle: Vehicle, automatic: bool) -> Self {
        Self {
            vehicle,
            driver: Driver::new(automatic),
            asked: Asked::default(),
            controls: Controls::default(),
        }
    }
}

/// A car's step, between its own dynamics and the physics step.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CarStep {
    /// The velocity and spin its dynamics ended the step with.
    velocity: Vec3,
    spin: Vec3,
    /// The velocity and spin its body was given, to carry it to where the step ends.
    carried: Vec3,
    turned: Vec3,
}

/// The scene, as a wheel finds it.
struct SceneGround<'a> {
    queries: QueryPipeline<'a>,
    surfaces: Option<&'a dyn Fn(Vec3) -> Surface>,
}

impl Ground for SceneGround<'_> {
    fn cast(&self, from: Vec3, direction: Vec3, reach: f32) -> Option<Contact> {
        let ray = Ray::new(to_vector(from), to_vector(direction));
        let (_, hit) = self.queries.cast_ray_and_get_normal(&ray, reach, true)?;
        let mut normal = from_vector(hit.normal);
        // A mesh is met from either side: the ground faces the wheel.
        if normal.dot(direction) > 0.0 {
            normal = -normal;
        }
        let point = from + direction * hit.time_of_impact;
        Some(Contact {
            distance: hit.time_of_impact,
            normal,
            surface: self
                .surfaces
                .map_or(Surface::Asphalt, |surface| surface(point)),
        })
    }
}

impl Physics {
    /// Says what the ground is made of at each place, for tyres. Without it, all is asphalt.
    pub fn set_surfaces(&mut self, surfaces: impl Fn(Vec3) -> Surface + 'static) {
        self.surfaces = Some(Box::new(surfaces));
    }

    /// Gives a body to each new car.
    pub(crate) fn take_in_cars(&mut self, world: &mut World) {
        let new: Vec<(Entity, Transform)> = world
            .entities
            .query::<(Entity, &Transform, &Car)>()
            .iter()
            .filter(|(entity, _, _)| !self.actors.contains_key(entity))
            .map(|(entity, transform, _)| (entity, *transform))
            .collect();
        for (entity, transform) in new {
            let sound = crate::reachable(transform.translation) && transform.rotation.is_finite();
            if !sound {
                self.warnings.push(format!(
                    "a car at {} is out of bounds, and was left out",
                    transform.translation
                ));
                let _ = world.entities.remove_one::<Car>(entity);
                continue;
            }
            let Ok(car) = world.entities.get::<&Car>(entity) else {
                continue;
            };
            let spec = car.vehicle.spec();
            // Its mass and inertia are the car's own, not a box's: the collider weighs nothing.
            let mass =
                MassProperties::new(to_vector(spec.centre), spec.mass, to_vector(spec.inertia));
            let handle = self.bodies.insert(
                RigidBodyBuilder::dynamic()
                    .pose(to_pose(transform.translation, transform.rotation))
                    .additional_mass_properties(mass)
                    // Gravity is among the forces the car's own dynamics sum.
                    .gravity_scale(0.0)
                    .linear_damping(0.0)
                    .angular_damping(0.0)
                    .can_sleep(false)
                    // At 80 metres a second a car crosses a barrier's thickness in a step.
                    .ccd_enabled(true)
                    .user_data(crate::entity_data(entity)),
            );
            // Rounded, so that it slides along a wall's joins rather than catching on them.
            const ROUNDING: f32 = 0.08;
            let half = (spec.body_size * 0.5 - Vec3::splat(ROUNDING)).max(Vec3::splat(0.01));
            let collider = self.colliders.insert_with_parent(
                ColliderBuilder::round_cuboid(half.x, half.y, half.z, ROUNDING)
                    .translation(to_vector(spec.body_at))
                    .density(0.0)
                    .friction(0.35)
                    .restitution(0.15)
                    .user_data(crate::entity_data(entity)),
                handle,
                &mut self.bodies,
            );
            self.owners.insert(collider, (entity, false));
            self.actors.insert(
                entity,
                crate::Body {
                    handle,
                    kind: crate::BodyKind::Dynamic(spec.mass),
                    character: None,
                    at: transform.translation,
                    rotation: transform.rotation,
                    colliders: vec![collider],
                },
            );
            self.cars.insert(entity, CarStep::default());
        }
        self.cars
            .retain(|entity, _| self.actors.contains_key(entity));
    }

    /// Runs each car's own dynamics for the step, and gives its body the velocity that carries
    /// it to where they end.
    pub(crate) fn drive_cars(&mut self, world: &mut World) {
        let dt = STEP_SECONDS as f32;
        let entities: Vec<Entity> = self.cars.keys().copied().collect();
        for entity in entities {
            let Some(handle) = self.actors.get(&entity).map(|actor| actor.handle) else {
                continue;
            };
            let Ok(mut car) = world.entities.get::<&mut Car>(entity) else {
                continue;
            };
            let car = &mut *car;
            let Some(rigid) = self.bodies.get(handle) else {
                continue;
            };
            let rotation = rigid.rotation();
            let before = solstice_vehicle::Body {
                position: from_vector(rigid.translation()),
                rotation: Quat::from_xyzw(rotation.x, rotation.y, rotation.z, rotation.w),
                velocity: from_vector(rigid.linvel()),
                spin: from_vector(rigid.angvel()),
            };
            let mut body = before;
            {
                let filter = QueryFilter::new()
                    .exclude_sensors()
                    .exclude_rigid_body(handle);
                let ground = SceneGround {
                    queries: self.broad_phase.as_query_pipeline(
                        self.narrow_phase.query_dispatcher(),
                        &self.bodies,
                        &self.colliders,
                        filter,
                    ),
                    surfaces: self.surfaces.as_deref(),
                };
                let controls = car
                    .driver
                    .drive(&car.asked, &car.vehicle, &body, GRAVITY, dt);
                car.vehicle.step(&mut body, &controls, &ground, dt);
                car.controls = controls;
            }
            if !(body.position.is_finite() && body.velocity.is_finite() && body.spin.is_finite()) {
                self.warnings
                    .push("a car's dynamics gave no answer for a step, which was skipped".into());
                continue;
            }
            let centre = car.vehicle.spec().centre;
            let moved = (body.position + body.rotation * centre)
                - (before.position + before.rotation * centre);
            let step = CarStep {
                velocity: body.velocity,
                spin: body.spin,
                carried: moved / dt,
                turned: (body.rotation * before.rotation.inverse()).to_scaled_axis() / dt,
            };
            let rigid = self.bodies.get_mut(handle).expect("found above");
            rigid.set_linvel(to_vector(step.carried), true);
            rigid.set_angvel(to_vector(step.turned), true);
            self.cars.insert(entity, step);
        }
    }

    /// After the physics step: each car's velocity is what its dynamics ended with, and what
    /// hitting things did to it.
    pub(crate) fn settle_cars(&mut self) {
        for (entity, step) in &self.cars {
            let Some(rigid) = self
                .actors
                .get(entity)
                .and_then(|actor| self.bodies.get_mut(actor.handle))
            else {
                continue;
            };
            let knocked = from_vector(rigid.linvel()) - step.carried;
            let twisted = from_vector(rigid.angvel()) - step.turned;
            rigid.set_linvel(to_vector(step.velocity + knocked), true);
            rigid.set_angvel(to_vector(step.spin + twisted), true);
        }
    }
}
