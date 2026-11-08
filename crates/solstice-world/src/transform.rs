//! Where things are: each entity's place relative to its parent, and the world's view of it.
//!
//! An entity's [`Transform`] is relative to its [`Parent`], or to the world if it has none. After
//! every step the world works out each one's [`GlobalTransform`], parents before children, and
//! keeps the one before, so a frame can draw things between the two steps it falls between.

use glam::{Affine3A, Mat4, Quat, Vec3};
use hecs::Entity;

/// A place, turn and size, relative to the parent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    pub fn with_rotation(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    pub fn affine(&self) -> Affine3A {
        Affine3A::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// The entity this one is attached to. Its transform is then relative to the parent's. A parent
/// that no longer exists counts as none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parent(pub Entity);

/// Where an entity is in the world after the latest step, and after the step before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalTransform {
    pub now: Affine3A,
    pub before: Affine3A,
}

impl GlobalTransform {
    /// Where to draw it, `alpha` of the way from the step before to the latest: the frame's
    /// place between the two steps it falls between. Positions and sizes are blended, turns are
    /// slerped.
    pub fn interpolated(&self, alpha: f32) -> Mat4 {
        if alpha >= 1.0 || self.now == self.before {
            return Mat4::from(self.now);
        }
        let (s0, r0, t0) = self.before.to_scale_rotation_translation();
        let (s1, r1, t1) = self.now.to_scale_rotation_translation();
        let alpha = alpha.max(0.0);
        Mat4::from_scale_rotation_translation(
            s0.lerp(s1, alpha),
            r0.slerp(r1, alpha),
            t0.lerp(t1, alpha),
        )
    }
}

/// The longest chain of parents followed. A longer one is taken as a loop, and cut there.
const MAX_DEPTH: u32 = 256;

/// Marks an entity id with no transform in [`Propagation::slots`].
const NONE: u32 = u32::MAX;

/// The working arrays of one update, indexed densely rather than hashed: entity ids are small
/// and packed, so an array as long as the largest id costs little and finds each in one step.
struct Propagation {
    /// Where each entity id's transform is in the arrays below, or [`NONE`].
    slots: Vec<u32>,
    locals: Vec<Affine3A>,
    /// Each one's parent's slot, or [`NONE`].
    parents: Vec<u32>,
    globals: Vec<Affine3A>,
    done: Vec<bool>,
}

impl Propagation {
    /// An entity's place in the world, from its parents', each worked out once.
    fn resolve(&mut self, slot: u32, depth: u32) -> Affine3A {
        let i = slot as usize;
        if self.done[i] {
            return self.globals[i];
        }
        let parent = self.parents[i];
        let global = if parent != NONE && depth < MAX_DEPTH {
            self.resolve(parent, depth + 1) * self.locals[i]
        } else {
            self.locals[i]
        };
        self.globals[i] = global;
        self.done[i] = true;
        global
    }
}

/// Works out every entity's [`GlobalTransform`] from its [`Transform`] and its parents', keeping
/// last step's as the one before. Entities without one yet get one, with no motion.
pub fn update_global_transforms(world: &mut hecs::World) {
    let mut query = world.query::<(Entity, &Transform, Option<&Parent>)>();
    let found: Vec<(u32, Affine3A, Option<Entity>)> = query
        .iter()
        .map(|(entity, transform, parent)| (entity.id(), transform.affine(), parent.map(|p| p.0)))
        .collect();
    drop(query);

    let longest = found.iter().map(|f| f.0 as usize + 1).max().unwrap_or(0);
    let mut slots = vec![NONE; longest];
    for (slot, &(id, _, _)) in found.iter().enumerate() {
        slots[id as usize] = slot as u32;
    }
    // A parent counts only if it is the entity named, alive, and has a transform.
    let parents = found
        .iter()
        .map(|&(_, _, parent)| {
            parent
                .filter(|&p| world.contains(p))
                .and_then(|p| slots.get(p.id() as usize).copied())
                .unwrap_or(NONE)
        })
        .collect();
    let count = found.len();
    let mut work = Propagation {
        slots,
        locals: found.iter().map(|f| f.1).collect(),
        parents,
        globals: vec![Affine3A::IDENTITY; count],
        done: vec![false; count],
    };
    for slot in 0..count as u32 {
        work.resolve(slot, 0);
    }

    let mut new = Vec::new();
    for (entity, global) in world.query_mut::<(Entity, Option<&mut GlobalTransform>)>() {
        let Some(&slot) = work.slots.get(entity.id() as usize) else {
            continue;
        };
        if slot == NONE {
            continue;
        }
        let now = work.globals[slot as usize];
        match global {
            Some(global) => {
                global.before = global.now;
                global.now = now;
            }
            None => new.push((entity, now)),
        }
    }
    for (entity, now) in new {
        world
            .insert_one(entity, GlobalTransform { now, before: now })
            .expect("the entity was just queried");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f32) -> Transform {
        Transform::from_translation(Vec3::new(x, 0.0, 0.0))
    }

    fn position(world: &hecs::World, entity: Entity) -> Vec3 {
        world
            .get::<&GlobalTransform>(entity)
            .unwrap()
            .now
            .translation
            .into()
    }

    #[test]
    fn children_follow_their_parents_however_deep() {
        let mut world = hecs::World::new();
        let root = world
            .spawn((at(1.0).with_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)),));
        let child = world.spawn((at(2.0), Parent(root)));
        let grandchild = world.spawn((at(3.0).with_scale(Vec3::splat(2.0)), Parent(child)));
        let great = world.spawn((at(1.0), Parent(grandchild)));
        update_global_transforms(&mut world);
        // The root turns everything a quarter turn: +X becomes -Z.
        assert!(position(&world, child).abs_diff_eq(Vec3::new(1.0, 0.0, -2.0), 1.0e-5));
        assert!(position(&world, grandchild).abs_diff_eq(Vec3::new(1.0, 0.0, -5.0), 1.0e-5));
        // The grandchild's scale doubles the step to its own child.
        assert!(position(&world, great).abs_diff_eq(Vec3::new(1.0, 0.0, -7.0), 1.0e-5));
    }

    #[test]
    fn the_step_before_is_kept_and_new_entities_start_still() {
        let mut world = hecs::World::new();
        let ball = world.spawn((at(1.0),));
        update_global_transforms(&mut world);
        let global = *world.get::<&GlobalTransform>(ball).unwrap();
        assert_eq!(global.now, global.before);
        world.get::<&mut Transform>(ball).unwrap().translation.x = 4.0;
        update_global_transforms(&mut world);
        let global = *world.get::<&GlobalTransform>(ball).unwrap();
        assert_eq!(Vec3::from(global.before.translation).x, 1.0);
        assert_eq!(Vec3::from(global.now.translation).x, 4.0);
        // Halfway between the two steps.
        let between = global.interpolated(0.5).w_axis.truncate();
        assert!(between.abs_diff_eq(Vec3::new(2.5, 0.0, 0.0), 1.0e-5));
        // Each step's place becomes the next step's place before.
        world.get::<&mut Transform>(ball).unwrap().translation.x = 6.0;
        update_global_transforms(&mut world);
        let global = *world.get::<&GlobalTransform>(ball).unwrap();
        assert_eq!(Vec3::from(global.before.translation).x, 4.0);
        assert_eq!(Vec3::from(global.now.translation).x, 6.0);
    }

    #[test]
    fn interpolation_turns_the_short_way_at_a_steady_rate() {
        let global = GlobalTransform {
            before: Affine3A::from_rotation_y(0.0),
            now: Affine3A::from_rotation_y(1.0),
        };
        let quarter = global.interpolated(0.25);
        let angle = Quat::from_mat4(&quarter).angle_between(Quat::IDENTITY);
        assert!((angle - 0.25).abs() < 1.0e-4, "{angle}");
        assert_eq!(global.interpolated(1.0), Mat4::from(global.now));
        assert!(global
            .interpolated(0.0)
            .abs_diff_eq(Mat4::from(global.before), 1.0e-6));
    }

    #[test]
    fn a_missing_parent_counts_as_none_and_a_loop_does_not_hang() {
        let mut world = hecs::World::new();
        let gone = world.spawn((at(100.0),));
        let orphan = world.spawn((at(2.0), Parent(gone)));
        world.despawn(gone).unwrap();
        let a = world.spawn((at(1.0),));
        let b = world.spawn((at(1.0), Parent(a)));
        world.insert_one(a, Parent(b)).unwrap();
        update_global_transforms(&mut world);
        assert_eq!(position(&world, orphan), Vec3::new(2.0, 0.0, 0.0));
        assert!(position(&world, a).is_finite() && position(&world, b).is_finite());
    }

    #[test]
    fn entities_without_a_transform_are_left_alone() {
        let mut world = hecs::World::new();
        let thing = world.spawn((42_u32,));
        update_global_transforms(&mut world);
        assert!(world.get::<&GlobalTransform>(thing).is_err());
    }
}
