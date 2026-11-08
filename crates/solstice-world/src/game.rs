//! Running a world: tick groups, the systems in them, and the frame.
//!
//! ```text
//!  each frame:
//!    0 to 3 fixed steps of 1/60 s, each:
//!      PrePhysics    gameplay: script ticks, Rust systems, root motion
//!      Physics       the physics step
//!      PostPhysics   reactions to hits and overlaps
//!      (transforms)  every entity's place in the world
//!    Animation       poses, at the frame's own time
//!    Late            the camera
//! ```
//!
//! Everything that decides what happens runs in the fixed steps, from input recorded per step,
//! so a run replays exactly at any frame rate. The per-frame groups only work out what is drawn:
//! they read the world, but send no events and set no timers.

use crate::clock::{FixedClock, FrameSteps};
use crate::input::{Input, Recording};
use crate::transform::update_global_transforms;
use crate::world::World;

/// When in a frame a system runs. Within a group, systems run in the order they were added.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TickGroup {
    /// Each step, before physics: gameplay decides what happens.
    PrePhysics,
    /// Each step: the physics step.
    Physics,
    /// Each step, after physics: reactions to what it found.
    PostPhysics,
    /// Each frame, after the steps: poses for drawing.
    Animation,
    /// Each frame, last: the camera, and anything that follows what was drawn.
    Late,
}

impl TickGroup {
    pub const ALL: [Self; 5] = [
        Self::PrePhysics,
        Self::Physics,
        Self::PostPhysics,
        Self::Animation,
        Self::Late,
    ];
    const STEP: [Self; 3] = [Self::PrePhysics, Self::Physics, Self::PostPhysics];
    const FRAME: [Self; 2] = [Self::Animation, Self::Late];

    /// Whether it runs in each fixed step, rather than each frame.
    pub fn is_fixed(self) -> bool {
        Self::STEP.contains(&self)
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// A system: a name, for profiling and errors, and what it does.
struct System {
    name: &'static str,
    run: Box<dyn FnMut(&mut World)>,
}

/// The systems of each tick group, in order.
#[derive(Default)]
pub struct Schedule {
    groups: [Vec<System>; TickGroup::ALL.len()],
}

impl Schedule {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a system to run in `group`, after those already there.
    pub fn add(
        &mut self,
        group: TickGroup,
        name: &'static str,
        run: impl FnMut(&mut World) + 'static,
    ) {
        self.groups[group.index()].push(System {
            name,
            run: Box::new(run),
        });
    }

    /// The names of a group's systems, in the order they run.
    pub fn names(&self, group: TickGroup) -> impl Iterator<Item = &'static str> + '_ {
        self.groups[group.index()].iter().map(|system| system.name)
    }

    fn run(&mut self, group: TickGroup, world: &mut World) {
        for system in &mut self.groups[group.index()] {
            (system.run)(world);
        }
    }
}

/// A world, its systems and its clock.
pub struct Game {
    pub world: World,
    pub schedule: Schedule,
    clock: FixedClock,
    /// Every step's input, while recording.
    recording: Option<Recording>,
}

impl Game {
    pub fn new(world: World, schedule: Schedule) -> Self {
        Self {
            world,
            schedule,
            clock: FixedClock::new(),
            recording: None,
        }
    }

    /// Runs a frame `seconds` long, with `input` as the player's input during it: the fixed
    /// steps the frame holds, then the per-frame groups. Returns how many steps ran, and how far
    /// the frame is between the last step and the next, for drawing.
    pub fn frame(&mut self, seconds: f64, input: &Input) -> FrameSteps {
        let steps = self.clock.advance(seconds);
        for _ in 0..steps.steps {
            self.step(input);
        }
        for group in TickGroup::FRAME {
            self.world.set_in_frame_group(true);
            self.schedule.run(group, &mut self.world);
            self.world.set_in_frame_group(false);
        }
        steps
    }

    /// Runs one fixed step with `input`.
    pub fn step(&mut self, input: &Input) {
        if let Some(recording) = &mut self.recording {
            recording.push(input);
        }
        self.world.begin_step(input);
        for group in TickGroup::STEP {
            self.schedule.run(group, &mut self.world);
            self.world.end_step_group();
        }
        update_global_transforms(&mut self.world.entities);
    }

    /// Starts recording every step's input, from the next step on.
    pub fn record(&mut self) {
        self.recording = Some(Recording::new());
    }

    /// Stops recording, and gives back what was recorded.
    pub fn take_recording(&mut self) -> Option<Recording> {
        self.recording.take()
    }

    /// Runs the steps of a recording, one per input, with no frames between them.
    pub fn replay(&mut self, recording: &Recording) {
        for input in recording.steps() {
            self.step(input);
        }
    }
}
