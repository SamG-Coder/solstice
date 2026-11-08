//! The world: every entity and its components, with the events, timers and input of the step
//! being run.

use crate::clock::STEP_SECONDS;
use crate::events::{steps_for, Events, TimerId, Timers};
use crate::input::Input;
use hecs::{CommandBuffer, Entity};

/// Everything gameplay works on.
///
/// Systems read and change components through [`Self::entities`] directly. Spawning, despawning,
/// and adding or removing components while a tick group runs go through [`Self::commands`]
/// instead, and happen when the group ends, so no system sees the world change shape under it.
pub struct World {
    /// Every entity and its components.
    pub entities: hecs::World,
    events: Events,
    timers: Timers,
    commands: CommandBuffer,
    /// Fixed steps run so far.
    step: u64,
    input: Input,
    /// Whether a per-frame tick group is running, which must not send events or set timers.
    in_frame_group: bool,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            entities: hecs::World::new(),
            events: Events::new(),
            timers: Timers::new(),
            commands: CommandBuffer::new(),
            step: 0,
            input: Input::new(),
            in_frame_group: false,
        }
    }

    /// Spawns an entity at once. While a tick group runs, use [`Self::commands`].
    pub fn spawn(&mut self, components: impl hecs::DynamicBundle) -> Entity {
        self.entities.spawn(components)
    }

    /// Changes to make when the current tick group ends: spawning, despawning, and adding or
    /// removing components.
    pub fn commands(&mut self) -> &mut CommandBuffer {
        &mut self.commands
    }

    /// Sends an event, to be read during the next tick group.
    ///
    /// Only the fixed-step groups send events. The per-frame groups run as often as frames
    /// come, which differs from run to run, so anything they sent would make runs differ.
    pub fn send<E: 'static>(&mut self, event: E) {
        debug_assert!(
            !self.in_frame_group,
            "per-frame tick groups don't send events"
        );
        self.events.send(event);
    }

    /// The events of one type delivered for this tick group.
    pub fn read<E: 'static>(&self) -> &[E] {
        self.events.read()
    }

    /// Sends `event` after `seconds`, rounded up to whole steps.
    pub fn after<E: Clone + 'static>(&mut self, seconds: f64, event: E) -> TimerId {
        debug_assert!(
            !self.in_frame_group,
            "per-frame tick groups don't set timers"
        );
        self.timers.after(self.step, steps_for(seconds), event)
    }

    /// Sends `event` every `seconds`, rounded up to whole steps, until cancelled.
    pub fn every<E: Clone + 'static>(&mut self, seconds: f64, event: E) -> TimerId {
        debug_assert!(
            !self.in_frame_group,
            "per-frame tick groups don't set timers"
        );
        self.timers.every(self.step, steps_for(seconds), event)
    }

    /// Stops a timer. Returns whether it was still waiting.
    pub fn cancel(&mut self, timer: TimerId) -> bool {
        debug_assert!(!self.in_frame_group, "per-frame tick groups don't cancel timers");
        self.timers.cancel(timer)
    }

    /// The input of the step being run.
    pub fn input(&self) -> &Input {
        &self.input
    }

    /// Fixed steps run so far.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Simulated seconds so far: the steps run, times the step's length.
    pub fn seconds(&self) -> f64 {
        self.step as f64 * STEP_SECONDS
    }

    /// Starts a step: takes its input, and fires the timers due on it.
    pub(crate) fn begin_step(&mut self, input: &Input) {
        self.step += 1;
        self.input.clone_from(input);
        self.timers.fire(self.step, &mut self.events);
    }

    /// Ends a fixed-step tick group: applies its commands, and delivers its events.
    pub(crate) fn end_step_group(&mut self) {
        self.commands.run_on(&mut self.entities);
        self.events.deliver();
    }

    /// Starts or ends a per-frame tick group. Ending one applies its commands. Events stay as
    /// they are: a per-frame group can read what the last step delivered, as the next step's
    /// first group will.
    pub(crate) fn set_in_frame_group(&mut self, in_frame_group: bool) {
        if !in_frame_group {
            self.commands.run_on(&mut self.entities);
        }
        self.in_frame_group = in_frame_group;
    }
}
