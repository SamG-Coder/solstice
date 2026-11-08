//! What the player is doing, as named actions, and a recording of it, step by step.
//!
//! Keys, buttons and touches are mapped to actions before they reach the world (step 4 of
//! docs/gameplay.md). The world only sees each action's value: 1 for a button held, 0 for
//! released, anything between for a stick or a trigger. Gameplay reads the input of the step it
//! is in, never of the frame, so a recording of each step's input replays a run exactly.

use std::collections::BTreeMap;
use std::sync::Arc;

/// Each action's value for one step. Actions not mentioned are 0.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Input {
    actions: BTreeMap<Arc<str>, f32>,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an action's value. Setting it to 0 forgets it.
    pub fn set(&mut self, action: &str, value: f32) {
        if value == 0.0 {
            self.actions.remove(action);
        } else {
            self.actions.insert(Arc::from(action), value);
        }
    }

    /// An action's value, or 0.
    pub fn value(&self, action: &str) -> f32 {
        self.actions.get(action).copied().unwrap_or(0.0)
    }

    /// Whether an action is held: its value is at least a half.
    pub fn held(&self, action: &str) -> bool {
        self.value(action) >= 0.5
    }

    /// Every action that isn't 0, in name order.
    pub fn actions(&self) -> impl Iterator<Item = (&str, f32)> {
        self.actions.iter().map(|(name, &value)| (&**name, value))
    }
}

/// The input of every step of a run, in order. Runs of steps with the same input are stored
/// once, since input changes far less often than steps go by.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recording {
    /// Each input, with how many steps in a row it lasted.
    runs: Vec<(Input, u64)>,
}

impl Recording {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds the input of the next step.
    pub fn push(&mut self, input: &Input) {
        match self.runs.last_mut() {
            Some((last, count)) if last == input => *count += 1,
            _ => self.runs.push((input.clone(), 1)),
        }
    }

    /// How many steps it holds.
    pub fn len(&self) -> u64 {
        self.runs.iter().map(|(_, count)| count).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Each step's input, in order.
    pub fn steps(&self) -> impl Iterator<Item = &Input> {
        self.runs
            .iter()
            .flat_map(|(input, count)| std::iter::repeat_n(input, *count as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_not_set_are_zero_and_zero_is_forgotten() {
        let mut input = Input::new();
        assert_eq!(input.value("jump"), 0.0);
        input.set("jump", 1.0);
        input.set("throttle", 0.25);
        assert!(input.held("jump") && !input.held("throttle"));
        input.set("jump", 0.0);
        assert_eq!(input, {
            let mut only = Input::new();
            only.set("throttle", 0.25);
            only
        });
        assert_eq!(
            input.actions().collect::<Vec<_>>(),
            vec![("throttle", 0.25)]
        );
    }

    #[test]
    fn a_recording_gives_back_every_step_in_order() {
        let (mut idle, mut walk) = (Input::new(), Input::new());
        idle.set("look", 0.1);
        walk.set("forward", 1.0);
        let steps = [&idle, &idle, &walk, &walk, &walk, &idle];
        let mut recording = Recording::new();
        for input in steps {
            recording.push(input);
        }
        assert_eq!(recording.len(), 6);
        // Three runs, not six inputs.
        assert_eq!(recording.runs.len(), 3);
        assert!(recording.steps().eq(steps));
    }
}
