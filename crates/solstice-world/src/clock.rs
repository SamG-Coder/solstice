//! The fixed clock: turns frames of any length into whole steps of the same length.
//!
//! Gameplay and physics advance only in steps of [`STEP_SECONDS`], so a run gives the same
//! result at any frame rate, natively and on the web. Each frame adds its time to an
//! accumulator and takes out as many whole steps as it holds. What is left over is how far the
//! frame is between the last step and the next, for drawing things in between.
//!
//! Time is counted in whole microseconds, so that adding up frames never drifts.

/// Steps a second.
pub const STEPS_PER_SECOND: u32 = 60;
/// The length of one step.
pub const STEP_SECONDS: f64 = 1.0 / STEPS_PER_SECOND as f64;
/// The most steps one frame runs. A frame that would need more is cut short, and the game runs
/// slower for it, rather than falling further behind with every frame.
pub const MAX_STEPS_PER_FRAME: u32 = 3;

const MICROS_PER_SECOND: u64 = 1_000_000;
/// Microseconds in a step, rounded: 16,667. Steps are counted, not timed, so the rounding never
/// adds up: a run's simulated time is its step count times [`STEP_SECONDS`].
const STEP_MICROS: u64 =
    (MICROS_PER_SECOND + STEPS_PER_SECOND as u64 / 2) / STEPS_PER_SECOND as u64;

/// Turns frame times into steps.
#[derive(Clone, Debug, Default)]
pub struct FixedClock {
    /// Time received but not yet stepped through, in microseconds.
    pending: u64,
}

/// What one frame of a [`FixedClock`] holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameSteps {
    /// Whole steps to run.
    pub steps: u32,
    /// How far the frame is from the last step towards the next, from 0 to just under 1.
    pub alpha: f32,
    /// Whether time was dropped because the frame needed more than [`MAX_STEPS_PER_FRAME`].
    pub slowed: bool,
}

impl FixedClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a frame of `seconds` and says how many steps it holds. Negative or not-a-number
    /// frame times count as nothing.
    pub fn advance(&mut self, seconds: f64) -> FrameSteps {
        let micros = if seconds.is_finite() && seconds > 0.0 {
            // A frame of over an hour is a paused tab coming back, and is cut short anyway.
            (seconds.min(3600.0) * MICROS_PER_SECOND as f64).round() as u64
        } else {
            0
        };
        self.pending += micros;
        let wanted = self.pending / STEP_MICROS;
        let steps = wanted.min(u64::from(MAX_STEPS_PER_FRAME));
        let slowed = wanted > steps;
        if slowed {
            // Keep only what is left over from the last step that runs: the rest is dropped.
            self.pending %= STEP_MICROS;
        } else {
            self.pending -= steps * STEP_MICROS;
        }
        FrameSteps {
            steps: steps as u32,
            alpha: self.pending as f32 / STEP_MICROS as f32,
            slowed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_of_frames_is_sixty_steps_whatever_their_length() {
        for frame in [1.0 / 30.0, 1.0 / 60.0, 1.0 / 144.0, 1.0 / 75.0] {
            let mut clock = FixedClock::new();
            let frames = (1.0_f64 / frame).round() as u32;
            let steps: u32 = (0..frames).map(|_| clock.advance(frame).steps).sum();
            assert!(
                (59..=60).contains(&steps),
                "{steps} steps in a second of {frame} s frames"
            );
        }
    }

    #[test]
    fn uneven_frames_add_up_without_drift() {
        // An hour of frames that alternate between short and long: none of the time is lost.
        let mut clock = FixedClock::new();
        let mut steps = 0_u64;
        for i in 0..216_000 {
            let frame = if i % 2 == 0 { 0.009 } else { 0.0243 };
            steps += u64::from(clock.advance(frame).steps);
        }
        let seconds = 108_000.0 * (0.009 + 0.0243);
        let expected = seconds * f64::from(STEPS_PER_SECOND) * 16_666.666 / STEP_MICROS as f64;
        assert!(
            (steps as f64 - expected).abs() < 2.0,
            "{steps} against {expected}"
        );
    }

    #[test]
    fn a_slow_frame_runs_at_most_the_limit_and_drops_the_rest() {
        let mut clock = FixedClock::new();
        let frame = clock.advance(0.5);
        assert_eq!(frame.steps, MAX_STEPS_PER_FRAME);
        assert!(frame.slowed);
        // The dropped time doesn't come back as a burst of steps afterwards.
        let next = clock.advance(1.0 / 60.0);
        assert!(next.steps <= 1 && !next.slowed);
    }

    #[test]
    fn alpha_is_how_far_into_the_next_step_the_frame_is() {
        let mut clock = FixedClock::new();
        let frame = clock.advance(STEP_SECONDS * 1.5);
        assert_eq!(frame.steps, 1);
        assert!((frame.alpha - 0.5).abs() < 1.0e-3, "{}", frame.alpha);
        let frame = clock.advance(STEP_SECONDS * 0.25);
        assert_eq!(frame.steps, 0);
        assert!((frame.alpha - 0.75).abs() < 1.0e-3, "{}", frame.alpha);
        assert!(clock.advance(STEP_SECONDS * 0.25).alpha < 1.0e-3);
    }

    #[test]
    fn nonsense_frame_times_count_as_nothing() {
        let mut clock = FixedClock::new();
        for seconds in [-1.0, f64::NAN, f64::NEG_INFINITY, f64::INFINITY, 0.0] {
            assert_eq!(clock.advance(seconds).steps, 0);
        }
        // A huge but real one is cut short.
        assert_eq!(clock.advance(1.0e9).steps, MAX_STEPS_PER_FRAME);
    }
}
