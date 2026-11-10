//! Timing laps.
//!
//! A lap is timed from the start line round to it again, and counts only if every sector's
//! line was crossed in order on the way: a car that turns round, or cuts across the infield,
//! doesn't set a time. A lap on which the car left the track with all four wheels is still
//! timed, and marked as not valid, as race control would.

use crate::track::Track;
use glam::Vec3;

/// Something that happened on a lap this step.
#[derive(Clone, Debug, PartialEq)]
pub enum LapEvent {
    /// Crossed the start line for the first time: the first timed lap begins.
    Started,
    /// Finished a sector, in this many seconds, and how that compares with the best so far.
    Sector {
        sector: usize,
        seconds: f64,
        best: Option<f64>,
    },
    /// Finished a lap.
    Finished {
        lap: u32,
        seconds: f64,
        valid: bool,
        /// The best valid lap before this one.
        best: Option<f64>,
    },
}

/// One car's laps.
#[derive(Clone, Debug, PartialEq)]
pub struct Lap {
    /// Laps finished.
    pub count: u32,
    /// Seconds into the lap being driven, once the start line has been crossed.
    pub current: Option<f64>,
    pub last: Option<f64>,
    /// The best valid lap.
    pub best: Option<f64>,
    /// Whether the lap being driven still counts.
    pub valid: bool,
    /// Whether the last lap finished did.
    pub last_valid: bool,
    /// The sector the car is in, from 0.
    pub sector: usize,
    /// This lap's sector times so far, and the best there has been of each.
    pub splits: Vec<f64>,
    pub best_splits: Vec<Option<f64>>,
    /// How far round the lap the car is, in metres from the start line.
    pub along: f32,
    /// The line to cross next: 0 is the start line, then each sector's.
    next: usize,
    /// When the sector being driven began, in seconds into the lap.
    sector_began: f64,
    /// The sample of the track the car was last nearest.
    hint: Option<usize>,
    before: Option<f32>,
}

/// Further than this between two steps is not driving: the car was put somewhere.
const JUMP: f32 = 40.0;

impl Lap {
    pub fn new(track: &Track) -> Self {
        Self {
            count: 0,
            current: None,
            last: None,
            best: None,
            valid: true,
            last_valid: true,
            sector: 0,
            splits: Vec::new(),
            best_splits: vec![None; track.sectors.len() + 1],
            along: 0.0,
            next: 0,
            sector_began: 0.0,
            hint: None,
            before: None,
        }
    }

    /// The car was put back on the track: the lap being driven no longer counts.
    pub fn interrupt(&mut self) {
        self.valid = false;
        self.before = None;
        self.hint = None;
    }

    /// Follows the car through one step of `seconds`. `off_track` is whether it has all four
    /// wheels off the track.
    pub fn step(
        &mut self,
        track: &Track,
        position: Vec3,
        off_track: bool,
        seconds: f64,
    ) -> Vec<LapEvent> {
        let mut events = Vec::new();
        let place = track.locate(position, self.hint);
        self.hint = Some(place.sample);
        self.along = place.along;
        if let Some(current) = &mut self.current {
            *current += seconds;
        }
        if off_track {
            self.valid = false;
        }
        let Some(before) = self.before.replace(place.along) else {
            return events;
        };
        let length = track.length();
        // How far it went, the short way round: forwards is above zero.
        let moved = (place.along - before + length * 0.5).rem_euclid(length) - length * 0.5;
        if moved <= 0.0 || moved > JUMP {
            return events;
        }
        let lines = track.sector_starts();
        // Whether a line at `line` metres lies in the stretch just driven: after where the car
        // was, and up to where it is. A car that stops exactly on a line has crossed it, once.
        let crossed = |line: f32| {
            let ahead = (line - before).rem_euclid(length);
            ahead > 0.0 && ahead <= moved
        };
        if !crossed(lines[self.next]) {
            return events;
        }
        if self.next == 0 {
            if let Some(current) = self.current {
                self.finish_sector(current, &mut events);
                events.push(LapEvent::Finished {
                    lap: self.count + 1,
                    seconds: current,
                    valid: self.valid,
                    best: self.best,
                });
                self.count += 1;
                self.last = Some(current);
                self.last_valid = self.valid;
                if self.valid && self.best.is_none_or(|best| current < best) {
                    self.best = Some(current);
                }
            } else {
                events.push(LapEvent::Started);
            }
            self.current = Some(0.0);
            self.valid = !off_track;
            self.splits.clear();
            self.sector = 0;
            self.sector_began = 0.0;
        } else if let Some(current) = self.current {
            self.finish_sector(current, &mut events);
            self.sector = self.next;
            self.sector_began = current;
        }
        self.next = (self.next + 1) % lines.len();
        events
    }

    fn finish_sector(&mut self, now: f64, events: &mut Vec<LapEvent>) {
        let seconds = now - self.sector_began;
        let best = self.best_splits[self.sector];
        events.push(LapEvent::Sector {
            sector: self.sector,
            seconds,
            best,
        });
        self.splits.push(seconds);
        if self.valid && best.is_none_or(|best| seconds < best) {
            self.best_splits[self.sector] = Some(seconds);
        }
    }
}

/// A lap time as a timing screen shows it: `1:23.456`.
pub fn format_time(seconds: f64) -> String {
    let thousandths = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{}:{:02}.{:03}",
        thousandths / 60_000,
        thousandths / 1000 % 60,
        thousandths % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::tests::oval;

    const DT: f64 = 1.0 / 60.0;

    /// Drives the centre line from `from` to `to` metres at a speed, offset to the left.
    fn drive(lap: &mut Lap, track: &Track, from: f32, to: f32, speed: f32) -> Vec<LapEvent> {
        let mut events = Vec::new();
        let step = speed * DT as f32;
        let mut along = from;
        while (step > 0.0 && along < to) || (step < 0.0 && along > to) {
            along += step;
            let (centre, _, _) = track.place(along);
            events.extend(lap.step(track, centre, false, DT));
        }
        events
    }

    #[test]
    fn a_lap_is_timed_from_the_line_to_the_line() {
        let track = oval();
        let length = track.length();
        let mut lap = Lap::new(&track);
        // From behind the line: the lap starts as it is crossed.
        let events = drive(&mut lap, &track, -20.0, 5.0, 50.0);
        assert_eq!(events, vec![LapEvent::Started]);
        assert!(lap.current.is_some() && lap.count == 0);
        let events = drive(&mut lap, &track, 5.0, length + 5.0, 50.0);
        let finished: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                LapEvent::Finished {
                    lap,
                    seconds,
                    valid,
                    best,
                } => Some((*lap, *seconds, *valid, *best)),
                _ => None,
            })
            .collect();
        assert_eq!(finished.len(), 1);
        let (number, seconds, valid, best) = finished[0];
        let expected = f64::from(length) / 50.0;
        assert!(
            (seconds - expected).abs() < 0.03,
            "{seconds} against {expected}"
        );
        assert!(number == 1 && valid && best.is_none());
        assert_eq!(lap.best, lap.last);
        assert_eq!(lap.count, 1);
        // Three sectors were timed, and they add up to the lap.
        let sectors: Vec<f64> = events
            .iter()
            .filter_map(|event| match event {
                LapEvent::Sector { seconds, .. } => Some(*seconds),
                _ => None,
            })
            .collect();
        assert_eq!(sectors.len(), 3);
        assert!((sectors.iter().sum::<f64>() - seconds).abs() < 1.0e-9);
        // A slower lap is the last, not the best.
        drive(&mut lap, &track, 5.0, length + 5.0, 40.0);
        assert!(lap.last.unwrap() > lap.best.unwrap() && lap.count == 2);
    }

    #[test]
    fn going_back_over_the_line_and_forward_again_is_no_lap() {
        let track = oval();
        let mut lap = Lap::new(&track);
        drive(&mut lap, &track, -20.0, 30.0, 50.0);
        let events = drive(&mut lap, &track, 30.0, -30.0, -50.0);
        assert!(events.is_empty());
        let events = drive(&mut lap, &track, -30.0, 30.0, 50.0);
        assert!(events.is_empty(), "{events:?}");
        assert_eq!(lap.count, 0);
    }

    #[test]
    fn leaving_the_track_spoils_the_lap_but_not_the_next() {
        let track = oval();
        let length = track.length();
        let mut lap = Lap::new(&track);
        drive(&mut lap, &track, -20.0, 100.0, 50.0);
        let (centre, _, _) = track.place(100.0);
        lap.step(&track, centre, true, DT);
        assert!(!lap.valid);
        drive(&mut lap, &track, 100.0, length + 5.0, 50.0);
        assert!(lap.count == 1 && !lap.last_valid && lap.best.is_none() && lap.valid);
        drive(&mut lap, &track, 5.0, length + 5.0, 50.0);
        assert!(lap.last_valid && lap.best.is_some());
    }

    #[test]
    fn being_put_back_on_the_track_is_not_driving_there() {
        let track = oval();
        let mut lap = Lap::new(&track);
        drive(&mut lap, &track, -20.0, 30.0, 50.0);
        // Put down just before the line, having skipped the lap: crossing it is no lap.
        lap.interrupt();
        let events = drive(
            &mut lap,
            &track,
            track.length() - 20.0,
            track.length() + 20.0,
            50.0,
        );
        assert!(events.is_empty(), "{events:?}");
        assert_eq!(lap.count, 0);
    }

    #[test]
    fn times_are_written_as_a_timing_screen_does() {
        assert_eq!(format_time(83.4564), "1:23.456");
        assert_eq!(format_time(59.9996), "1:00.000");
        assert_eq!(format_time(0.0), "0:00.000");
        assert_eq!(format_time(605.05), "10:05.050");
    }
}
