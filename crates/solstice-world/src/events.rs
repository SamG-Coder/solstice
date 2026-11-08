//! Events, and timers that send them.
//!
//! Systems talk through events rather than calling each other. An event is any value; each type
//! has its own queue. An event sent during a tick group is delivered when that group ends, and
//! can be read throughout the next group, and then it is gone. So a handler never runs halfway
//! through another system, and events always arrive in the order they were sent.
//!
//! Timers send an event after a number of steps, once or repeatedly. They count steps, not
//! seconds, so they fire on the same step in every run. A timer's event is delivered at the
//! start of the step it fires on, after what the step before sent, so the step's first group
//! reads it.

use crate::clock::STEPS_PER_SECOND;
use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};

/// Every event queue.
#[derive(Default)]
pub struct Events {
    queues: HashMap<TypeId, Box<dyn Queue>>,
}

trait Queue: Any {
    fn deliver(&mut self);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// One type's events: those sent in this group, and those delivered for it to read.
struct TypedQueue<E> {
    sent: Vec<E>,
    delivered: Vec<E>,
}

impl<E: 'static> Queue for TypedQueue<E> {
    fn deliver(&mut self) {
        // The delivered list's allocation is reused for the next group's sending.
        std::mem::swap(&mut self.sent, &mut self.delivered);
        self.sent.clear();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Events {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sends an event, to be read during the next tick group.
    pub fn send<E: 'static>(&mut self, event: E) {
        self.queue::<E>().sent.push(event);
    }

    /// Delivers an event at once, after those already delivered, to be read during the current
    /// tick group: for timers, which fire as a step starts.
    pub(crate) fn send_now<E: 'static>(&mut self, event: E) {
        self.queue::<E>().delivered.push(event);
    }

    fn queue<E: 'static>(&mut self) -> &mut TypedQueue<E> {
        self.queues
            .entry(TypeId::of::<E>())
            .or_insert_with(|| {
                Box::new(TypedQueue::<E> {
                    sent: Vec::new(),
                    delivered: Vec::new(),
                })
            })
            .as_any_mut()
            .downcast_mut::<TypedQueue<E>>()
            .expect("each type has a queue of its own")
    }

    /// The events of one type delivered for this tick group to read, in the order they were
    /// sent.
    pub fn read<E: 'static>(&self) -> &[E] {
        self.queues
            .get(&TypeId::of::<E>())
            .and_then(|queue| queue.as_any().downcast_ref::<TypedQueue<E>>())
            .map_or(&[], |queue| &queue.delivered)
    }

    /// Ends a tick group: what it sent is delivered, and what it read is gone.
    pub(crate) fn deliver(&mut self) {
        for queue in self.queues.values_mut() {
            queue.deliver();
        }
    }
}

/// Names a timer, to cancel it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimerId(u64);

/// Timers waiting to send their events.
#[derive(Default)]
pub struct Timers {
    next_id: u64,
    /// By the step each fires on, then by when it was set, so ties fire in order.
    waiting: BTreeMap<(u64, TimerId), Timer>,
    /// Where each waiting timer is in `waiting`, to cancel it.
    due: HashMap<TimerId, u64>,
}

struct Timer {
    /// Steps between firings, for one that repeats.
    every: Option<u64>,
    send: Box<dyn FnMut(&mut Events)>,
}

/// The steps that `seconds` covers, rounded up, and at least one.
pub fn steps_for(seconds: f64) -> u64 {
    let steps = (seconds * f64::from(STEPS_PER_SECOND)).ceil();
    if steps.is_finite() && steps >= 1.0 {
        steps as u64
    } else {
        1
    }
}

impl Timers {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sends `event` `steps` steps after step `now`, at least one, so on a later step.
    pub fn after<E: Clone + 'static>(&mut self, now: u64, steps: u64, event: E) -> TimerId {
        self.add(now + steps.max(1), None, event)
    }

    /// Sends `event` every `steps` steps from step `now` on, until cancelled.
    pub fn every<E: Clone + 'static>(&mut self, now: u64, steps: u64, event: E) -> TimerId {
        let steps = steps.max(1);
        self.add(now + steps, Some(steps), event)
    }

    fn add<E: Clone + 'static>(&mut self, due: u64, every: Option<u64>, event: E) -> TimerId {
        let id = TimerId(self.next_id);
        self.next_id += 1;
        let timer = Timer {
            every,
            send: Box::new(move |events| events.send_now(event.clone())),
        };
        self.waiting.insert((due, id), timer);
        self.due.insert(id, due);
        id
    }

    /// Stops a timer. Returns whether it was still waiting.
    pub fn cancel(&mut self, id: TimerId) -> bool {
        match self.due.remove(&id) {
            Some(due) => self.waiting.remove(&(due, id)).is_some(),
            None => false,
        }
    }

    /// How many timers are waiting.
    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }

    /// Delivers the events of every timer due by step `now`, in the order they fall due, and
    /// sets repeating ones again.
    pub(crate) fn fire(&mut self, now: u64, events: &mut Events) {
        while let Some(entry) = self.waiting.first_entry() {
            let (due, id) = *entry.key();
            if due > now {
                break;
            }
            let mut timer = entry.remove();
            (timer.send)(events);
            match timer.every {
                Some(every) => {
                    let next = due + every;
                    self.due.insert(id, next);
                    self.waiting.insert((next, id), timer);
                }
                None => {
                    self.due.remove(&id);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Ping(u32);
    #[derive(Clone, Debug, PartialEq)]
    struct Pong(&'static str);

    #[test]
    fn events_arrive_in_the_next_group_in_order_and_then_are_gone() {
        let mut events = Events::new();
        events.send(Ping(1));
        events.send(Pong("a"));
        events.send(Ping(2));
        assert!(
            events.read::<Ping>().is_empty(),
            "read in the group that sent it"
        );
        events.deliver();
        assert_eq!(events.read::<Ping>(), [Ping(1), Ping(2)]);
        assert_eq!(events.read::<Pong>(), [Pong("a")]);
        // Sent while those are being read: they wait for the group after.
        events.send(Ping(3));
        assert_eq!(events.read::<Ping>().len(), 2);
        events.deliver();
        assert_eq!(events.read::<Ping>(), [Ping(3)]);
        assert!(events.read::<Pong>().is_empty());
        events.deliver();
        assert!(events.read::<Ping>().is_empty());
    }

    #[test]
    fn a_type_never_sent_reads_as_nothing() {
        assert!(Events::new().read::<Ping>().is_empty());
    }

    /// Runs steps from 1 to `last`, firing timers, and lists what each step sent.
    fn run(timers: &mut Timers, last: u64) -> Vec<(u64, Vec<Ping>)> {
        let mut events = Events::new();
        let mut log = Vec::new();
        for step in 1..=last {
            timers.fire(step, &mut events);
            if !events.read::<Ping>().is_empty() {
                log.push((step, events.read::<Ping>().to_vec()));
            }
            events.deliver();
        }
        log
    }

    #[test]
    fn timers_fire_on_their_step_in_the_order_they_were_set() {
        let mut timers = Timers::new();
        timers.after(0, 3, Ping(1));
        timers.after(0, 2, Ping(2));
        timers.after(0, 3, Ping(3));
        // Zero steps still means a later step, not this one.
        timers.after(0, 0, Ping(4));
        assert_eq!(
            run(&mut timers, 5),
            vec![
                (1, vec![Ping(4)]),
                (2, vec![Ping(2)]),
                (3, vec![Ping(1), Ping(3)])
            ]
        );
        assert!(timers.is_empty());
    }

    #[test]
    fn repeating_timers_keep_firing_until_cancelled() {
        let mut timers = Timers::new();
        let every = timers.every(0, 2, Ping(7));
        let once = timers.after(0, 5, Ping(8));
        assert_eq!(
            run(&mut timers, 6),
            vec![
                (2, vec![Ping(7)]),
                (4, vec![Ping(7)]),
                (5, vec![Ping(8)]),
                (6, vec![Ping(7)])
            ]
        );
        assert!(!timers.cancel(once), "a timer that fired is gone");
        assert!(timers.cancel(every));
        assert!(!timers.cancel(every));
        assert!(timers.is_empty());
    }

    #[test]
    fn a_late_firing_catches_up_on_everything_due() {
        let mut timers = Timers::new();
        timers.after(0, 1, Ping(1));
        timers.after(0, 2, Ping(2));
        let mut events = Events::new();
        timers.fire(5, &mut events);
        assert_eq!(events.read::<Ping>(), [Ping(1), Ping(2)]);
    }

    #[test]
    fn a_timer_is_read_after_what_the_last_group_sent() {
        let mut timers = Timers::new();
        timers.after(0, 1, Ping(2));
        let mut events = Events::new();
        events.send(Ping(1));
        events.deliver();
        timers.fire(1, &mut events);
        assert_eq!(events.read::<Ping>(), [Ping(1), Ping(2)]);
    }

    #[test]
    fn seconds_become_whole_steps_rounded_up() {
        assert_eq!(steps_for(1.0), 60);
        assert_eq!(steps_for(0.01), 1);
        assert_eq!(steps_for(0.0), 1);
        assert_eq!(steps_for(-3.0), 1);
        assert_eq!(steps_for(f64::NAN), 1);
        assert_eq!(steps_for(2.5), 150);
    }
}
