//! Decides which pages should be on the GPU. No GPU code, so all of it can be tested alone.
//!
//! The rule that keeps a partly loaded scene watertight: a page only installs after every page
//! it depends on, and only leaves after every page depending on it. A page's dependencies hold
//! the parents of its clusters, so a parent is always there to draw until its children are.

use std::collections::BTreeSet;

/// In the request array, a page that was seen but occluded: worth keeping, not worth loading.
pub const REQUEST_KEEP: u32 = 1;

/// Frames a page stays safe from eviction after it was last wanted. Long enough that looking
/// around doesn't throw away what's about to be seen again.
pub const PROTECTED_FRAMES: u32 = 30;

/// Frames an unpacked page may wait for a slot, unwanted, before it is thrown away.
const ARRIVED_PATIENCE_FRAMES: u32 = 120;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Absent,
    Fetching,
    /// Data is on the CPU, waiting for dependencies or a slot.
    Arrived,
    Installed(u32),
}

/// A page moved onto the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Install {
    pub page: u32,
    pub slot: u32,
    /// The page that had to leave that slot.
    pub evicted: Option<u32>,
}

/// Tracks every page's state, and owns the allocation of pool slots.
pub struct Residency {
    dependencies: Vec<Vec<u32>>,
    dependants: Vec<Vec<u32>>,
    pinned: u32,
    states: Vec<State>,
    /// How many pages depending on this one are installed.
    installed_dependants: Vec<u32>,
    last_wanted: Vec<u32>,
    free_slots: Vec<u32>,
    arrived: BTreeSet<u32>,
    fetching: u32,
    /// Pages that wanted loading but found no room in the pool at the last plan.
    starved: u32,
    /// What the last plan wanted and could not start yet, most wanted last.
    queue: Vec<u32>,
    /// Slots the last plan found reclaimable, which is slow to work out.
    reclaimable_at_plan: u32,
}

impl Residency {
    /// `dependencies` must only point at lower page ids, which is how scenes are baked. The
    /// first `pinned` pages never leave. There must be slots for them.
    pub fn new(dependencies: Vec<Vec<u32>>, pinned: u32, slots: u32) -> Self {
        assert!(slots >= pinned, "the pool must hold the pinned pages");
        let count = dependencies.len();
        let mut dependants = vec![Vec::new(); count];
        for (page, list) in dependencies.iter().enumerate() {
            for &dependency in list {
                assert!((dependency as usize) < page, "dependencies point coarser");
                dependants[dependency as usize].push(page as u32);
            }
        }
        Self {
            dependencies,
            dependants,
            pinned,
            states: vec![State::Absent; count],
            installed_dependants: vec![0; count],
            last_wanted: vec![0; count],
            free_slots: (0..slots).rev().collect(),
            arrived: BTreeSet::new(),
            fetching: 0,
            starved: 0,
            queue: Vec::new(),
            reclaimable_at_plan: 0,
        }
    }

    pub fn is_installed(&self, page: u32) -> bool {
        matches!(self.states[page as usize], State::Installed(_))
    }

    pub fn installed_count(&self) -> u32 {
        self.states
            .iter()
            .filter(|state| matches!(state, State::Installed(_)))
            .count() as u32
    }

    /// Pages being fetched, or fetched and waiting to install.
    pub fn in_transit(&self) -> u32 {
        self.fetching + self.arrived.len() as u32
    }

    /// Pages that wanted to load at the last plan, but the pool had no room for.
    pub fn starved(&self) -> u32 {
        self.starved
    }

    /// Takes in one frame's request array, and returns the pages to start fetching now, most
    /// wanted first. At most `max_fetching` pages are ever being fetched at once, and it only
    /// fetches what the pool could make room for.
    ///
    /// `requests` holds, per page, 0 if unseen, [`REQUEST_KEEP`] if occluded, and otherwise a
    /// priority, higher for more visible error.
    pub fn plan(&mut self, frame: u32, requests: &[u32], max_fetching: u32) -> Vec<u32> {
        let count = self.states.len();
        let mut priority = vec![0u32; count];
        priority[..count.min(requests.len())]
            .copy_from_slice(&requests[..count.min(requests.len())]);

        // Parents have lower ids, so one pass downwards carries wants all the way up. A parent
        // is at least as wanted as any child, as it must arrive first.
        for page in (0..count).rev() {
            let wanted = priority[page];
            if wanted == 0 {
                continue;
            }
            self.last_wanted[page] = frame;
            for index in 0..self.dependencies[page].len() {
                let dependency = self.dependencies[page][index] as usize;
                priority[dependency] = priority[dependency].max(wanted);
            }
        }

        // Unpacked pages nobody wants any more shouldn't hold a place in the queue forever.
        let stale: Vec<u32> = self
            .arrived
            .iter()
            .copied()
            .filter(|&page| {
                frame.saturating_sub(self.last_wanted[page as usize]) > ARRIVED_PATIENCE_FRAMES
            })
            .collect();
        for page in stale {
            self.arrived.remove(&page);
            self.states[page as usize] = State::Absent;
        }

        let mut wanted: Vec<u32> = (0..count as u32)
            .filter(|&page| {
                self.states[page as usize] == State::Absent
                    && priority[page as usize] > REQUEST_KEEP
            })
            .collect();
        wanted.sort_by_key(|&page| (std::cmp::Reverse(priority[page as usize]), page));

        self.reclaimable_at_plan = self.reclaimable(frame);
        let room = self.room();
        self.starved = (wanted.len() as u32).saturating_sub(room);
        wanted.reverse();
        self.queue = wanted;
        self.refill(max_fetching)
    }

    /// Starts fetching more of what the last plan wanted, as earlier fetches finish. Requests
    /// come back from the GPU only every few frames, and between them the pipeline would
    /// otherwise run dry.
    pub fn refill(&mut self, max_fetching: u32) -> Vec<u32> {
        let can_fetch = self.room().min(max_fetching.saturating_sub(self.fetching));
        let mut started = Vec::new();
        while (started.len() as u32) < can_fetch {
            let Some(page) = self.queue.pop() else {
                break;
            };
            if self.states[page as usize] == State::Absent {
                self.states[page as usize] = State::Fetching;
                started.push(page);
            }
        }
        self.fetching += started.len() as u32;
        started
    }

    /// How many more pages could be brought in without running out of slots.
    fn room(&self) -> u32 {
        (self.free_slots.len() as u32 + self.reclaimable_at_plan).saturating_sub(self.in_transit())
    }

    /// Installed pages that could be evicted, now or once what depends on them has gone.
    fn reclaimable(&self, frame: u32) -> u32 {
        let mut needed = vec![false; self.states.len()];
        let mut count = 0;
        for page in (self.pinned as usize..self.states.len()).rev() {
            if !self.is_installed(page as u32) {
                continue;
            }
            let recent = frame.saturating_sub(self.last_wanted[page]) <= PROTECTED_FRAMES;
            let held = self.dependants[page]
                .iter()
                .any(|&d| needed[d as usize] && self.is_installed(d));
            needed[page] = recent || held;
            if !needed[page] {
                count += 1;
            }
        }
        count
    }

    #[cfg(test)]
    fn mark_fetching(&mut self, page: u32) {
        if self.states[page as usize] == State::Absent {
            self.states[page as usize] = State::Fetching;
            self.fetching += 1;
        }
    }

    /// The data for a page has come. Returns false if the page's data isn't needed any more.
    pub fn arrived(&mut self, page: u32) -> bool {
        if self.states[page as usize] != State::Fetching {
            return false;
        }
        self.fetching -= 1;
        self.states[page as usize] = State::Arrived;
        self.arrived.insert(page);
        true
    }

    /// A fetch failed. The page can be requested again.
    pub fn failed(&mut self, page: u32) {
        if self.states[page as usize] == State::Fetching {
            self.fetching -= 1;
            self.states[page as usize] = State::Absent;
        }
    }

    /// Whether a page's data is still being held for it.
    pub fn is_waiting(&self, page: u32) -> bool {
        self.states[page as usize] == State::Arrived
    }

    /// Finds the next page ready to install, and gives it a slot, evicting if it must. Returns
    /// `None` when nothing can install this frame. Coarsest pages go first.
    pub fn next_install(&mut self, frame: u32) -> Option<Install> {
        let page = self.arrived.iter().copied().find(|&page| {
            self.dependencies[page as usize]
                .iter()
                .all(|&d| self.is_installed(d))
        })?;
        let (slot, evicted) = match self.free_slots.pop() {
            Some(slot) => (slot, None),
            None => {
                let victim = self.choose_victim(frame)?;
                let State::Installed(slot) = self.states[victim as usize] else {
                    unreachable!("victims are installed");
                };
                self.remove(victim);
                (slot, Some(victim))
            }
        };
        self.arrived.remove(&page);
        self.place(page, slot);
        Some(Install {
            page,
            slot,
            evicted,
        })
    }

    /// Installs a page at once, whatever its state. Its dependencies must be installed.
    pub fn install_now(&mut self, page: u32) -> u32 {
        assert!(
            self.dependencies[page as usize]
                .iter()
                .all(|&d| self.is_installed(d)),
            "dependencies install first"
        );
        let slot = self.free_slots.pop().expect("a free slot");
        self.arrived.remove(&page);
        if self.states[page as usize] == State::Fetching {
            self.fetching -= 1;
        }
        self.place(page, slot);
        slot
    }

    fn place(&mut self, page: u32, slot: u32) {
        self.states[page as usize] = State::Installed(slot);
        for index in 0..self.dependencies[page as usize].len() {
            let dependency = self.dependencies[page as usize][index];
            self.installed_dependants[dependency as usize] += 1;
        }
    }

    fn remove(&mut self, page: u32) {
        self.states[page as usize] = State::Absent;
        for index in 0..self.dependencies[page as usize].len() {
            let dependency = self.dependencies[page as usize][index];
            self.installed_dependants[dependency as usize] -= 1;
        }
    }

    /// The least recently wanted page that nothing depends on, unprotected. Ties go to the
    /// finest page, which has the least effect on the image.
    fn choose_victim(&self, frame: u32) -> Option<u32> {
        (self.pinned as usize..self.states.len())
            .filter(|&page| {
                self.is_installed(page as u32)
                    && self.installed_dependants[page] == 0
                    && frame.saturating_sub(self.last_wanted[page]) > PROTECTED_FRAMES
            })
            .min_by_key(|&page| (self.last_wanted[page], std::cmp::Reverse(page)))
            .map(|page| page as u32)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fetches_keep_coming_between_read_backs() {
        // Six independent pages, of which only two can be fetched at once.
        let mut residency = Residency::new(vec![Vec::new(); 6], 0, 6);
        let mut requests = vec![10u32; 6];
        requests[5] = 90;
        let first = residency.plan(1, &requests, 2);
        assert_eq!(first.len(), 2);
        assert_eq!(first[0], 5, "most wanted first");

        assert!(residency.refill(2).is_empty(), "the limit still holds");
        assert!(residency.arrived(first[0]));
        let more = residency.refill(2);
        assert_eq!(more.len(), 1, "a finished fetch makes room for the next");
        assert!(!first.contains(&more[0]));

        // A new plan replaces what was queued, and nothing is fetched twice.
        let next = residency.plan(2, &requests, 8);
        let everything: Vec<u32> = first.iter().chain(&more).chain(&next).copied().collect();
        let mut unique = everything.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), everything.len());
        assert_eq!(unique.len(), 6);
    }

    use super::*;

    /// A small DAG: page 0 is pinned, 1 and 2 hang off it, 3 needs both, 4 needs 2.
    fn dag(slots: u32) -> Residency {
        let mut r = Residency::new(
            vec![vec![], vec![0], vec![0], vec![1, 2], vec![2]],
            1,
            slots,
        );
        r.install_now(0);
        r
    }

    fn want(pages: &[(u32, u32)], count: usize) -> Vec<u32> {
        let mut requests = vec![0; count];
        for &(page, priority) in pages {
            requests[page as usize] = priority;
        }
        requests
    }

    fn install_all(r: &mut Residency, frame: u32) -> Vec<Install> {
        let mut out = Vec::new();
        while let Some(install) = r.next_install(frame) {
            out.push(install);
        }
        out
    }

    #[test]
    fn plans_parents_of_what_is_wanted_most_wanted_first() {
        let mut r = dag(5);
        let fetch = r.plan(1, &want(&[(3, 10), (4, 50)], 5), 16);
        // Page 4 is most wanted, and page 2 must come before it at the same priority. Page 1
        // is only needed by the less wanted page 3.
        assert_eq!(fetch, vec![2, 4, 1, 3]);
    }

    #[test]
    fn nothing_is_fetched_twice_or_past_the_limit() {
        let mut r = dag(5);
        let requests = want(&[(3, 10), (4, 10)], 5);
        assert_eq!(r.plan(1, &requests, 2).len(), 2);
        assert_eq!(
            r.plan(2, &requests, 2).len(),
            0,
            "both slots of the limit are used"
        );
        r.arrived(1);
        let next = r.plan(3, &requests, 2);
        assert_eq!(next.len(), 1);
        assert!(!next.contains(&1));
    }

    #[test]
    fn occluded_pages_are_kept_but_not_loaded() {
        let mut r = dag(5);
        assert!(r.plan(1, &want(&[(1, REQUEST_KEEP)], 5), 16).is_empty());
    }

    #[test]
    fn installs_wait_for_dependencies() {
        let mut r = dag(5);
        for page in r.plan(1, &want(&[(3, 10)], 5), 16) {
            assert!(r.dependencies[page as usize].iter().all(|&d| d != 3));
        }
        assert!(r.arrived(3));
        assert_eq!(r.next_install(1), None, "parents haven't come");
        assert!(r.arrived(1));
        assert_eq!(r.next_install(1).map(|i| i.page), Some(1));
        assert_eq!(r.next_install(1), None, "page 2 is still missing");
        assert!(r.arrived(2));
        let pages: Vec<u32> = install_all(&mut r, 1).iter().map(|i| i.page).collect();
        assert_eq!(pages, vec![2, 3]);
    }

    /// Fills a pool of 3 with pages 0, 1 and 2 at `frame`.
    fn filled(frame: u32) -> Residency {
        let mut r = dag(3);
        let fetch = r.plan(frame, &want(&[(1, 9), (2, 9)], 5), 16);
        for page in fetch {
            r.arrived(page);
        }
        install_all(&mut r, frame);
        r
    }

    #[test]
    fn eviction_only_takes_old_pages_with_no_installed_dependants() {
        let mut r = filled(1);
        // Page 4 needs page 2, so page 1 must leave. Page 2 is still wanted through page 4.
        let later = PROTECTED_FRAMES + 2;
        let fetch = r.plan(later, &want(&[(4, 9)], 5), 16);
        assert_eq!(fetch, vec![4]);
        r.arrived(4);
        let install = r.next_install(later).unwrap();
        assert_eq!(install.page, 4);
        assert_eq!(install.evicted, Some(1));
        assert_eq!(r.installed_dependants[2], 1);
    }

    #[test]
    fn recently_wanted_pages_are_safe_and_the_load_waits() {
        let mut r = filled(1);
        // Page 1 and 2 are both protected, so there is no room to fetch page 4.
        let fetch = r.plan(2, &want(&[(4, 9), (1, 9)], 5), 16);
        assert!(fetch.is_empty());
        assert_eq!(r.starved(), 1);
        assert!(r.is_installed(1));
    }

    #[test]
    fn pinned_pages_and_parents_of_installed_pages_never_leave() {
        let mut r = dag(4);
        for page in r.plan(1, &want(&[(4, 9), (1, 9)], 5), 16) {
            r.arrived(page);
        }
        install_all(&mut r, 1);
        // Pages 0 (pinned), 1, 2 and 4 fill the pool. Far in the future, only leaves can go.
        let frame = 1000;
        for page in r.plan(frame, &want(&[(3, 9)], 5), 16) {
            r.arrived(page);
        }
        // Page 3 needs 1 and 2, so the only page free to leave is 4.
        let install = r.next_install(frame).unwrap();
        assert_eq!((install.page, install.evicted), (3, Some(4)));
        assert!(r.is_installed(0) && r.is_installed(1) && r.is_installed(2));
    }

    #[test]
    fn a_failed_fetch_can_be_tried_again() {
        let mut r = dag(5);
        let requests = want(&[(1, 9)], 5);
        assert_eq!(r.plan(1, &requests, 16), vec![1]);
        r.failed(1);
        assert_eq!(r.plan(2, &requests, 16), vec![1]);
    }

    #[test]
    fn a_page_nobody_wants_stops_waiting_for_a_slot() {
        let mut r = dag(1);
        r.mark_fetching(1);
        assert!(r.arrived(1));
        assert!(r.is_waiting(1));
        r.plan(ARRIVED_PATIENCE_FRAMES + 5, &[0; 5], 16);
        assert!(!r.is_waiting(1));
        assert_eq!(r.in_transit(), 0);
    }

    /// A deterministic random run over a larger DAG: whatever is asked for, the pool never
    /// overflows, and a page is never installed without its dependencies.
    #[test]
    fn random_requests_never_break_the_rules() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let count = 300usize;
        let pinned = 6u32;
        let mut dependencies = vec![Vec::new(); count];
        for (page, list) in dependencies.iter_mut().enumerate().skip(pinned as usize) {
            for _ in 0..1 + next() % 3 {
                list.push((next() % page as u64) as u32);
            }
            list.sort_unstable();
            list.dedup();
        }
        let slots = 40;
        let mut r = Residency::new(dependencies.clone(), pinned, slots);
        for page in 0..pinned {
            r.install_now(page);
        }

        let mut in_flight: Vec<u32> = Vec::new();
        let mut used = pinned;
        for frame in 1..3000 {
            let mut requests = vec![0; count];
            let focus = next() % count as u64;
            for _ in 0..8 {
                let page = (focus + next() % 20).min(count as u64 - 1) as usize;
                requests[page] = if next() % 4 == 0 {
                    REQUEST_KEEP
                } else {
                    2 + (next() % 100) as u32
                };
            }
            in_flight.extend(r.plan(frame, &requests, 8));
            // Some fetches finish, a few fail, in no particular order.
            for _ in 0..3 {
                if !in_flight.is_empty() {
                    let at = (next() % in_flight.len() as u64) as usize;
                    let page = in_flight.swap_remove(at);
                    if next() % 16 == 0 {
                        r.failed(page);
                    } else {
                        r.arrived(page);
                    }
                }
            }
            while let Some(install) = r.next_install(frame) {
                used += 1;
                if install.evicted.is_some() {
                    used -= 1;
                }
                assert!(install.evicted.is_none_or(|page| page >= pinned));
            }
            assert!(used <= slots);
            assert_eq!(used, r.installed_count());
            for page in 0..count as u32 {
                if r.is_installed(page) {
                    for &dependency in &dependencies[page as usize] {
                        assert!(r.is_installed(dependency), "{page} lost {dependency}");
                    }
                }
            }
            for page in 0..pinned {
                assert!(r.is_installed(page));
            }
        }
    }
}
