//! State machines: which clips play, and when they cross-fade into one another.
//!
//! ```text
//! # scenes/harbour/walker.anim
//! source ../../target/characters/keeper_clips.glb     # the glTF the clips are in
//! roles ../../characters/rigs/game_engine.roles       # its bones, for retargeting (optional)
//! clip idle Idle                                      # a name here, and in the file
//! clip walk Walk
//! clip run Run
//! notify walk 0.05 step_l                             # sent to scripts as `on_anim(name)`
//! notify walk 0.55 step_r
//! param speed 0
//! state idle clip idle                                # plays a clip
//! state moving blend1d speed 0 idle 1.4 walk 4.0 run  # blends by a parameter
//! state aiming blend2d dx dy 0 0 idle 1 0 walk 0 1 run  # blends by two
//! start idle
//! transition idle moving speed > 0.1 fade 0.25        # from, to, when, how long
//! transition moving idle speed < 0.1 fade 0.3
//! transition * idle dead == 1 fade 0.1                # from any state
//! ```
//!
//! Every clip of a blend shares one phase, so blended walk and run cycles keep their feet in step
//! (Unreal's sync groups).

use crate::clip::{duration, sample};
use crate::pose::{blend, Pose};
use solstice_asset::Clip;
use std::collections::HashMap;

/// What a state plays.
#[derive(Clone, Debug, PartialEq)]
pub enum Motion {
    Clip(String),
    /// A parameter, and each clip at its place along it.
    Blend1d(String, Vec<(f32, String)>),
    /// Two parameters, and each clip at its place in their plane.
    Blend2d(String, String, Vec<([f32; 2], String)>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compare {
    Less,
    Greater,
    Equal,
    NotEqual,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    pub from: Option<String>,
    pub to: String,
    param: String,
    compare: Compare,
    value: f32,
    pub fade: f32,
}

impl Transition {
    fn holds(&self, value: f32) -> bool {
        match self.compare {
            Compare::Less => value < self.value,
            Compare::Greater => value > self.value,
            Compare::Equal => (value - self.value).abs() < 1.0e-6,
            Compare::NotEqual => (value - self.value).abs() >= 1.0e-6,
        }
    }
}

/// A clip's name in the graph, its name in the source file, and whether it loops.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipLine {
    pub name: String,
    pub source_name: String,
    pub looping: bool,
}

/// A parsed `.anim` file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimGraph {
    /// The glTF the clips are in, relative to this file.
    pub source: Option<String>,
    /// The role file of the source's rig, for retargeting. Without one, bones match by name.
    pub roles: Option<String>,
    pub clips: Vec<ClipLine>,
    pub notifies: Vec<(String, f32, String)>,
    pub params: Vec<(String, f32)>,
    pub states: Vec<(String, Motion)>,
    pub transitions: Vec<Transition>,
    pub start: String,
}

fn number(word: &str) -> Result<f32, String> {
    word.parse()
        .map_err(|_| format!("`{word}` is not a number"))
}

impl AnimGraph {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut graph = AnimGraph::default();
        for (index, line) in text.lines().enumerate() {
            let at = |m: String| format!("line {}: {m}", index + 1);
            let words: Vec<&str> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            let Some((&keyword, rest)) = words.split_first() else {
                continue;
            };
            match (keyword, rest) {
                ("source", [path]) => graph.source = Some((*path).to_owned()),
                ("roles", [path]) => graph.roles = Some((*path).to_owned()),
                ("clip", [name, source] | [name, source, "loop"]) => graph.clips.push(ClipLine {
                    name: (*name).to_owned(),
                    source_name: (*source).to_owned(),
                    looping: true,
                }),
                ("clip", [name, source, "once"]) => graph.clips.push(ClipLine {
                    name: (*name).to_owned(),
                    source_name: (*source).to_owned(),
                    looping: false,
                }),
                ("notify", [clip, time, name]) => graph.notifies.push((
                    (*clip).to_owned(),
                    number(time).map_err(at)?,
                    (*name).to_owned(),
                )),
                ("param", [name, value]) => graph
                    .params
                    .push(((*name).to_owned(), number(value).map_err(at)?)),
                ("state", [name, "clip", clip]) => graph
                    .states
                    .push(((*name).to_owned(), Motion::Clip((*clip).to_owned()))),
                ("state", [name, "blend1d", param, points @ ..]) => {
                    if points.is_empty() || !points.len().is_multiple_of(2) {
                        return Err(at("a blend1d lists position and clip, in pairs".to_owned()));
                    }
                    let mut list = Vec::new();
                    for pair in points.chunks_exact(2) {
                        list.push((number(pair[0]).map_err(at)?, pair[1].to_owned()));
                    }
                    list.sort_by(|a, b| a.0.total_cmp(&b.0));
                    graph.states.push((
                        (*name).to_owned(),
                        Motion::Blend1d((*param).to_owned(), list),
                    ));
                }
                ("state", [name, "blend2d", x, y, points @ ..]) => {
                    if points.is_empty() || !points.len().is_multiple_of(3) {
                        return Err(at("a blend2d lists x, y and clip, in threes".to_owned()));
                    }
                    let mut list = Vec::new();
                    for triple in points.chunks_exact(3) {
                        list.push((
                            [
                                number(triple[0]).map_err(at)?,
                                number(triple[1]).map_err(at)?,
                            ],
                            triple[2].to_owned(),
                        ));
                    }
                    graph.states.push((
                        (*name).to_owned(),
                        Motion::Blend2d((*x).to_owned(), (*y).to_owned(), list),
                    ));
                }
                ("start", [name]) => graph.start = (*name).to_owned(),
                ("transition", [from, to, param, op, value, "fade", fade]) => {
                    let compare = match *op {
                        "<" => Compare::Less,
                        ">" => Compare::Greater,
                        "==" => Compare::Equal,
                        "!=" => Compare::NotEqual,
                        other => return Err(at(format!("`{other}` is not <, >, == or !="))),
                    };
                    graph.transitions.push(Transition {
                        from: (*from != "*").then(|| (*from).to_owned()),
                        to: (*to).to_owned(),
                        param: (*param).to_owned(),
                        compare,
                        value: number(value).map_err(at)?,
                        fade: number(fade).map_err(at)?.max(0.0),
                    });
                }
                _ => return Err(at(format!("didn't understand `{}`", words.join(" ")))),
            }
        }
        graph.check()?;
        Ok(graph)
    }

    fn check(&self) -> Result<(), String> {
        if self.states.is_empty() {
            return Err("a graph needs at least one state".to_owned());
        }
        let has_state = |name: &str| self.states.iter().any(|(n, _)| n == name);
        let has_clip = |name: &str| self.clips.iter().any(|c| c.name == name);
        if !has_state(&self.start) {
            return Err(format!("`start` names no state: `{}`", self.start));
        }
        let mut clip_names = Vec::new();
        for (name, motion) in &self.states {
            match motion {
                Motion::Clip(c) => clip_names.push(c),
                Motion::Blend1d(_, points) => clip_names.extend(points.iter().map(|p| &p.1)),
                Motion::Blend2d(_, _, points) => clip_names.extend(points.iter().map(|p| &p.1)),
            }
            if clip_names.iter().any(|c| !has_clip(c)) {
                return Err(format!("state `{name}` uses a clip no `clip` line names"));
            }
        }
        for transition in &self.transitions {
            if !has_state(&transition.to) || transition.from.as_ref().is_some_and(|f| !has_state(f))
            {
                return Err("a transition names a state that doesn't exist".to_owned());
            }
        }
        for (clip, _, name) in &self.notifies {
            if !has_clip(clip) {
                return Err(format!("notify `{name}` names no clip: `{clip}`"));
            }
        }
        Ok(())
    }
}

/// A graph's clips, resolved to a loaded clip by the graph's own names.
pub struct ClipSet {
    clips: HashMap<String, usize>,
    pub loaded: Vec<Clip>,
}

impl ClipSet {
    pub fn new(names: Vec<(String, Clip)>) -> Self {
        let mut clips = HashMap::new();
        let mut loaded = Vec::with_capacity(names.len());
        for (name, clip) in names {
            clips.insert(name, loaded.len());
            loaded.push(clip);
        }
        Self { clips, loaded }
    }

    fn get(&self, name: &str) -> &Clip {
        &self.loaded[self.clips[name]]
    }

    pub fn contains(&self, name: &str) -> bool {
        self.clips.contains_key(name)
    }
}

/// A state as it runs: which one, and how far through it is, from 0 to 1.
#[derive(Clone, Copy, Debug)]
struct Run {
    state: usize,
    phase: f32,
}

/// One clip of a state and its share of the blend, sorted out from the parameters.
fn shares_of(motion: &Motion, value: &dyn Fn(&str) -> f32) -> Vec<(String, f32)> {
    match motion {
        Motion::Clip(c) => vec![(c.clone(), 1.0)],
        Motion::Blend1d(param, points) => {
            let v = value(param);
            let first = points[0].0;
            let last = points[points.len() - 1].0;
            if v <= first {
                return vec![(points[0].1.clone(), 1.0)];
            }
            if v >= last {
                return vec![(points[points.len() - 1].1.clone(), 1.0)];
            }
            let at = points
                .windows(2)
                .position(|w| v >= w[0].0 && v <= w[1].0)
                .unwrap_or(0);
            let (lo, hi) = (&points[at], &points[at + 1]);
            let t = (v - lo.0) / (hi.0 - lo.0).max(1.0e-6);
            vec![(lo.1.clone(), 1.0 - t), (hi.1.clone(), t)]
        }
        Motion::Blend2d(x, y, points) => {
            let p = [value(x), value(y)];
            let mut weights: Vec<(String, f32)> = points
                .iter()
                .map(|(place, clip)| {
                    let d2 = (place[0] - p[0]).powi(2) + (place[1] - p[1]).powi(2);
                    (clip.clone(), if d2 < 1.0e-8 { 1.0e8 } else { 1.0 / d2 })
                })
                .collect();
            let total: f32 = weights.iter().map(|w| w.1).sum();
            for w in &mut weights {
                w.1 /= total;
            }
            weights
        }
    }
}

/// A graph, running: parameters, the state playing, and any fade under way.
pub struct Animator {
    pub graph: AnimGraph,
    params: Vec<f32>,
    current: Run,
    fade: Option<(Run, f32, f32)>,
    scratch: Pose,
    blended: Pose,
}

impl Animator {
    pub fn new(graph: AnimGraph, bones: usize) -> Self {
        let params = graph.params.iter().map(|p| p.1).collect();
        let start = graph
            .states
            .iter()
            .position(|(n, _)| *n == graph.start)
            .unwrap_or(0);
        Self {
            graph,
            params,
            current: Run {
                state: start,
                phase: 0.0,
            },
            fade: None,
            scratch: Pose::with_bones(bones),
            blended: Pose::with_bones(bones),
        }
    }

    pub fn state_name(&self) -> &str {
        &self.graph.states[self.current.state].0
    }

    /// Sets a parameter by name. Returns false if the graph has no such parameter.
    pub fn set(&mut self, param: &str, value: f32) -> bool {
        match self.graph.params.iter().position(|(n, _)| n == param) {
            Some(index) => {
                self.params[index] = value;
                true
            }
            None => false,
        }
    }

    /// Fades to a state now, whatever the transitions say.
    pub fn go_to(&mut self, state: &str, fade: f32) -> bool {
        match self.graph.states.iter().position(|(n, _)| n == state) {
            Some(index) => {
                self.begin(index, fade);
                true
            }
            None => false,
        }
    }

    fn begin(&mut self, state: usize, fade: f32) {
        if state == self.current.state {
            return;
        }
        self.fade = (fade > 0.0).then_some((self.current, 0.0, fade));
        // A locomotion cycle stays in step across a fade.
        self.current = Run {
            state,
            phase: self.current.phase,
        };
    }

    fn value(&self, name: &str) -> f32 {
        self.graph
            .params
            .iter()
            .position(|(n, _)| n == name)
            .map_or(0.0, |i| self.params[i])
    }

    /// How long a state's cycle lasts now: its clips' lengths, by their shares, so a blend's feet
    /// keep the blended pace.
    fn cycle_seconds(&self, clips: &ClipSet, state: usize) -> f32 {
        let value = |name: &str| self.value(name);
        let shares = shares_of(&self.graph.states[state].1, &value);
        shares
            .iter()
            .map(|(clip, share)| duration(clips.get(clip)) * share)
            .sum::<f32>()
            .max(1.0e-3)
    }

    /// Moves on by `seconds`: takes any transition that holds, advances the phases, and returns
    /// the notifies the dominant clip crossed.
    pub fn advance(&mut self, seconds: f32, clips: &ClipSet) -> Vec<String> {
        let here = self.graph.states[self.current.state].0.clone();
        for transition in &self.graph.transitions.clone() {
            let from_here = transition.from.as_ref().is_none_or(|f| *f == here);
            if from_here && transition.to != here && transition.holds(self.value(&transition.param))
            {
                let to = self
                    .graph
                    .states
                    .iter()
                    .position(|(n, _)| *n == transition.to)
                    .expect("checked when parsed");
                self.begin(to, transition.fade);
                break;
            }
        }

        if let Some((run, elapsed, length)) = self.fade {
            let cycle = self.cycle_seconds(clips, run.state);
            let run = Run {
                state: run.state,
                phase: run.phase + seconds / cycle,
            };
            let elapsed = elapsed + seconds;
            self.fade = (elapsed < length).then_some((run, elapsed, length));
        }

        let cycle = self.cycle_seconds(clips, self.current.state);
        let before = self.current.phase;
        let mut after = before + seconds / cycle;
        let dominant = self.dominant_clip();
        let clip = clips.get(&dominant);
        let notifies = self.crossed(clip, &dominant, before, after);
        if clip.loops {
            after %= 1.0;
        } else {
            after = after.min(1.0);
        }
        self.current.phase = after;
        notifies
    }

    fn dominant_clip(&self) -> String {
        let value = |name: &str| self.value(name);
        shares_of(&self.graph.states[self.current.state].1, &value)
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(clip, _)| clip)
            .expect("states have at least one clip")
    }

    fn crossed(&self, clip: &Clip, name: &str, before: f32, after: f32) -> Vec<String> {
        let length = duration(clip).max(1.0e-3);
        let mut out = Vec::new();
        for (clip_name, time, notify) in &self.graph.notifies {
            if clip_name != name {
                continue;
            }
            let at = time / length;
            let hit = (at >= before && at < after)
                || (clip.loops && after >= 1.0 && at + 1.0 >= before && at + 1.0 < after);
            if hit {
                out.push(notify.clone());
            }
        }
        out
    }

    /// The current pose, into `out`.
    pub fn evaluate(&mut self, clips: &ClipSet, rest: &Pose, out: &mut Pose) {
        let fade = self.fade;
        let graph = std::mem::take(&mut self.graph);
        let params = std::mem::take(&mut self.params);
        let value = |name: &str| {
            graph
                .params
                .iter()
                .position(|(n, _)| n == name)
                .map_or(0.0, |i| params[i])
        };
        let current = shares_of(&graph.states[self.current.state].1, &value);
        Self::pose_of(
            &current,
            self.current.phase,
            clips,
            rest,
            out,
            &mut self.scratch,
        );
        if let Some((run, elapsed, length)) = fade {
            let older = shares_of(&graph.states[run.state].1, &value);
            let t = (elapsed / length).clamp(0.0, 1.0);
            Self::pose_of(
                &older,
                run.phase,
                clips,
                rest,
                &mut self.blended,
                &mut self.scratch,
            );
            // From the old state towards the new one.
            let finished = out.clone();
            out.copy_from(&self.blended);
            blend(out, &finished, t, None);
        }
        self.params = params;
        self.graph = graph;
    }

    fn pose_of(
        shares: &[(String, f32)],
        phase: f32,
        clips: &ClipSet,
        rest: &Pose,
        out: &mut Pose,
        scratch: &mut Pose,
    ) {
        out.copy_from(rest);
        let mut filled = 0.0f32;
        for (clip, share) in shares {
            if *share <= 0.0 {
                continue;
            }
            let c = clips.get(clip);
            sample(c, phase * duration(c), scratch);
            let t = share / (filled + share);
            if filled == 0.0 {
                out.copy_from(scratch);
            } else {
                blend(out, scratch, t, None);
            }
            filled += share;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{compress, Frames};
    use glam::{Quat, Vec3};

    const GRAPH: &str = "
        clip idle I
        clip walk W
        clip run R
        notify walk 0.5 step
        param speed 0
        state idle clip idle
        state moving blend1d speed 0 idle 2 walk 4 run
        start idle
        transition idle moving speed > 0.1 fade 0.2
        transition moving idle speed < 0.1 fade 0.2
    ";

    fn clip(name: &str, seconds: u32, height: f32) -> Clip {
        let frames = (seconds as usize) * 30;
        compress(
            name,
            0,
            &Frames {
                rotations: vec![vec![Quat::IDENTITY]; frames],
                translations: (0..frames)
                    .map(|i| vec![Vec3::Y * height * (i as f32 / frames as f32)])
                    .collect(),
            },
            true,
        )
    }

    fn set() -> ClipSet {
        ClipSet::new(vec![
            ("idle".to_owned(), clip("idle", 2, 0.0)),
            ("walk".to_owned(), clip("walk", 1, 2.0)),
            ("run".to_owned(), clip("run", 1, 4.0)),
        ])
    }

    #[test]
    fn it_parses_and_refuses_mistakes() {
        let graph = AnimGraph::parse(GRAPH).unwrap();
        assert_eq!(graph.states.len(), 2);
        for (bad, says) in [
            ("clip a A\nstate a clip b\nstart a", "uses a clip"),
            ("clip a A\nstate a clip a\nstart b", "start"),
            ("state a blend1d x 1\nstart a", "pairs"),
            ("state", "understand"),
            (
                "clip a A\nstate a clip a\nstart a\ntransition a b x > 1 fade 0",
                "doesn't exist",
            ),
        ] {
            let error = AnimGraph::parse(bad).unwrap_err();
            assert!(error.contains(says), "{error}");
        }
    }

    #[test]
    fn a_blend_space_shares_by_where_the_parameter_lies() {
        let graph = AnimGraph::parse(GRAPH).unwrap();
        let (_, motion) = &graph.states[1];
        let value = |v: f32| shares_of(motion, &move |_: &str| v);
        assert_eq!(value(0.0), [("idle".to_owned(), 1.0)]);
        let half = value(3.0);
        assert_eq!(half.len(), 2);
        assert!((half[0].1 - 0.5).abs() < 1.0e-5 && half[0].0 == "walk");
        assert_eq!(value(99.0), [("run".to_owned(), 1.0)]);
    }

    #[test]
    fn a_2d_space_goes_wholly_to_the_clip_at_that_spot() {
        let graph = AnimGraph::parse(
            "clip a A\nclip b B\nparam x 0\nparam y 0\nstate s blend2d x y 0 0 a 1 0 b\nstart s",
        )
        .unwrap();
        let (_, motion) = &graph.states[0];
        let at =
            |p: [f32; 2]| shares_of(motion, &move |n: &str| if n == "x" { p[0] } else { p[1] });
        assert!(at([1.0, 0.0]).iter().any(|(c, w)| c == "b" && *w > 0.999));
        let middle = at([0.5, 0.0]);
        assert!((middle[0].1 - 0.5).abs() < 1.0e-3);
    }

    #[test]
    fn a_transition_fades_and_the_phases_stay_in_step() {
        let clips = set();
        let mut animator = Animator::new(AnimGraph::parse(GRAPH).unwrap(), 1);
        assert_eq!(animator.state_name(), "idle");
        animator.advance(0.1, &clips);
        animator.set("speed", 2.0);
        animator.advance(0.0, &clips);
        assert_eq!(animator.state_name(), "moving");
        assert!(animator.fade.is_some());
        let rest = Pose::with_bones(1);
        let mut pose = Pose::with_bones(1);
        animator.evaluate(&clips, &rest, &mut pose);
        animator.advance(0.5, &clips);
        assert!(animator.fade.is_none(), "the fade should be finished");
        animator.evaluate(&clips, &rest, &mut pose);
        assert!((pose.translations[0].y - 2.0 * animator.current.phase).abs() < 0.01);
    }

    #[test]
    fn notifies_fire_once_as_their_time_is_crossed() {
        let clips = set();
        let mut animator = Animator::new(AnimGraph::parse(GRAPH).unwrap(), 1);
        animator.set("speed", 2.0);
        animator.advance(0.0, &clips);
        let mut count = 0;
        for _ in 0..120 {
            count += animator.advance(1.0 / 30.0, &clips).len();
        }
        assert_eq!(count, 4, "four cycles of a 1 s clip, one notify each");
    }

    #[test]
    fn a_blend_keeps_the_pace_of_its_shares() {
        let clips = set();
        let mut animator = Animator::new(AnimGraph::parse(GRAPH).unwrap(), 1);
        animator.set("speed", 1.0);
        animator.advance(0.0, &clips);
        // Half idle (2 s), half walk (1 s).
        assert!((animator.cycle_seconds(&clips, 1) - 1.5).abs() < 1.0e-3);
    }
}
