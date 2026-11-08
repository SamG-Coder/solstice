//! Actors placed in a scene.
//!
//! A scene's text file places an actor as it places models, at x, y, z and a yaw in degrees,
//! with settings for its properties: `actor gate.actor 20.7 0 0 90 locked=true`.
//!
//! The bake carries the scene's gameplay into the scene file as [`solstice_asset::GameFile`]s:
//! each actor file and script, the input actions, and [`GAME_FILE`], which lists the actors in
//! the same form, with the renderer instances the bake placed for each actor's mesh.

use crate::{is_name, lines, number, FileError};
use glam::Vec3;
use std::fmt::Write;
use std::ops::Range;

/// The name of the gameplay file that lists a baked scene's actors and its input actions.
pub const GAME_FILE: &str = "scene.game";

/// One actor placed in a scene.
#[derive(Clone, Debug, PartialEq)]
pub struct ActorLine {
    /// The actor file, relative to the scene's text file.
    pub file: String,
    pub position: Vec3,
    pub yaw_degrees: f32,
    /// Property values for this placement, as written: `name=value`, a vector as `x,y,z`.
    pub settings: Vec<(String, String)>,
    /// The renderer instances drawing its mesh, once baked.
    pub instances: Option<Range<u32>>,
}

impl ActorLine {
    /// Reads the words of an `actor` line, after `actor` itself.
    pub fn parse(words: &[&str]) -> Result<Self, String> {
        if words.len() < 5 {
            return Err("expected `actor FILE X Y Z YAW [NAME=VALUE...]`".to_owned());
        }
        let mut actor = ActorLine {
            file: words[0].to_owned(),
            position: Vec3::new(number(words[1])?, number(words[2])?, number(words[3])?),
            yaw_degrees: number(words[4])?,
            settings: Vec::new(),
            instances: None,
        };
        for word in &words[5..] {
            let (name, value) = word
                .split_once('=')
                .ok_or_else(|| format!("`{word}` should be NAME=VALUE"))?;
            if name == "instances" {
                let range = value
                    .split_once('+')
                    .and_then(|(first, count)| {
                        Some((first.parse::<u32>().ok()?, count.parse::<u32>().ok()?))
                    })
                    .and_then(|(first, count)| Some(first..first.checked_add(count)?))
                    .ok_or_else(|| format!("`{word}` should be instances=FIRST+COUNT"))?;
                actor.instances = Some(range);
                continue;
            }
            if !is_name(name) || value.is_empty() {
                return Err(format!("`{word}` should be NAME=VALUE"));
            }
            if actor.settings.iter().any(|(n, _)| n == name) {
                return Err(format!("`{name}` is set twice"));
            }
            actor.settings.push((name.to_owned(), value.to_owned()));
        }
        Ok(actor)
    }

    /// The line, `actor` and all, which [`ActorLine::parse`] reads back as it is.
    pub fn to_text(&self) -> String {
        let p = self.position;
        let mut text = format!(
            "actor {} {} {} {} {}",
            self.file, p.x, p.y, p.z, self.yaw_degrees
        );
        for (name, value) in &self.settings {
            let _ = write!(text, " {name}={value}");
        }
        if let Some(instances) = &self.instances {
            let _ = write!(text, " instances={}+{}", instances.start, instances.len());
        }
        text
    }
}

/// A baked scene's [`GAME_FILE`]: its actors, and its input actions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneGame {
    pub actors: Vec<ActorLine>,
    /// The `.input` file, relative to the scene's text file, or the built-in actions if none.
    pub input: Option<String>,
}

impl SceneGame {
    pub fn parse(text: &str) -> Result<Self, FileError> {
        let mut game = SceneGame::default();
        for (line, words) in lines(text) {
            let error = |message: String| FileError { line, message };
            match words[0] {
                "actor" => game
                    .actors
                    .push(ActorLine::parse(&words[1..]).map_err(error)?),
                "input" if words.len() == 2 => {
                    if game.input.is_some() {
                        return Err(error("`input` is given twice".to_owned()));
                    }
                    game.input = Some(words[1].to_owned());
                }
                "input" => return Err(error("expected `input FILE`".to_owned())),
                other => return Err(error(format!("unknown line `{other}`"))),
            }
        }
        Ok(game)
    }

    pub fn to_text(&self) -> String {
        let mut text = String::new();
        if let Some(input) = &self.input {
            let _ = writeln!(text, "input {input}");
        }
        for actor in &self.actors {
            text.push_str(&actor.to_text());
            text.push('\n');
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(line: &str) -> Vec<&str> {
        line.split_whitespace().collect()
    }

    #[test]
    fn an_actor_line_reads_back_what_it_wrote() {
        let line =
            ActorLine::parse(&words("gate.actor 20.7 0 -3.25 90 locked=true hinge=0,1,2")).unwrap();
        assert_eq!(line.file, "gate.actor");
        assert_eq!(line.position, Vec3::new(20.7, 0.0, -3.25));
        assert_eq!(line.yaw_degrees, 90.0);
        assert_eq!(
            line.settings,
            vec![
                ("locked".to_owned(), "true".to_owned()),
                ("hinge".to_owned(), "0,1,2".to_owned())
            ]
        );
        assert_eq!(line.instances, None);
        let baked = ActorLine {
            instances: Some(120..124),
            ..line.clone()
        };
        let text = baked.to_text();
        assert_eq!(
            text,
            "actor gate.actor 20.7 0 -3.25 90 locked=true hinge=0,1,2 instances=120+4"
        );
        assert_eq!(ActorLine::parse(&words(&text)[1..]).unwrap(), baked);
    }

    #[test]
    fn mistakes_in_actor_lines_are_explained() {
        let cases = [
            ("gate.actor 1 2 3", "expected `actor FILE X Y Z YAW"),
            ("gate.actor 1 2 x 0", "`x` is not a number"),
            ("gate.actor 1 2 3 0 locked", "should be NAME=VALUE"),
            ("gate.actor 1 2 3 0 locked=", "should be NAME=VALUE"),
            ("gate.actor 1 2 3 0 a=1 a=2", "set twice"),
            ("gate.actor 1 2 3 0 instances=4", "instances=FIRST+COUNT"),
            (
                "gate.actor 1 2 3 0 instances=4294967295+2",
                "instances=FIRST+COUNT",
            ),
        ];
        for (line, part) in cases {
            let error = ActorLine::parse(&words(line)).unwrap_err();
            assert!(error.contains(part), "{line:?}: {error}");
        }
    }

    #[test]
    fn a_scene_game_reads_back_what_it_wrote() {
        let text = "input cove/play.input\nactor cove/gate.actor 20.7 0 0 90 instances=3+2\nactor cove/player.actor 0 0 0 0\n";
        let game = SceneGame::parse(text).unwrap();
        assert_eq!(game.input.as_deref(), Some("cove/play.input"));
        assert_eq!(game.actors.len(), 2);
        assert_eq!(game.to_text(), text);
        let error = SceneGame::parse("actor a 0 0 0 0\nmodel x").unwrap_err();
        assert_eq!(error.line, 2);
        assert_eq!(SceneGame::parse("input a\ninput b").unwrap_err().line, 2);
        assert_eq!(SceneGame::parse("actor a 0 0").unwrap_err().line, 1);
    }
}
