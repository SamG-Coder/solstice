//! Actor files: a kind of actor, in the style of the `.ocean` and `.weather` files.
//!
//! ```text
//! # scenes/cove/gate.actor
//! name gate
//! mesh ../../target/scan/large_iron_gate/large_iron_gate.gltf 1.15   # a model, and its scale
//! script gate.star
//! property open_seconds float 2.5      # editable per placement in the scene
//! property locked bool false
//! camera 0 1.6 0                       # the player's eye, for the actor the player plays
//! ```
//!
//! Paths are relative to the actor file. The mesh is placed by the bake, wherever the scene
//! places the actor, and moves with it. Every line but `name` may be left out.

use crate::{is_name, lines, number, FileError};
use glam::Vec3;
use solstice_script::ScriptValue;
use std::fmt::Write;

/// What a property holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    Bool,
    Int,
    Float,
    /// One word of text.
    Text,
    Vec3,
}

impl PropertyKind {
    const ALL: [(Self, &'static str); 5] = [
        (Self::Bool, "bool"),
        (Self::Int, "int"),
        (Self::Float, "float"),
        (Self::Text, "text"),
        (Self::Vec3, "vec3"),
    ];

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().find(|(_, n)| *n == name).map(|(k, _)| *k)
    }

    pub fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(k, _)| *k == self)
            .expect("listed")
            .1
    }

    /// Reads a value of this kind from its words: one, or three for a vector.
    pub fn parse(self, words: &[&str]) -> Result<ScriptValue, String> {
        let expected = if self == Self::Vec3 { 3 } else { 1 };
        if words.len() != expected {
            return Err(format!(
                "a {} takes {expected} value{}, not {}",
                self.name(),
                if expected == 1 { "" } else { "s" },
                words.len()
            ));
        }
        let word = words[0];
        Ok(match self {
            Self::Bool => match word {
                "true" => ScriptValue::Bool(true),
                "false" => ScriptValue::Bool(false),
                _ => return Err(format!("a bool is true or false, not `{word}`")),
            },
            Self::Int => ScriptValue::Int(
                word.parse()
                    .map_err(|_| format!("`{word}` is not a whole number"))?,
            ),
            Self::Float => ScriptValue::Float(f64::from(number(word)?)),
            Self::Text => ScriptValue::from(word),
            Self::Vec3 => ScriptValue::Vec3(Vec3::new(
                number(words[0])?,
                number(words[1])?,
                number(words[2])?,
            )),
        })
    }

    /// Reads a value of this kind as a scene line writes it: one word, with a vector's three
    /// numbers joined by commas.
    pub fn parse_setting(self, word: &str) -> Result<ScriptValue, String> {
        if self == Self::Vec3 {
            self.parse(&word.split(',').collect::<Vec<_>>())
        } else {
            self.parse(&[word])
        }
    }

    /// A value's words, as [`PropertyKind::parse`] reads them.
    pub fn words(value: &ScriptValue) -> Vec<String> {
        match value {
            ScriptValue::Bool(b) => vec![b.to_string()],
            ScriptValue::Int(i) => vec![i.to_string()],
            // Shortest form that reads back as the same f32.
            ScriptValue::Float(f) => vec![(*f as f32).to_string()],
            ScriptValue::Str(s) => vec![s.to_string()],
            ScriptValue::Vec3(v) => v.to_array().iter().map(f32::to_string).collect(),
            other => vec![format!("{other:?}")],
        }
    }
}

/// A property an actor's script can read and set, with the value each placement starts from
/// unless the scene says otherwise.
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    pub name: String,
    pub kind: PropertyKind,
    pub default: ScriptValue,
}

/// A model drawn wherever the actor is, as the bake imports it.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    /// A glTF file, relative to the actor file.
    pub path: String,
    pub scale: f32,
}

/// One kind of actor.
#[derive(Clone, Debug, PartialEq)]
pub struct ActorFile {
    /// Each placed actor's [`solstice_world::Name`], which scripts `find` it by.
    pub name: String,
    pub mesh: Option<Mesh>,
    /// A Starlark script, relative to the actor file.
    pub script: Option<String>,
    pub properties: Vec<Property>,
    /// Where the player's eye is, relative to the actor, if this is the actor the player plays.
    pub camera: Option<Vec3>,
}

/// Names `self` already has in a script, which a property can't take.
const RESERVED: [&str; 5] = ["entity", "position", "yaw", "vars", "instances"];

impl ActorFile {
    pub fn parse(text: &str) -> Result<Self, FileError> {
        let mut name = None;
        let mut actor = ActorFile {
            name: String::new(),
            mesh: None,
            script: None,
            properties: Vec::new(),
            camera: None,
        };
        for (line, words) in lines(text) {
            let error = |message: String| FileError { line, message };
            let (keyword, values) = (words[0], &words[1..]);
            let count = |range: std::ops::RangeInclusive<usize>, form: &str| {
                if range.contains(&values.len()) {
                    Ok(())
                } else {
                    Err(error(format!("expected `{keyword} {form}`")))
                }
            };
            let once = |taken: bool| {
                if taken {
                    Err(error(format!("`{keyword}` is given twice")))
                } else {
                    Ok(())
                }
            };
            match keyword {
                "name" => {
                    count(1..=1, "NAME")?;
                    once(name.is_some())?;
                    if !is_name(values[0]) {
                        return Err(error(format!(
                            "`{}` can't be a name: use letters, digits and _",
                            values[0]
                        )));
                    }
                    name = Some(values[0].to_owned());
                }
                "mesh" => {
                    count(1..=2, "PATH [SCALE]")?;
                    once(actor.mesh.is_some())?;
                    let scale = values
                        .get(1)
                        .map_or(Ok(1.0), |s| number(s))
                        .map_err(error)?;
                    if scale <= 0.0 {
                        return Err(error(format!(
                            "a mesh's scale must be above 0, not {scale}"
                        )));
                    }
                    actor.mesh = Some(Mesh {
                        path: values[0].to_owned(),
                        scale,
                    });
                }
                "script" => {
                    count(1..=1, "PATH")?;
                    once(actor.script.is_some())?;
                    actor.script = Some(values[0].to_owned());
                }
                "property" => {
                    if values.len() < 3 {
                        return Err(error("expected `property NAME KIND VALUE`".to_owned()));
                    }
                    let property = values[0];
                    if !is_name(property) || RESERVED.contains(&property) {
                        return Err(error(format!("`{property}` can't be a property's name")));
                    }
                    if actor.properties.iter().any(|p| p.name == property) {
                        return Err(error(format!("property `{property}` is given twice")));
                    }
                    let kind = PropertyKind::from_name(values[1]).ok_or_else(|| {
                        error(format!(
                            "`{}` is not a kind of property: use bool, int, float, text or vec3",
                            values[1]
                        ))
                    })?;
                    let default = kind
                        .parse(&values[2..])
                        .map_err(|e| error(format!("property `{property}`: {e}")))?;
                    actor.properties.push(Property {
                        name: property.to_owned(),
                        kind,
                        default,
                    });
                }
                "camera" => {
                    count(3..=3, "X Y Z")?;
                    once(actor.camera.is_some())?;
                    let [x, y, z] = [0, 1, 2].map(|i| number(values[i]));
                    actor.camera = Some(Vec3::new(
                        x.map_err(error)?,
                        y.map_err(error)?,
                        z.map_err(error)?,
                    ));
                }
                other => {
                    return Err(error(format!(
                        "unknown line `{other}`: an actor file has name, mesh, script, property \
                         and camera lines"
                    )))
                }
            }
        }
        actor.name = name.ok_or(FileError {
            line: 1,
            message: "an actor file needs a `name` line".to_owned(),
        })?;
        Ok(actor)
    }

    /// The file's text, which [`ActorFile::parse`] reads back as it is.
    pub fn to_text(&self) -> String {
        let mut text = format!("name {}\n", self.name);
        if let Some(mesh) = &self.mesh {
            let _ = writeln!(text, "mesh {} {}", mesh.path, mesh.scale);
        }
        if let Some(script) = &self.script {
            let _ = writeln!(text, "script {script}");
        }
        for property in &self.properties {
            let _ = writeln!(
                text,
                "property {} {} {}",
                property.name,
                property.kind.name(),
                PropertyKind::words(&property.default).join(" ")
            );
        }
        if let Some(eye) = self.camera {
            let _ = writeln!(text, "camera {} {} {}", eye.x, eye.y, eye.z);
        }
        text
    }

    pub fn property(&self, name: &str) -> Option<&Property> {
        self.properties.iter().find(|p| p.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE: &str = "\
# scenes/cove/gate.actor
name gate
mesh ../gate/gate.gltf 1.15   # the model
script gate.star
property open_seconds float 2.5
property locked bool false
property knocks int 0
property sign text closed
property hinge vec3 0 0 -1.7
camera 0 1.6 0
";

    #[test]
    fn an_actor_file_reads_back_what_it_wrote() {
        let gate = ActorFile::parse(GATE).unwrap();
        assert_eq!(gate.name, "gate");
        assert_eq!(
            gate.mesh,
            Some(Mesh {
                path: "../gate/gate.gltf".to_owned(),
                scale: 1.15
            })
        );
        assert_eq!(gate.script.as_deref(), Some("gate.star"));
        assert_eq!(gate.camera, Some(Vec3::new(0.0, 1.6, 0.0)));
        let defaults: Vec<_> = gate.properties.iter().map(|p| p.default.clone()).collect();
        assert_eq!(
            defaults,
            vec![
                ScriptValue::Float(2.5),
                ScriptValue::Bool(false),
                ScriptValue::Int(0),
                ScriptValue::from("closed"),
                ScriptValue::Vec3(Vec3::new(0.0, 0.0, -1.7)),
            ]
        );
        assert_eq!(ActorFile::parse(&gate.to_text()).unwrap(), gate);
        let bare = ActorFile::parse("name rock").unwrap();
        assert_eq!(bare.to_text(), "name rock\n");
        assert!(bare.mesh.is_none() && bare.script.is_none() && bare.camera.is_none());
    }

    #[test]
    fn mistakes_name_their_line() {
        let cases = [
            ("script a.star", 1, "needs a `name` line"),
            ("name gate\nname door", 2, "given twice"),
            ("name gate gate", 1, "expected `name NAME`"),
            ("name a-b", 1, "can't be a name"),
            ("name g\nmesh a.gltf 0", 2, "above 0"),
            ("name g\nmesh a.gltf big", 2, "`big` is not a number"),
            (
                "name g\n\nproperty locked boolean false",
                3,
                "not a kind of property",
            ),
            ("name g\nproperty locked bool yes", 2, "true or false"),
            ("name g\nproperty n int 2.5", 2, "whole number"),
            ("name g\nproperty v vec3 1 2", 2, "takes 3 values"),
            (
                "name g\nproperty position vec3 1 2 3",
                2,
                "can't be a property's name",
            ),
            (
                "name g\nproperty a int 1\nproperty a int 2",
                3,
                "given twice",
            ),
            (
                "name g\nproperty a int",
                2,
                "expected `property NAME KIND VALUE`",
            ),
            ("name g\ncamera 0 1", 2, "expected `camera X Y Z`"),
            ("name g\ncollider box 1 2 3", 2, "unknown line `collider`"),
        ];
        for (text, line, part) in cases {
            let error = ActorFile::parse(text).unwrap_err();
            assert_eq!(error.line, line, "{text:?}: {error}");
            assert!(error.message.contains(part), "{text:?}: {error}");
        }
    }

    #[test]
    fn settings_read_as_their_kind() {
        assert_eq!(
            PropertyKind::Vec3.parse_setting("1,2,3"),
            Ok(ScriptValue::Vec3(Vec3::new(1.0, 2.0, 3.0)))
        );
        assert_eq!(
            PropertyKind::Bool.parse_setting("true"),
            Ok(ScriptValue::Bool(true))
        );
        assert!(PropertyKind::Float.parse_setting("x").is_err());
        assert!(PropertyKind::Vec3.parse_setting("1,2").is_err());
    }
}
