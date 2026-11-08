//! Each test writes a small skinned glTF by hand, so every value checked is known exactly.

use super::*;
use std::path::PathBuf;

/// A scratch folder, with a one-pixel texture in it, removed when dropped.
struct Folder(PathBuf);

impl Folder {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let unique = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "solstice-character-{}-{unique}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = std::fs::File::create(dir.join("t.png")).unwrap();
        let mut png = png::Encoder::new(std::io::BufWriter::new(file), 1, 1);
        png.set_color(png::ColorType::Rgba);
        png.write_header()
            .unwrap()
            .write_image_data(&[200, 150, 120, 255])
            .unwrap();
        Self(dir)
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const IDENTITY: [f32; 16] = [
    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
];

/// A character to write: a quad of four vertices, skinned to a chain of bones.
struct Character {
    /// Each bone, the child of the one before.
    bones: Vec<&'static str>,
    /// Each vertex's four bones and their weights.
    weights: [([u16; 4], [f32; 4]); 4],
    morph: bool,
    second_weights: bool,
    skinned: bool,
    /// How many copies of the skin the file has.
    skins: usize,
    /// Base colour, normal, and metallic-roughness textures.
    textures: [bool; 3],
    alpha: &'static str,
}

impl Default for Character {
    fn default() -> Self {
        Self {
            bones: vec!["Root", "pelvis", "spine"],
            weights: [
                ([1, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]),
                ([1, 2, 0, 0], [0.5, 0.5, 0.0, 0.0]),
                ([2, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]),
                ([2, 1, 0, 0], [0.75, 0.25, 0.0, 0.0]),
            ],
            morph: false,
            second_weights: false,
            skinned: true,
            skins: 1,
            textures: [true; 3],
            alpha: "OPAQUE",
        }
    }
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl Character {
    /// Writes it, and checks it against `roles`.
    fn check(&self, roles: Option<&str>) -> CharacterReport {
        let folder = Folder::new();
        let mut bin = Vec::new();
        let mut views = Vec::new();
        let mut add = |bytes: Vec<u8>| {
            views.push(format!(
                r#"{{"buffer": 0, "byteOffset": {}, "byteLength": {}}}"#,
                bin.len(),
                bytes.len()
            ));
            bin.extend(bytes);
            views.len() - 1
        };
        let p = add(floats(&[0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.]));
        let i = add([0u16, 1, 2, 0, 2, 3]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
        let j = add(self
            .weights
            .iter()
            .flat_map(|(j, _)| j.iter().flat_map(|v| v.to_le_bytes()))
            .collect());
        let w = add(floats(
            &self
                .weights
                .iter()
                .flat_map(|(_, w)| *w)
                .collect::<Vec<_>>(),
        ));
        let bones = self.bones.len();
        let ibm = add(floats(
            &(0..bones).flat_map(|_| IDENTITY).collect::<Vec<_>>(),
        ));
        let mut accessors = vec![
            format!(
                r#"{{"bufferView": {p}, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0,0,0], "max": [1,1,0]}}"#
            ),
            format!(
                r#"{{"bufferView": {i}, "componentType": 5123, "count": 6, "type": "SCALAR"}}"#
            ),
            format!(r#"{{"bufferView": {j}, "componentType": 5123, "count": 4, "type": "VEC4"}}"#),
            format!(r#"{{"bufferView": {w}, "componentType": 5126, "count": 4, "type": "VEC4"}}"#),
            format!(
                r#"{{"bufferView": {ibm}, "componentType": 5126, "count": {bones}, "type": "MAT4"}}"#
            ),
        ];
        let mut attributes = r#""POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3"#.to_owned();
        if self.second_weights {
            attributes += r#", "JOINTS_1": 2, "WEIGHTS_1": 3"#;
        }
        let targets = if self.morph {
            accessors.push(accessors[0].clone());
            r#", "targets": [{"POSITION": 5}]"#
        } else {
            ""
        };
        let [base, normal, mr] = self.textures;
        let texture = |on: bool, key: &str| {
            if on {
                format!(r#""{key}": {{"index": 0}},"#)
            } else {
                String::new()
            }
        };
        let material = format!(
            r#"{{"name": "skin", {} "pbrMetallicRoughness": {{ {} {} "roughnessFactor": 1}}, "alphaMode": "{}"}}"#,
            texture(normal, "normalTexture"),
            texture(base, "baseColorTexture"),
            texture(mr, "metallicRoughnessTexture"),
            self.alpha
        );
        // Node 0 is the mesh; the bones follow, each the parent of the next.
        let mut nodes = vec![format!(
            r#"{{"name": "body", "mesh": 0 {}}}"#,
            if self.skinned { r#", "skin": 0"# } else { "" }
        )];
        for (index, bone) in self.bones.iter().enumerate() {
            let children = if index + 1 < bones {
                format!(r#", "children": [{}]"#, index + 2)
            } else {
                String::new()
            };
            nodes.push(format!(r#"{{"name": "{bone}" {children}}}"#));
        }
        let joints: Vec<String> = (1..=bones).map(|n| n.to_string()).collect();
        let json = format!(
            r#"{{
            "asset": {{"version": "2.0"}},
            "scene": 0, "scenes": [{{"nodes": [0, 1]}}],
            "nodes": [{}],
            "skins": [{}],
            "meshes": [{{"primitives": [{{"attributes": {{ {attributes} }}, "indices": 1, "material": 0 {targets}}}]}}],
            "materials": [{material}],
            "images": [{{"uri": "t.png"}}],
            "textures": [{{"source": 0}}],
            "buffers": [{{"uri": "c.bin", "byteLength": {}}}],
            "bufferViews": [{}],
            "accessors": [{}]
        }}"#,
            nodes.join(","),
            vec![
                format!(
                    r#"{{"joints": [{}], "inverseBindMatrices": 4}}"#,
                    joints.join(",")
                );
                self.skins
            ]
            .join(","),
            bin.len(),
            views.join(","),
            accessors.join(",")
        );
        std::fs::write(folder.0.join("c.bin"), &bin).unwrap();
        let path = folder.0.join("c.gltf");
        std::fs::write(&path, json).unwrap();
        let roles = roles.map(|r| Roles::parse(r).unwrap());
        check_character(&path, roles.as_ref()).unwrap()
    }
}

const ROLES: &str = "root Root\npelvis pelvis\nspine spine\n";

fn has(report: &CharacterReport, part: &str) -> bool {
    report.problems.iter().any(|p| p.contains(part))
}

#[test]
fn a_good_character_has_no_problems() {
    let report = Character::default().check(Some(ROLES));
    assert!(report.problems.is_empty(), "{}", report.summary());
    assert_eq!(report.joints, ["Root", "pelvis", "spine"]);
    assert_eq!(
        (report.meshes, report.vertices, report.triangles),
        (1, 4, 2)
    );
    assert_eq!(report.materials, 1);
    assert!(report.summary().ends_with("no problems"));
}

#[test]
fn every_kind_of_mistake_is_found() {
    let cases: Vec<(Character, &str)> = vec![
        (
            Character {
                morph: true,
                ..Default::default()
            },
            "morph targets",
        ),
        (
            Character {
                second_weights: true,
                ..Default::default()
            },
            "more than four weights",
        ),
        (
            Character {
                skinned: false,
                ..Default::default()
            },
            "isn't bound to the character's skin",
        ),
        (
            Character {
                textures: [true, false, true],
                ..Default::default()
            },
            "has no normal texture",
        ),
        (
            Character {
                textures: [false, true, false],
                ..Default::default()
            },
            "has no base colour or metallic-roughness texture",
        ),
        (
            Character {
                alpha: "BLEND",
                ..Default::default()
            },
            "is blended",
        ),
    ];
    for (character, part) in cases {
        let report = character.check(Some(ROLES));
        assert!(has(&report, part), "{part}: {}", report.summary());
    }
    let mut odd = Character::default();
    odd.weights[1].1 = [0.5, 0.4, 0.0, 0.0];
    assert!(has(
        &odd.check(None),
        "1 vertices have weights that don't sum to one"
    ));
    let mut nothing = Character::default();
    nothing.weights[2].1 = [0.0; 4];
    assert!(has(&nothing.check(None), "1 vertices follow no bone"));
    let mut stray = Character::default();
    stray.weights[3].0 = [7, 1, 0, 0];
    assert!(has(
        &stray.check(None),
        "1 weights name a bone the skin doesn't have"
    ));
}

#[test]
fn a_character_has_exactly_one_skin() {
    for skins in [0, 2] {
        let report = Character {
            skins,
            skinned: skins > 0,
            ..Default::default()
        }
        .check(None);
        let expected = format!("a character has one skin, not {skins}");
        assert!(has(&report, &expected), "{}", report.summary());
    }
}

#[test]
fn a_weight_sum_within_tolerance_is_fine() {
    let mut close = Character::default();
    // Off by a little less than the tolerance, as Blender's float weights can be.
    close.weights[1].1 = [0.5, 0.5015, 0.0, 0.0];
    let report = close.check(Some(ROLES));
    assert!(report.problems.is_empty(), "{}", report.summary());
}

#[test]
fn a_cut_out_material_is_fine() {
    let report = Character {
        alpha: "MASK",
        ..Default::default()
    }
    .check(Some(ROLES));
    assert!(report.problems.is_empty(), "{}", report.summary());
}

#[test]
fn bones_are_checked_against_the_role_file() {
    let report = Character::default().check(Some("root Root\nhead head\n"));
    assert!(has(
        &report,
        "the role file's bone `head` isn't in the skin"
    ));
    // A leaf bone that deforms nothing, and that no role names, should go. One that a role
    // names, or that other bones hang from, stays.
    let tip = Character {
        bones: vec!["Root", "pelvis", "spine", "tip"],
        ..Default::default()
    };
    assert!(has(&tip.check(Some(ROLES)), "bone `tip` deforms nothing"));
    let report = tip.check(Some("tip tip\n"));
    assert!(!has(&report, "`tip`"), "{}", report.summary());
    assert!(
        !has(&report, "`Root`"),
        "Root carries the others: {}",
        report.summary()
    );
}

#[test]
fn role_files_are_read_and_checked() {
    let roles = Roles::parse("# a rig\nspine a b  # a chain\n\nhead c\n").unwrap();
    assert_eq!(roles.bones().collect::<Vec<_>>(), ["a", "b", "c"]);
    let error = |text: &str| Roles::parse(text).unwrap_err();
    assert!(error("spine\n").contains("line 1: role `spine` names no bones"));
    assert!(error("a x\na y\n").contains("line 2: role `a` is given twice"));
    assert!(error("a x\nb x\n").contains("line 2: bone `x` already plays another role"));
    assert!(error("# nothing\n").contains("at least one role"));
}

#[test]
fn the_game_engine_rigs_role_file_reads() {
    let text = include_str!("../../../characters/rigs/game_engine.roles");
    let roles = Roles::parse(text).unwrap();
    assert_eq!(
        roles.bones().count(),
        53,
        "every bone of the rig plays a role"
    );
}
