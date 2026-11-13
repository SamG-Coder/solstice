//! A bake in the browser and one on a PC give the same bytes (`docs/editor.md`, "One bake") only
//! if every bit of maths they do is the same on both. The standard library's `sin`, `cos`,
//! `powf` and the like call each platform's own, which differ in their last bits, so what a bake
//! runs uses `solstice_asset::math` instead. This keeps it so: it finds any of them called on a
//! float in the bake's code, outside tests, where its result doesn't matter.

use std::path::Path;

/// Methods whose results differ between platforms. Square roots, rounding and the like are exact
/// everywhere, and fine.
const UNPORTABLE: [&str; 25] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sin_cos", "exp", "exp2", "exp_m1", "ln",
    "ln_1p", "log", "log2", "log10", "powf", "powi", "hypot", "cbrt", "sinh", "cosh", "tanh",
    "asinh", "acosh",
];

fn check(folder: &Path, found: &mut Vec<String>) {
    for entry in std::fs::read_dir(folder).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            check(&path, found);
            continue;
        }
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for (number, line) in text.lines().enumerate() {
            // A file's tests come last, in its `tests` module, however visible.
            let trimmed = line.trim_start();
            if trimmed.starts_with("mod tests") || trimmed.ends_with("mod tests {") {
                break;
            }
            let code = line.split("//").next().unwrap_or("");
            for name in UNPORTABLE {
                // A float's own method, `x.sin()`, or called by its type, `f32::sin(x)`.
                // `math::sin(x)`, the portable one, is neither.
                let calls = [
                    format!(".{name}("),
                    format!("f32::{name}("),
                    format!("f64::{name}("),
                ];
                if calls.iter().any(|call| code.contains(call.as_str())) {
                    found.push(format!(
                        "{}:{}: {}",
                        path.display(),
                        number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
}

#[test]
fn the_bake_uses_only_portable_maths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut found = Vec::new();
    for crate_folder in [
        "crates/solstice-procgen/src",
        "crates/solstice-bake/src",
        "crates/solstice-builder/src",
        "crates/solstice-asset/src",
        "crates/solstice-project/src",
    ] {
        check(&root.join(crate_folder), &mut found);
    }
    assert!(
        found.is_empty(),
        "these differ between the browser and a PC; use solstice_asset::math:\n{}",
        found.join("\n")
    );
}
