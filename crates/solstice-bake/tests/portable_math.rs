//! A bake in the browser and one on a PC give the same bytes (`docs/editor.md`, "One bake") only
//! if every bit of maths they do is the same on both. The standard library's `sin`, `cos`,
//! `powf` and the like call each platform's own, which differ in their last bits, so what a bake
//! runs uses `solstice_asset::math` instead. This keeps it so: it finds any of them called on a
//! float in the bake's code, outside tests, where its result doesn't matter.

use std::path::Path;

/// Methods whose results differ between platforms. Square roots, rounding and the like are exact
/// everywhere, and fine.
const UNPORTABLE: [&str; 17] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sin_cos", "exp", "exp2", "ln", "log2",
    "log10", "powf", "powi", "hypot", "tanh",
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
                let call = format!(".{name}(");
                if let Some(at) = code.find(&call) {
                    // `math::sin(` is the portable one; a receiver's `.sin(` is not.
                    if !code[..at].ends_with("math:") {
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
}

#[test]
fn the_bake_uses_only_portable_maths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut found = Vec::new();
    for crate_folder in [
        "crates/solstice-procgen/src",
        "crates/solstice-bake/src",
        "crates/solstice-builder/src",
    ] {
        check(&root.join(crate_folder), &mut found);
    }
    assert!(
        found.is_empty(),
        "these differ between the browser and a PC; use solstice_asset::math:\n{}",
        found.join("\n")
    );
}
