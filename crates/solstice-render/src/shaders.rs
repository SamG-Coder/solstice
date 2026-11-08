//! Shader sources and a tiny `#include` preprocessor.
//!
//! WGSL has no module system. Every file names its dependencies with
//! `#include "path"`, and each file is pasted in once per final shader, at its first
//! include. Sources are embedded at compile time, so there is no file I/O at runtime
//! and the browser build needs nothing extra.

use std::collections::HashSet;

macro_rules! shader_table {
    ($($path:literal),* $(,)?) => {
        &[$(($path, include_str!(concat!("../shaders/", $path)))),*]
    };
}

/// Every shader file, by its path relative to the `shaders` directory.
pub const SOURCES: &[(&str, &str)] = shader_table![
    "common/math.wgsl",
    "common/scene.wgsl",
    "common/frame.wgsl",
    "common/wetness.wgsl",
    "common/brdf.wgsl",
    "common/atmosphere.wgsl",
    "clouds/params.wgsl",
    "common/environment.wgsl",
    "clouds/common.wgsl",
    "vg/pass_view.wgsl",
    "vg/vis_write.wgsl",
    "vg/raster_common.wgsl",
    "vg/raster_hw.wgsl",
    "gi/probe_common.wgsl",
    "gi/radiance_codec.wgsl",
    "gi/screen_trace.wgsl",
    "gi/screen_history.wgsl",
    "vg/cull.wgsl",
    "vg/cull_control.wgsl",
    "vg/raster_hw_fragment.wgsl",
    "vg/raster_sw.wgsl",
    "vg/hzb_seed.wgsl",
    "vg/hzb_down.wgsl",
    "vg/resolve.wgsl",
    "sky/luts.wgsl",
    "clouds/noise.wgsl",
    "clouds/coverage.wgsl",
    "clouds/shadow.wgsl",
    "clouds/shadow_mean.wgsl",
    "clouds/panorama.wgsl",
    "clouds/screen.wgsl",
    "gi/voxelize.wgsl",
    "gi/jump_flood.wgsl",
    "gi/voxel_light.wgsl",
    "gi/volume_write.wgsl",
    "gi/depth_pyramid.wgsl",
    "gi/probe_trace.wgsl",
    "gi/probe_project.wgsl",
    "gi/probe_integrate.wgsl",
    "gi/reflections.wgsl",
    "ocean/common.wgsl",
    "ocean/fft.wgsl",
    "ocean/spectrum.wgsl",
    "ocean/evolve.wgsl",
    "ocean/fft_rows.wgsl",
    "ocean/fft_columns.wgsl",
    "ocean/mips.wgsl",
    "ocean/surface_params.wgsl",
    "ocean/seabed.wgsl",
    "ocean/waterline_params.wgsl",
    "ocean/waterline.wgsl",
    "ocean/surface.wgsl",
    "ocean/shade.wgsl",
    "sky/milky_way.wgsl",
    "sky/milky_way_bake.wgsl",
    "sky/bodies.wgsl",
    "lighting/sun_shadow.wgsl",
    "lighting/deferred.wgsl",
    "post/taa.wgsl",
    "post/exposure.wgsl",
    "post/bloom.wgsl",
    "post/tonemap.wgsl",
];

/// Entry-point files: the ones compiled into shader modules. Everything else exists
/// only to be included.
#[cfg(test)]
pub const ENTRY_POINTS: &[&str] = &[
    "vg/cull.wgsl",
    "vg/cull_control.wgsl",
    "vg/raster_hw_fragment.wgsl",
    "vg/raster_sw.wgsl",
    "vg/hzb_seed.wgsl",
    "vg/hzb_down.wgsl",
    "vg/resolve.wgsl",
    "sky/luts.wgsl",
    "ocean/spectrum.wgsl",
    "ocean/evolve.wgsl",
    "ocean/fft_rows.wgsl",
    "ocean/fft_columns.wgsl",
    "ocean/mips.wgsl",
    "ocean/surface.wgsl",
    "ocean/shade.wgsl",
    "ocean/waterline.wgsl",
    "sky/milky_way_bake.wgsl",
    "clouds/noise.wgsl",
    "clouds/coverage.wgsl",
    "clouds/shadow.wgsl",
    "clouds/shadow_mean.wgsl",
    "clouds/panorama.wgsl",
    "clouds/screen.wgsl",
    "gi/voxelize.wgsl",
    "gi/jump_flood.wgsl",
    "gi/voxel_light.wgsl",
    "gi/volume_write.wgsl",
    "gi/depth_pyramid.wgsl",
    "gi/probe_trace.wgsl",
    "gi/probe_project.wgsl",
    "gi/probe_integrate.wgsl",
    "gi/reflections.wgsl",
    "lighting/deferred.wgsl",
    "post/taa.wgsl",
    "post/exposure.wgsl",
    "post/bloom.wgsl",
    "post/tonemap.wgsl",
];

/// Returns the source of `path` with all includes expanded.
///
/// # Panics
/// If a file is missing from [`SOURCES`]. Shaders are compiled into the binary, so
/// this can only be a programming error, and the test suite covers every entry point.
pub fn expand(path: &str) -> String {
    let mut output = String::new();
    let mut seen = HashSet::new();
    append(path, &mut output, &mut seen);
    output
}

fn append<'a>(path: &'a str, output: &mut String, seen: &mut HashSet<&'a str>) {
    let (name, source) = SOURCES
        .iter()
        .find(|(name, _)| *name == path)
        .unwrap_or_else(|| panic!("unknown shader file: {path}"));
    if !seen.insert(name) {
        return;
    }
    for line in source.lines() {
        match line.trim().strip_prefix("#include ") {
            Some(include) => append(lookup_name(include.trim().trim_matches('"')), output, seen),
            None => {
                output.push_str(line);
                output.push('\n');
            }
        }
    }
}

/// Returns the `'static` table name for an include path, so `seen` can borrow it.
fn lookup_name(path: &str) -> &'static str {
    SOURCES
        .iter()
        .map(|(name, _)| *name)
        .find(|name| *name == path)
        .unwrap_or_else(|| panic!("unknown shader include: {path}"))
}

/// Compiles an entry-point file into a shader module.
pub fn module(device: &wgpu::Device, path: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(path),
        source: wgpu::ShaderSource::Wgsl(expand(path).into()),
    })
}

/// How naga lays out a struct, for tests that check a Rust twin matches it byte for byte: its
/// size, and each member's name and offset in order. `path` is an entry point that includes
/// the struct.
#[cfg(test)]
pub(crate) fn struct_layout(path: &str, name: &str) -> (usize, Vec<(String, usize)>) {
    let source = expand(path);
    let module = naga::front::wgsl::parse_str(&source).expect("the shader parses");
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{path} has no struct {name}"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("{name} is a struct");
    };
    let members = members
        .iter()
        .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
        .collect();
    (*span as usize, members)
}

/// Checks a Rust struct against its WGSL twin: the same size, and each field at the same
/// offset as the shader member in the same place. Fields are listed in order, one per shader
/// member. A Rust array that the shader spells out as several members is listed once for each,
/// with the extra bytes to each: `rows, rows + 16, rows + 32`.
#[cfg(test)]
macro_rules! assert_twin {
    ($path:expr, $wgsl:expr, $rust:ty, [$($field:ident $(+ $extra:literal)?),* $(,)?]) => {{
        let (size, members) = crate::shaders::struct_layout($path, $wgsl);
        assert_eq!(size, std::mem::size_of::<$rust>(), "size of {}", $wgsl);
        let rust: Vec<(&str, usize)> = vec![$((
            stringify!($field),
            std::mem::offset_of!($rust, $field) $(+ $extra)?,
        )),*];
        assert_eq!(rust.len(), members.len(), "member count of {}", $wgsl);
        for ((field, offset), (member, member_offset)) in rust.iter().zip(&members) {
            assert_eq!(offset, member_offset, "{field} against {member} in {}", $wgsl);
        }
    }};
}
#[cfg(test)]
pub(crate) use assert_twin;

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(path: &str) -> Result<(), String> {
        let source = expand(path);
        let module = naga::front::wgsl::parse_str(&source)
            .map_err(|error| format!("{path}:\n{}", error.emit_to_string(&source)))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .map_err(|error| format!("{path}:\n{}", error.emit_to_string(&source)))?;
        Ok(())
    }

    /// Catches every WGSL error at `cargo test` time, with no GPU required.
    #[test]
    fn all_entry_points_compile() {
        let failures: Vec<String> = ENTRY_POINTS
            .iter()
            .filter_map(|path| validate(path).err())
            .collect();
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    /// Operators WGSL will not mix without parentheses: a bitwise operator with any other
    /// binary operator, or two different bitwise operators. Some compilers, naga among them,
    /// accept the mix anyway, so validation alone misses it, but browsers reject the shader.
    fn unparenthesised_mixes(source: &str) -> Vec<String> {
        const BITWISE: [&str; 3] = [" ^ ", " | ", " & "];
        const OTHER: [&str; 7] = [" * ", " / ", " % ", " + ", " - ", " << ", " >> "];
        let mut found = Vec::new();
        for (number, line) in source.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            // Text at each depth of parentheses, with nested groups left out.
            let mut levels: Vec<String> = vec![String::new()];
            for c in code.chars() {
                match c {
                    '(' | '[' => {
                        levels.last_mut().unwrap().push(' ');
                        levels.push(String::new());
                    }
                    ')' | ']' => {
                        let inner = levels.pop().unwrap_or_default();
                        check(&inner, number, line, &mut found);
                        if levels.is_empty() {
                            levels.push(String::new());
                        }
                    }
                    _ => levels.last_mut().unwrap().push(c),
                }
            }
            for level in &levels {
                check(level, number, line, &mut found);
            }
        }
        fn check(text: &str, number: usize, line: &str, found: &mut Vec<String>) {
            let text = text.replace("&&", "  ").replace("||", "  ");
            let bitwise: Vec<_> = BITWISE.iter().filter(|op| text.contains(**op)).collect();
            if bitwise.is_empty() {
                return;
            }
            if bitwise.len() > 1 || OTHER.iter().any(|op| text.contains(op)) {
                found.push(format!("line {}: {}", number + 1, line.trim()));
            }
        }
        found
    }

    #[test]
    fn no_shader_mixes_operators_that_browsers_reject() {
        let problems: Vec<String> = ENTRY_POINTS
            .iter()
            .flat_map(|path| {
                unparenthesised_mixes(&expand(path))
                    .into_iter()
                    .map(move |p| format!("{path}: {p}"))
            })
            .collect();
        assert!(
            problems.is_empty(),
            "
{}",
            problems.join(
                "
"
            )
        );
        // And it does catch the mistake that once left the browser's sky black.
        assert_eq!(unparenthesised_mixes("let h = a * 3u ^ b * 5u;").len(), 1);
        assert!(unparenthesised_mixes("let h = (a * 3u) ^ (b * 5u);").is_empty());
    }

    #[test]
    fn includes_expand_exactly_once() {
        let source = expand("vg/raster_sw.wgsl");
        assert_eq!(source.matches("fn pcg(").count(), 1);
        assert!(!source.contains("#include"));
    }
}
