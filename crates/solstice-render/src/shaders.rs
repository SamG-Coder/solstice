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
    "common/brdf.wgsl",
    "common/atmosphere.wgsl",
    "common/environment.wgsl",
    "vg/pass_view.wgsl",
    "vg/vis_write.wgsl",
    "vg/raster_common.wgsl",
    "vg/raster_hw.wgsl",
    "gi/probe_common.wgsl",
    "gi/radiance_codec.wgsl",
    "vg/cull.wgsl",
    "vg/cull_control.wgsl",
    "vg/raster_hw_fragment.wgsl",
    "vg/raster_sw.wgsl",
    "vg/hzb_seed.wgsl",
    "vg/hzb_down.wgsl",
    "vg/resolve.wgsl",
    "sky/luts.wgsl",
    "gi/voxelize.wgsl",
    "gi/jump_flood.wgsl",
    "gi/voxel_light.wgsl",
    "gi/volume_write.wgsl",
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
    "gi/voxelize.wgsl",
    "gi/jump_flood.wgsl",
    "gi/voxel_light.wgsl",
    "gi/volume_write.wgsl",
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

    #[test]
    fn includes_expand_exactly_once() {
        let source = expand("vg/raster_sw.wgsl");
        assert_eq!(source.matches("fn pcg(").count(), 1);
        assert!(!source.contains("#include"));
    }
}
