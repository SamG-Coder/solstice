//! The wind on the CPU, the field the GPU fills from it, and what the bending shader does to a
//! blade in it. The GPU tests pass trivially, with a note, if no GPU is available.

use super::*;
use crate::scene::INSTANCE_SWAYS;
use std::sync::Arc;

fn weather(speed: f32, from: f32, gusts: f32, turbulence: f32) -> Weather {
    Weather {
        wind_speed: speed,
        wind_from: from,
        gusts,
        turbulence,
        ..Weather::default()
    }
}

/// The wind after `steps` frames of `seconds` each.
fn blown(weather: &Weather, steps: u32, seconds: f32) -> Wind {
    let mut wind = Wind::new();
    for _ in 0..steps {
        wind.advance(weather, seconds);
    }
    wind
}

/// Points on a grid `spacing` metres apart, `count` along each side, from `start`.
fn grid(start: Vec2, spacing: f32, count: u32) -> Vec<Vec2> {
    (0..count * count)
        .map(|i| start + Vec2::new((i % count) as f32, (i / count) as f32) * spacing)
        .collect()
}

#[test]
fn still_air_stays_still() {
    let wind = blown(&weather(0.0, 270.0, 1.0, 1.0), 30, 1.0 / 30.0);
    for p in grid(Vec2::splat(-100.0), 7.0, 30) {
        assert_eq!(wind.now().velocity_at(p), Vec2::ZERO);
    }
}

#[test]
fn a_steady_wind_is_its_mean_everywhere() {
    let wind = blown(&weather(12.0, 270.0, 0.0, 0.0), 10, 0.1);
    let mean = Vec2::from_array(wind.now().mean);
    for p in grid(Vec2::splat(-300.0), 23.0, 30) {
        assert!((wind.now().velocity_at(p) - mean).length() < 1.0e-5);
    }
}

#[test]
fn the_mean_follows_the_weather_and_wanders_a_little() {
    let weather = weather(16.0, 270.0, 0.5, 0.5);
    let mut wind = Wind::new();
    let mut turns = Vec::new();
    for _ in 0..3000 {
        wind.advance(&weather, 0.1);
        let mean = Vec2::from_array(wind.now().mean);
        // From the west, so towards +X; at half the cloud-height wind.
        assert!(mean.x > 0.0);
        let speed = mean.length();
        let ground = 16.0 * GROUND_SHARE;
        assert!((speed / ground - 1.0).abs() <= SWELL + 1.0e-4, "{speed}");
        turns.push(mean.y.atan2(mean.x).to_degrees());
    }
    let most = turns.iter().fold(0.0_f32, |m, t| m.max(t.abs()));
    assert!(most <= VEER_DEGREES + 1.0e-3, "turned {most}°");
    // Over five minutes it does wander, by some degrees.
    let least = turns.iter().cloned().fold(f32::MAX, f32::min);
    let greatest = turns.iter().cloned().fold(f32::MIN, f32::max);
    assert!(greatest - least > 4.0, "only {least}..{greatest}");
}

#[test]
fn the_air_carries_everything_at_its_mean_speed() {
    let weather = weather(10.0, 180.0, 0.5, 0.5);
    let mut wind = Wind::new();
    let mut expected = Vec2::ZERO;
    for _ in 0..600 {
        wind.advance(&weather, 1.0 / 60.0);
        expected += Vec2::from_array(wind.now().mean) / 60.0;
    }
    let offset = Vec2::from_array(wind.now().offset);
    assert!((offset - expected).length() < 1.0e-3, "{offset} {expected}");
    // A southerly blows towards the north, -Z.
    assert!(offset.y < -40.0 && offset.x.abs() < offset.y.abs() * 0.3);
}

#[test]
fn gusts_are_carried_downwind() {
    // Gusts only: how alike the field is to itself two seconds on, moved downwind by the air in
    // that time, and left where it was.
    let weather = weather(16.0, 270.0, 1.0, 0.0);
    let mut wind = blown(&weather, 100, 0.1);
    let before = *wind.now();
    for _ in 0..20 {
        wind.advance(&weather, 0.1);
    }
    let after = *wind.now();
    let carried = Vec2::from_array(after.offset) - Vec2::from_array(before.offset);
    assert!(carried.length() > 10.0);
    let (mut moved, mut stayed) = (0.0, 0.0);
    for p in grid(Vec2::splat(-60.0), 3.0, 40) {
        let was = before.velocity_at(p).length();
        moved += (after.velocity_at(p + carried).length() - was).powi(2);
        stayed += (after.velocity_at(p).length() - was).powi(2);
    }
    assert!(moved * 4.0 < stayed, "carried {moved}, left {stayed}");
}

/// Speeds over a square of ground, as shares of the mean.
fn spread(gusts: f32) -> (f32, f32, f32) {
    let wind = blown(&weather(16.0, 230.0, gusts, 0.0), 50, 0.1);
    let mean = Vec2::from_array(wind.now().mean).length();
    let speeds: Vec<f32> = grid(Vec2::splat(-200.0), 4.0, 100)
        .into_iter()
        .map(|p| wind.now().velocity_at(p).length() / mean)
        .collect();
    let average = speeds.iter().sum::<f32>() / speeds.len() as f32;
    let least = speeds.iter().cloned().fold(f32::MAX, f32::min);
    let most = speeds.iter().cloned().fold(f32::MIN, f32::max);
    (least, average, most)
}

#[test]
fn gusty_weather_has_gusts_and_lulls() {
    let (least, average, most) = spread(0.85);
    // The mean stays the mean.
    assert!((average - 1.0).abs() < 0.1, "{average}");
    // Gusts about half again the mean, as over open land, and lulls well below it.
    assert!((1.3..2.4).contains(&most), "gusts to {most}");
    assert!(least < 0.6, "lulls only to {least}");
    // A steadier day gusts less.
    let (calm_least, _, calm_most) = spread(0.25);
    assert!(calm_most < most && calm_least > least);
    assert!(calm_most < 1.4, "{calm_most}");
}

#[test]
fn turbulence_turns_the_wind_from_place_to_place() {
    let angles = |turbulence: f32| -> f32 {
        let wind = blown(&weather(10.0, 270.0, 0.0, turbulence), 10, 0.1);
        let mean = Vec2::from_array(wind.now().mean);
        grid(Vec2::splat(-200.0), 8.0, 50)
            .into_iter()
            .map(|p| mean.angle_to(wind.now().velocity_at(p)).abs())
            .fold(0.0, f32::max)
    };
    assert!(angles(0.0) < 1.0e-4);
    let turned = angles(1.0);
    assert!(turned > 0.2 && turned < VEER_ANGLE, "{turned}");
}

#[test]
fn with_no_time_passing_the_wind_is_as_it_was() {
    let weather = weather(16.0, 230.0, 0.8, 0.6);
    let mut wind = blown(&weather, 10, 0.1);
    wind.advance(&weather, 0.0);
    assert_eq!(wind.now(), wind.previous());
    // And the first step has nothing before it.
    let first = blown(&weather, 1, 0.1);
    assert_eq!(first.now(), first.previous());
}

#[test]
fn the_field_is_centred_on_the_camera_in_whole_texels() {
    let wind = blown(&weather(10.0, 270.0, 0.5, 0.5), 3, 0.1);
    for camera in [Vec2::ZERO, Vec2::new(123.4, -987.6), Vec2::new(-0.3, 0.9)] {
        let params = FieldParams::around(camera, &wind, true, [40.0, 60.0]);
        let origin = Vec2::from_array(params.origin);
        let centre = origin + Vec2::splat((FIELD_TEXELS / 2) as f32 * FIELD_TEXEL_METRES);
        assert!((centre - camera).abs().max_element() <= FIELD_TEXEL_METRES);
        let texels = (origin / FIELD_TEXEL_METRES) - Vec2::splat(0.5);
        assert!((texels - texels.round()).abs().max_element() < 1.0e-3);
    }
}

#[test]
fn records_lay_out_as_their_shader_twins() {
    crate::shaders::assert_twin!(
        "wind/field.wgsl",
        "WindSnapshot",
        WindSnapshot,
        [mean, offset, time, gusts, turbulence, _pad]
    );
    crate::shaders::assert_twin!(
        "wind/field.wgsl",
        "FieldParams",
        FieldParams,
        [now, previous, origin, texel, sway, fade, _pad]
    );
}

fn request_gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    match pollster::block_on(Gpu::request(
        &instance,
        None,
        wgpu::PowerPreference::HighPerformance,
    )) {
        Ok(gpu) => {
            gpu.device
                .on_uncaptured_error(Arc::new(|error: wgpu::Error| panic!("GPU error: {error}")));
            Some(gpu)
        }
        Err(error) => {
            eprintln!("skipping GPU test: {error}");
            None
        }
    }
}

/// Reads `buffer`'s first `bytes` back from the GPU.
fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    gpu.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |result| result.expect("mapped"));
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the GPU finished");
    let data = slice.get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    data
}

/// The whole field texture, row by row.
fn read_field(gpu: &Gpu, field: &Target) -> Vec<[f32; 4]> {
    let row = FIELD_TEXELS * 16;
    let rows = FIELD_TEXELS + 1;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("field readback"),
        size: u64::from(row * rows),
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        field.texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(rows),
            },
        },
        wgpu::Extent3d {
            width: FIELD_TEXELS,
            height: rows,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    bytemuck::cast_slice(&read_buffer(gpu, &buffer, u64::from(row * rows))).to_vec()
}

/// Fills a field for `wind` around `camera`.
fn fill(gpu: &Gpu, wind: &Wind, camera: Vec2, sway: bool, fade: [f32; 2]) -> (Target, FieldParams) {
    let field = field_texture(gpu);
    let pass = WindField::new(gpu, &field);
    let params = FieldParams::around(camera, wind, sway, fade);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    pass.record(gpu, &mut encoder, &params);
    gpu.queue.submit([encoder.finish()]);
    (field, params)
}

#[test]
fn the_gpu_field_is_the_cpus() {
    let Some(gpu) = request_gpu() else { return };
    let weather = weather(18.0, 215.0, 0.9, 0.7);
    let mut wind = blown(&weather, 200, 1.0 / 60.0);
    wind.advance(&weather, 1.0 / 30.0);
    let camera = Vec2::new(311.7, -52.3);
    let (field, params) = fill(&gpu, &wind, camera, true, [40.0, 60.0]);
    let texels = read_field(&gpu, &field);
    let origin = Vec2::from_array(params.origin);
    let mut worst = 0.0_f32;
    for y in 0..FIELD_TEXELS {
        for x in 0..FIELD_TEXELS {
            let p = origin + Vec2::new(x as f32, y as f32) * FIELD_TEXEL_METRES;
            let t = texels[(y * FIELD_TEXELS + x) as usize];
            let now = wind.now().velocity_at(p);
            let before = wind.previous().velocity_at(p);
            worst = worst
                .max((Vec2::new(t[0], t[1]) - now).length())
                .max((Vec2::new(t[2], t[3]) - before).length());
        }
    }
    // Hundredths of a millimetre a second: only rounding differs.
    assert!(worst < 1.0e-3, "differs by {worst} m/s");
    // The row below holds what readers need besides.
    let extra = &texels[(FIELD_TEXELS * FIELD_TEXELS) as usize..];
    assert_eq!(extra[0], [origin.x, origin.y, FIELD_TEXEL_METRES, 1.0]);
    assert_eq!(
        extra[1],
        [wind.now().time, wind.previous().time, 40.0, 60.0]
    );
}

/// A swaying instance standing at the origin, as the GPU holds it: identity transform, scaled
/// by `scale`, and flagged to sway if `sways`.
fn instance_words(scale: f32, sways: bool) -> [u32; 52] {
    let mut words = [0u32; 52];
    let one = |w: &mut [u32; 52], at: usize, row: [f32; 4]| {
        for (i, v) in row.iter().enumerate() {
            w[at + i] = v.to_bits();
        }
    };
    for base in [0, 12] {
        one(&mut words, base, [scale, 0.0, 0.0, 0.0]);
        one(&mut words, base + 4, [0.0, scale, 0.0, 0.0]);
        one(&mut words, base + 8, [0.0, 0.0, scale, 0.0]);
    }
    one(&mut words, 24, [1.0 / scale, 0.0, 0.0, 0.0]);
    one(&mut words, 28, [0.0, 1.0 / scale, 0.0, 0.0]);
    one(&mut words, 32, [0.0, 0.0, 1.0 / scale, 0.0]);
    words[39] = 1.0_f32.to_bits(); // quant_step
    words[45] = if sways { INSTANCE_SWAYS } else { 0 };
    words[46] = scale.to_bits(); // max_scale
    words[47] = u32::MAX; // no texture
    words[48] = 1.0_f32.to_bits(); // glow
    words[49] = crate::skin::NO_BONES;
    words
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Probe {
    local: [f32; 3],
    _a: f32,
    eye: [f32; 3],
    _b: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
struct ProbeResult {
    now: [f32; 4],
    before: [f32; 4],
    placed: [f32; 4],
    up: [f32; 4],
}

/// Runs the bending shader on `points` of an instance, seen from `eye`, in the field `field`.
fn probe(
    gpu: &Gpu,
    field: &Target,
    instance: [u32; 52],
    points: &[Vec3],
    eye: Vec3,
) -> Vec<ProbeResult> {
    let layout = LayoutBuilder::new()
        .texture_unfilterable()
        .storage_read()
        .storage_read()
        .storage_write()
        .build(gpu, "sway probe");
    let module = shaders::module(&gpu.device, "wind/sway_probe.wgsl");
    let pipeline = gpu.compute_pipeline("sway probe", &module, "main", &[&layout]);
    let probes: Vec<Probe> = points
        .iter()
        .map(|p| Probe {
            local: p.to_array(),
            _a: 0.0,
            eye: eye.to_array(),
            _b: 0.0,
        })
        .collect();
    let instances = gpu.storage_buffer_init("instances", &instance);
    let input = gpu.storage_buffer_init("probes", &probes);
    let bytes = (points.len() * std::mem::size_of::<ProbeResult>()) as u64;
    let output = gpu.storage_buffer("probe results", bytes, wgpu::BufferUsages::COPY_SRC);
    let group = gpu.bind_group(
        "sway probe",
        &layout,
        &[
            wgpu::BindingResource::TextureView(&field.view),
            instances.as_entire_binding(),
            input.as_entire_binding(),
            output.as_entire_binding(),
        ],
    );
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(groups_for(points.len() as u32, 64), 1, 1);
    }
    gpu.queue.submit([encoder.finish()]);
    bytemuck::cast_slice(&read_buffer(gpu, &output, bytes)).to_vec()
}

/// Points up a blade at (x, z) from its root to `height`, `steps` of them.
fn blade(x: f32, z: f32, height: f32, steps: u32) -> Vec<Vec3> {
    (0..=steps)
        .map(|i| Vec3::new(x, height * i as f32 / steps as f32, z))
        .collect()
}

fn offset(r: &ProbeResult) -> Vec3 {
    Vec3::new(r.now[0], r.now[1], r.now[2])
}

#[test]
fn a_blade_bends_downwind_and_keeps_its_length() {
    let Some(gpu) = request_gpu() else { return };
    let weather = weather(20.0, 270.0, 0.6, 0.4);
    let wind = blown(&weather, 120, 1.0 / 60.0);
    let (field, _) = fill(&gpu, &wind, Vec2::ZERO, true, [40.0, 60.0]);
    let eye = Vec3::new(0.0, 1.7, -3.0);
    for (x, z) in [(0.3, 0.2), (-1.7, 2.9), (4.1, -3.3)] {
        let points = blade(x, z, 0.6, 60);
        let results = probe(&gpu, &field, instance_words(1.0, true), &points, eye);
        // The root stays put.
        assert_eq!(offset(&results[0]), Vec3::ZERO);
        let bent: Vec<Vec3> = points
            .iter()
            .zip(&results)
            .map(|(p, r)| *p + offset(r))
            .collect();
        let length: f32 = bent.windows(2).map(|w| w[0].distance(w[1])).sum();
        assert!(
            (length - 0.6).abs() < 0.6 * 0.002,
            "the blade is {length} m long"
        );
        let wind_here = wind.now().velocity_at(Vec2::new(x, z));
        let tip = *results.last().unwrap();
        let lean = Vec2::new(tip.now[0], tip.now[2]);
        assert!(lean.length() > 0.05, "a gale hardly moves it: {lean}");
        assert!(lean.normalize().dot(wind_here.normalize()) > 0.9);
        // Never past the reach culling allows for.
        for (p, r) in points.iter().zip(&results) {
            let sideways = Vec2::new(r.now[0], r.now[2]).length();
            assert!(sideways <= GRASS_REACH * p.y + 1.0e-5);
            // The rasterizers place it where resolve does.
            for i in 0..3 {
                assert!((r.now[i] - r.placed[i]).abs() < 1.0e-6);
            }
            // Up turns downwind, and stays a unit direction.
            let up = Vec3::new(r.up[0], r.up[1], r.up[2]);
            assert!((up.length() - 1.0).abs() < 1.0e-5);
            if p.y > 0.1 {
                assert!(Vec2::new(up.x, up.z).dot(wind_here) > 0.0);
            }
        }
    }
}

/// The bend of a blade's tip, `height` up at (x, z), in `wind`.
fn tip(
    gpu: &Gpu,
    wind: &Wind,
    instance: [u32; 52],
    at: Vec3,
    eye: Vec3,
    sway: bool,
) -> ProbeResult {
    let (field, _) = fill(gpu, wind, Vec2::ZERO, sway, [40.0, 60.0]);
    probe(gpu, &field, instance, &[at], eye)[0]
}

#[test]
fn a_stronger_wind_bends_grass_further() {
    let Some(gpu) = request_gpu() else { return };
    let at = Vec3::new(0.5, 0.5, 0.5);
    let eye = Vec3::new(0.0, 1.7, -3.0);
    let lean = |speed: f32| {
        let wind = blown(&weather(speed, 270.0, 0.0, 0.0), 10, 0.1);
        let r = tip(&gpu, &wind, instance_words(1.0, true), at, eye, true);
        Vec2::new(r.now[0], r.now[2]).length()
    };
    let (breeze, wind, gale) = (lean(6.0), lean(14.0), lean(30.0));
    assert!(
        breeze > 0.002 && breeze < wind && wind < gale,
        "{breeze} {wind} {gale}"
    );
    // A breeze of 3 m/s near the ground stirs a half-metre blade, but leaves it standing.
    assert!(breeze < 0.1, "{breeze}");
    // A gale lays it well over.
    assert!(gale > 0.2, "{gale}");
}

#[test]
fn only_what_sways_and_is_near_moves() {
    let Some(gpu) = request_gpu() else { return };
    let wind = blown(&weather(20.0, 270.0, 0.6, 0.4), 30, 0.1);
    let at = Vec3::new(1.0, 0.5, 1.0);
    let near = Vec3::new(0.0, 1.7, -3.0);
    let moved = |r: ProbeResult| Vec3::new(r.placed[0], r.placed[1], r.placed[2]).length();
    assert!(moved(tip(&gpu, &wind, instance_words(1.0, true), at, near, true)) > 0.01);
    // Not flagged to sway.
    assert_eq!(
        moved(tip(&gpu, &wind, instance_words(1.0, false), at, near, true)),
        0.0
    );
    // Swaying turned off this frame.
    assert_eq!(
        moved(tip(&gpu, &wind, instance_words(1.0, true), at, near, false)),
        0.0
    );
    // Past where swaying fades out.
    let far = Vec3::new(0.0, 1.7, -70.0);
    assert_eq!(
        moved(tip(&gpu, &wind, instance_words(1.0, true), at, far, true)),
        0.0
    );
    // Halfway through the fade, partly.
    let fading = Vec3::new(0.0, 1.7, -49.0);
    let part = moved(tip(
        &gpu,
        &wind,
        instance_words(1.0, true),
        at,
        fading,
        true,
    ));
    let whole = moved(tip(&gpu, &wind, instance_words(1.0, true), at, near, true));
    assert!(part > 0.0 && part < whole, "{part} {whole}");
}

#[test]
fn a_scaled_instance_bends_by_its_height_in_the_world() {
    let Some(gpu) = request_gpu() else { return };
    let wind = blown(&weather(16.0, 270.0, 0.0, 0.0), 10, 0.1);
    let eye = Vec3::new(0.0, 1.7, -3.0);
    // Half a metre up in a mesh scaled by two stands a metre up.
    let scaled = tip(
        &gpu,
        &wind,
        instance_words(2.0, true),
        Vec3::new(0.25, 0.5, 0.25),
        eye,
        true,
    );
    let tall = tip(
        &gpu,
        &wind,
        instance_words(1.0, true),
        Vec3::new(0.5, 1.0, 0.5),
        eye,
        true,
    );
    assert!((scaled.now[3] - 1.0).abs() < 1.0e-5);
    for i in 0..3 {
        assert!((scaled.now[i] - tall.now[i]).abs() < 1.0e-5);
    }
}

#[test]
fn last_frames_bend_comes_from_last_frames_wind() {
    let Some(gpu) = request_gpu() else { return };
    let weather = weather(16.0, 270.0, 0.8, 0.5);
    let mut wind = blown(&weather, 60, 1.0 / 60.0);
    let at = Vec3::new(0.5, 0.5, 0.5);
    let eye = Vec3::new(0.0, 1.7, -3.0);
    let before = tip(&gpu, &wind, instance_words(1.0, true), at, eye, true);
    wind.advance(&weather, 1.0 / 60.0);
    let after = tip(&gpu, &wind, instance_words(1.0, true), at, eye, true);
    // What was "now" a frame ago is "before" now.
    for i in 0..3 {
        assert!((after.before[i] - before.now[i]).abs() < 1.0e-6);
    }
    assert_ne!(after.now[..3], after.before[..3]);
    // Paused, nothing moves.
    wind.advance(&weather, 0.0);
    let paused = tip(&gpu, &wind, instance_words(1.0, true), at, eye, true);
    assert_eq!(paused.now[..3], paused.before[..3]);
}
