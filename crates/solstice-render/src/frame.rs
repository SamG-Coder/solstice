//! The camera and the per-frame constants derived from it.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};

/// A perspective camera with reversed depth and an infinite far plane.
///
/// Reversed Z spends floating-point precision where perspective needs it most, and
/// an infinite far plane removes one thing to tune.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    /// Rotation around +Y, in radians. Zero looks along -Z.
    pub yaw: f32,
    /// Rotation above the horizon, in radians.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 10.0),
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 55.0_f32.to_radians(),
            near: 0.05,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(-sy * cp, sp, -cy * cp)
    }

    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, -sy)
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect, self.near)
    }
}

/// How the sun and sky light the scene.
#[derive(Clone, Copy, Debug)]
pub struct Environment {
    /// Direction towards the sun.
    pub sun_direction: Vec3,
    /// Illuminance of the sun at the top of the atmosphere, in scene units.
    pub sun_illuminance: f32,
    pub sun_color: Vec3,
    pub sky_intensity: f32,
    /// Multiplier on indirect lighting. 1.0 is physically consistent.
    pub gi_intensity: f32,
    /// The sun's illuminance in full daylight, which exposure is held to, so that night stays
    /// dark rather than being brightened to grey. 0 means `sun_illuminance`.
    pub daylight_illuminance: f32,
    /// Radiance of the night sky's own faint glow, airglow and starlight, added to the sky. It
    /// shines by day too, lost in the sunlit sky.
    pub night_sky: f32,
    /// Radiance of the brightest stars, before cloud hides them. 0 by day.
    pub stars: f32,
    /// The celestial sphere: the north celestial pole, and the March equinox on the celestial
    /// equator, where right ascension starts. Together they place every star.
    pub star_pole: Vec3,
    pub star_equinox: Vec3,
    /// Where the sun and the moon stand in the sky, to draw them. The scene is lit from
    /// `sun_direction`, which by night is the moon. `None` draws the sun at `sun_direction`, and
    /// no moon.
    pub sun_disc: Option<Vec3>,
    pub moon_disc: Option<Vec3>,
    /// Radiance of the moon's sunlit highlands, seen face on.
    pub moon_radiance: f32,
}

impl Environment {
    /// The sun's illuminance in full daylight: `daylight_illuminance`, or the sun's own when
    /// that is unset.
    pub fn daylight(&self) -> f32 {
        if self.daylight_illuminance > 0.0 {
            self.daylight_illuminance
        } else {
            self.sun_illuminance
        }
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_direction: Vec3::new(0.45, 0.35, 0.3).normalize(),
            sun_illuminance: 10.0,
            sun_color: Vec3::ONE,
            sky_intensity: 1.0,
            gi_intensity: 1.0,
            daylight_illuminance: 0.0,
            night_sky: 0.0,
            stars: 0.0,
            star_pole: Vec3::Y,
            star_equinox: Vec3::X,
            sun_disc: None,
            moon_disc: None,
            moon_radiance: 0.0,
        }
    }
}

/// What the renderer shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DebugView {
    #[default]
    Lit = 0,
    Clusters = 1,
    Triangles = 2,
    RasterMode = 3,
    Instances = 4,
    Albedo = 5,
    Normals = 6,
    DiffuseGi = 7,
    Reflections = 8,
    Voxels = 9,
    /// The sea's normals, where it is drawn.
    Ocean = 10,
    /// How wet each surface is: dark when dry, blue when soaked.
    Wetness = 11,
    /// How far each pixel moved since the last frame: grey where nothing moved, more red as it
    /// moved right on screen and more green as it moved up.
    Motion = 12,
    /// How many layers of particles cover each pixel: dark red for one, through yellow to white.
    Particles = 13,
    /// The light each particle receives, as a white particle would scatter it, without haze.
    ParticleLight = 14,
}

impl DebugView {
    /// Parses a mode name, as used by the viewer's command line and UI.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "lit" => Self::Lit,
            "clusters" => Self::Clusters,
            "triangles" => Self::Triangles,
            "raster" => Self::RasterMode,
            "instances" => Self::Instances,
            "albedo" => Self::Albedo,
            "normals" => Self::Normals,
            "gi" => Self::DiffuseGi,
            "reflections" => Self::Reflections,
            "voxels" => Self::Voxels,
            "ocean" => Self::Ocean,
            "wetness" => Self::Wetness,
            "motion" => Self::Motion,
            "particles" => Self::Particles,
            "particle-light" => Self::ParticleLight,
            _ => return None,
        })
    }
}

/// The sea level the shaders are given when there is no sea: far below anything, so that nothing
/// is under water or wet from it.
pub const NO_SEA: f32 = -1.0e30;

/// Twin of `Frame` in `common/frame.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameUniform {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub unjittered_view_proj: Mat4,
    pub prev_view_proj: Mat4,
    pub view: Mat4,
    pub camera_position: Vec3,
    pub frame_index: u32,
    pub viewport: Vec2,
    pub inv_viewport: Vec2,
    pub jitter_ndc: Vec2,
    pub near: f32,
    pub debug_mode: u32,
    pub sun_direction: Vec3,
    pub sun_illuminance: f32,
    pub sun_color: Vec3,
    pub sky_intensity: f32,
    pub light_count: u32,
    pub time: f32,
    pub gi_intensity: f32,
    pub temporal_accumulation: u32,
    /// Scales texture gradients: internal over output resolution, so that upscaling
    /// keeps the detail of the output resolution, times a detail bias for temporal filtering.
    pub mip_scale: f32,
    pub night_sky: f32,
    pub stars: f32,
    /// Radiance of the moon's sunlit highlands; 0 draws no moon.
    pub moon_radiance: f32,
    pub star_pole: Vec3,
    /// The sun's illuminance at noon, which night is measured against.
    pub daylight_illuminance: f32,
    pub star_equinox: Vec3,
    /// Seconds since the previous frame, which temporal filters scale their blend by.
    pub frame_seconds: f32,
    /// Where the sun and moon are drawn: the true sun, whatever lights the scene.
    pub sun_disc: Vec3,
    /// The calm sea's level; [`NO_SEA`] when there is none.
    pub sea_level: f32,
    pub moon_disc: Vec3,
    pub _pad2: f32,
}

/// Sub-pixel offset for frame `index`, from the Halton (2, 3) sequence, in `-0.5..0.5`.
pub fn jitter_offset(index: u32) -> Vec2 {
    Vec2::new(halton(index % 8 + 1, 2), halton(index % 8 + 1, 3)) - Vec2::splat(0.5)
}

fn halton(mut index: u32, base: u32) -> f32 {
    let (mut fraction, mut result) = (1.0, 0.0);
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// Adds a constant NDC offset to a projection matrix without changing depth.
pub fn jitter_projection(projection: Mat4, jitter_ndc: Vec2) -> Mat4 {
    let mut jittered = projection;
    // With w = -z_view, subtracting from the Z column shifts NDC by a constant amount.
    jittered.z_axis.x -= jitter_ndc.x;
    jittered.z_axis.y -= jitter_ndc.y;
    jittered
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL `Frame` must match `FrameUniform` byte for byte, or every value after the first
    /// difference is read from the wrong place. Naga lays out the WGSL; Rust reports its own.
    #[test]
    fn the_uniform_lays_out_as_its_shader_twin() {
        let source = include_str!("../shaders/common/frame.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("frame.wgsl parses");
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Frame"))
            .expect("a Frame struct");
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("Frame is a struct");
        };
        assert_eq!(*span as usize, std::mem::size_of::<FrameUniform>(), "size");
        assert_eq!(layouter[handle].size, *span);
        macro_rules! offsets {
            ($($field:ident),*) => {
                vec![$((stringify!($field), std::mem::offset_of!(FrameUniform, $field))),*]
            };
        }
        let rust = offsets!(
            view_proj,
            inv_view_proj,
            unjittered_view_proj,
            prev_view_proj,
            view,
            camera_position,
            frame_index,
            viewport,
            inv_viewport,
            jitter_ndc,
            near,
            debug_mode,
            sun_direction,
            sun_illuminance,
            sun_color,
            sky_intensity,
            light_count,
            time,
            gi_intensity,
            temporal_accumulation,
            mip_scale,
            night_sky,
            stars,
            star_pole,
            daylight_illuminance,
            star_equinox,
            moon_radiance,
            frame_seconds,
            sun_disc,
            sea_level,
            moon_disc
        );
        for (name, offset) in rust {
            let member = members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("{name} is missing from the WGSL"));
            assert_eq!(member.offset as usize, offset, "{name}");
        }
    }

    #[test]
    fn forward_matches_the_view_matrix() {
        let camera = Camera {
            yaw: 0.7,
            pitch: -0.3,
            position: Vec3::new(1.0, 2.0, 3.0),
            ..Default::default()
        };
        let ahead = camera
            .view()
            .transform_point3(camera.position + camera.forward());
        assert!(ahead.distance(Vec3::new(0.0, 0.0, -1.0)) < 1.0e-5);
    }

    #[test]
    fn projection_maps_near_plane_to_one_and_far_towards_zero() {
        let camera = Camera::default();
        let projection = camera.projection(1.5);
        let near = projection.project_point3(Vec3::new(0.0, 0.0, -camera.near));
        let far = projection.project_point3(Vec3::new(0.0, 0.0, -1.0e6));
        assert!((near.z - 1.0).abs() < 1.0e-5);
        assert!(far.z < 1.0e-5);
    }

    #[test]
    fn jitter_shifts_every_depth_by_the_same_amount() {
        let projection = Camera::default().projection(1.5);
        let offset = Vec2::new(0.01, -0.02);
        let jittered = jitter_projection(projection, offset);
        for z in [-0.5, -5.0, -500.0] {
            let p = Vec3::new(0.3, 0.2, z);
            let delta = jittered.project_point3(p) - projection.project_point3(p);
            assert!((delta.x - offset.x).abs() < 1.0e-5);
            assert!((delta.y - offset.y).abs() < 1.0e-5);
            assert!(delta.z.abs() < 1.0e-6);
        }
    }

    #[test]
    fn jitter_sequence_stays_within_half_a_pixel_and_averages_near_zero() {
        let mut sum = Vec2::ZERO;
        for i in 0..8 {
            let j = jitter_offset(i);
            assert!(j.x.abs() <= 0.5 && j.y.abs() <= 0.5);
            sum += j;
        }
        assert!((sum / 8.0).length() < 0.15);
    }
}
