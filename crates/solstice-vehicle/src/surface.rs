//! What a tyre runs on.

/// A kind of ground, as a tyre feels it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Surface {
    /// The racing surface.
    #[default]
    Asphalt,
    /// Painted, ribbed concrete at a corner's edge.
    Kerb,
    /// Mown grass beside the track.
    Grass,
    /// A gravel trap, which is there to stop a car.
    Gravel,
    /// Bare concrete: paddock and run-off.
    Concrete,
}

/// How a surface changes what a tyre can do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceGrip {
    /// Times the tyre's own friction on dry asphalt.
    pub grip: f32,
    /// Rolling resistance, as a share of the load on the tyre: at a crawl, and at speed. Loose
    /// ground drags harder the faster a tyre ploughs through it; a car can crawl out of a
    /// gravel trap that stopped it from a hundred.
    pub rolling_slow: f32,
    pub rolling: f32,
    /// How rough it is: the height, in metres, of the bumps a wheel rides over.
    pub bump_height: f32,
    /// How far apart those bumps are, in metres.
    pub bump_length: f32,
}

impl Surface {
    pub const ALL: [Self; 5] = [
        Self::Asphalt,
        Self::Kerb,
        Self::Grass,
        Self::Gravel,
        Self::Concrete,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Asphalt => "asphalt",
            Self::Kerb => "kerb",
            Self::Grass => "grass",
            Self::Gravel => "gravel",
            Self::Concrete => "concrete",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// Figures for a slick tyre in the dry. Grass and gravel are from the usual ranges quoted
    /// for race tyres off the track: a third to a half of the grip, and a gravel trap, which a
    /// car sinks into, slowing it at about half a g by drag alone.
    pub fn grip(self) -> SurfaceGrip {
        match self {
            Self::Asphalt => SurfaceGrip {
                grip: 1.0,
                rolling_slow: 0.012,
                rolling: 0.012,
                bump_height: 0.0,
                bump_length: 1.0,
            },
            Self::Kerb => SurfaceGrip {
                grip: 0.9,
                rolling_slow: 0.015,
                rolling: 0.015,
                bump_height: 0.0,
                bump_length: 1.0,
            },
            Self::Grass => SurfaceGrip {
                grip: 0.45,
                rolling_slow: 0.05,
                rolling: 0.07,
                bump_height: 0.018,
                bump_length: 1.3,
            },
            Self::Gravel => SurfaceGrip {
                grip: 0.42,
                rolling_slow: 0.14,
                rolling: 0.55,
                bump_height: 0.025,
                bump_length: 0.9,
            },
            Self::Concrete => SurfaceGrip {
                grip: 0.93,
                rolling_slow: 0.013,
                rolling: 0.013,
                bump_height: 0.0,
                bump_length: 1.0,
            },
        }
    }
}

impl SurfaceGrip {
    /// Rolling resistance at a speed, in metres a second, as a share of the load.
    pub fn rolling_at(&self, speed: f32) -> f32 {
        /// The speed by which loose ground drags its hardest.
        const FULL: f32 = 14.0;
        self.rolling_slow + (self.rolling - self.rolling_slow) * (speed.abs() / FULL).min(1.0)
    }
}

/// The height of a rough surface's bumps at a spot: smooth, repeatable noise between -1 and 1,
/// from where the wheel is on the ground and how far apart the bumps are.
pub fn bump(x: f32, z: f32, length: f32) -> f32 {
    let (u, v) = (x / length, z / length);
    let (iu, iv) = (u.floor(), v.floor());
    let (fu, fv) = (u - iu, v - iv);
    // Smoothstep, so that a wheel rolling over a cell's edge feels no kink.
    let (su, sv) = (fu * fu * (3.0 - 2.0 * fu), fv * fv * (3.0 - 2.0 * fv));
    let corner = |du: f32, dv: f32| lattice((iu + du) as i32, (iv + dv) as i32);
    let low = corner(0.0, 0.0) + (corner(1.0, 0.0) - corner(0.0, 0.0)) * su;
    let high = corner(0.0, 1.0) + (corner(1.0, 1.0) - corner(0.0, 1.0)) * su;
    low + (high - low) * sv
}

/// A value between -1 and 1 for a point of the lattice, the same every run and on every
/// platform: integer arithmetic only.
fn lattice(x: i32, z: i32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (z as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h & 0xFFFF) as f32 / 32_767.5 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_read_back() {
        for surface in Surface::ALL {
            assert_eq!(Surface::from_name(surface.name()), Some(surface));
        }
        assert_eq!(Surface::from_name("ice"), None);
    }

    #[test]
    fn loose_ground_drags_harder_at_speed() {
        let gravel = Surface::Gravel.grip();
        assert_eq!(gravel.rolling_at(0.0), gravel.rolling_slow);
        assert_eq!(gravel.rolling_at(30.0), gravel.rolling);
        assert_eq!(gravel.rolling_at(-30.0), gravel.rolling);
        assert!(
            gravel.rolling_at(7.0) > gravel.rolling_slow && gravel.rolling_at(7.0) < gravel.rolling
        );
        let road = Surface::Asphalt.grip();
        assert_eq!(road.rolling_at(0.0), road.rolling_at(80.0));
    }

    #[test]
    fn bumps_are_smooth_bounded_and_repeat() {
        let mut last = bump(0.0, 0.3, 1.3);
        for i in 1..2000 {
            let x = i as f32 * 0.01;
            let here = bump(x, 0.3, 1.3);
            assert!((-1.0..=1.0).contains(&here));
            assert!(
                (here - last).abs() < 0.05,
                "a kink at {x}: {last} to {here}"
            );
            last = here;
        }
        assert_eq!(bump(12.5, -7.25, 0.9), bump(12.5, -7.25, 0.9));
        // Not flat.
        let spread = (0..200)
            .map(|i| bump(i as f32 * 0.37, 1.0, 1.0).abs())
            .fold(0.0, f32::max);
        assert!(spread > 0.5);
    }
}
