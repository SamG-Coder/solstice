//! Maths that gives the same answer on every platform. The standard library's `sin`, `cos`,
//! `pow` and the like call the platform's own, which differ in their last bits between Windows,
//! Linux and the browser's wasm; so a scene baked in the browser would differ from the same scene
//! baked on a PC (`docs/editor.md`, "One bake"). Everything a bake computes with them goes through
//! these instead, which are `libm`'s, the same everywhere. Square roots and rounding are exact
//! on every platform already, and need nothing here.

/// A float these functions take: `f32` or `f64`.
pub trait Float: Copy {
    fn sin_(self) -> Self;
    fn cos_(self) -> Self;
    fn tan_(self) -> Self;
    fn asin_(self) -> Self;
    fn acos_(self) -> Self;
    fn atan_(self) -> Self;
    fn atan2_(self, x: Self) -> Self;
    fn exp_(self) -> Self;
    fn ln_(self) -> Self;
    fn log2_(self) -> Self;
    fn exp2_(self) -> Self;
    fn powf_(self, e: Self) -> Self;
    fn hypot_(self, o: Self) -> Self;
    fn one() -> Self;
    fn mul_(self, o: Self) -> Self;
    fn div_(self, o: Self) -> Self;
}

macro_rules! float {
    ($t:ty, $sin:ident, $cos:ident, $tan:ident, $asin:ident, $acos:ident, $atan:ident,
     $atan2:ident, $exp:ident, $ln:ident, $log2:ident, $exp2:ident, $pow:ident, $hypot:ident) => {
        impl Float for $t {
            fn sin_(self) -> Self {
                libm::$sin(self)
            }
            fn cos_(self) -> Self {
                libm::$cos(self)
            }
            fn tan_(self) -> Self {
                libm::$tan(self)
            }
            fn asin_(self) -> Self {
                libm::$asin(self)
            }
            fn acos_(self) -> Self {
                libm::$acos(self)
            }
            fn atan_(self) -> Self {
                libm::$atan(self)
            }
            fn atan2_(self, x: Self) -> Self {
                libm::$atan2(self, x)
            }
            fn exp_(self) -> Self {
                libm::$exp(self)
            }
            fn ln_(self) -> Self {
                libm::$ln(self)
            }
            fn log2_(self) -> Self {
                libm::$log2(self)
            }
            fn exp2_(self) -> Self {
                libm::$exp2(self)
            }
            fn powf_(self, e: Self) -> Self {
                libm::$pow(self, e)
            }
            fn hypot_(self, o: Self) -> Self {
                libm::$hypot(self, o)
            }
            fn one() -> Self {
                1.0
            }
            fn mul_(self, o: Self) -> Self {
                self * o
            }
            fn div_(self, o: Self) -> Self {
                self / o
            }
        }
    };
}
float!(f32, sinf, cosf, tanf, asinf, acosf, atanf, atan2f, expf, logf, log2f, exp2f, powf, hypotf);
float!(f64, sin, cos, tan, asin, acos, atan, atan2, exp, log, log2, exp2, pow, hypot);

pub fn sin<T: Float>(x: T) -> T {
    x.sin_()
}
pub fn cos<T: Float>(x: T) -> T {
    x.cos_()
}
pub fn tan<T: Float>(x: T) -> T {
    x.tan_()
}
pub fn asin<T: Float>(x: T) -> T {
    x.asin_()
}
pub fn acos<T: Float>(x: T) -> T {
    x.acos_()
}
pub fn atan<T: Float>(x: T) -> T {
    x.atan_()
}
/// The angle of `(x, y)`, as `y.atan2(x)` is.
pub fn atan2<T: Float>(y: T, x: T) -> T {
    y.atan2_(x)
}
pub fn exp<T: Float>(x: T) -> T {
    x.exp_()
}
pub fn ln<T: Float>(x: T) -> T {
    x.ln_()
}
pub fn log2<T: Float>(x: T) -> T {
    x.log2_()
}
pub fn exp2<T: Float>(x: T) -> T {
    x.exp2_()
}
pub fn powf<T: Float>(x: T, e: T) -> T {
    x.powf_(e)
}
pub fn hypot<T: Float>(x: T, y: T) -> T {
    x.hypot_(y)
}
pub fn sin_cos<T: Float>(x: T) -> (T, T) {
    (x.sin_(), x.cos_())
}

/// `x` to a whole power, by squaring, the same on every platform.
pub fn powi<T: Float>(x: T, n: i32) -> T {
    let (mut base, mut left, mut out) = (x, n.unsigned_abs(), T::one());
    while left > 0 {
        if left & 1 == 1 {
            out = out.mul_(base);
        }
        left >>= 1;
        if left > 0 {
            base = base.mul_(base);
        }
    }
    if n < 0 {
        T::one().div_(out)
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn they_agree_with_the_standard_library_closely() {
        for k in -200..200 {
            let x = k as f32 * 0.037;
            assert!((sin(x) - x.sin()).abs() < 1e-6);
            assert!((cos(x) - x.cos()).abs() < 1e-6);
            assert!((atan2(x, 0.7) - x.atan2(0.7)).abs() < 1e-6);
            assert!((powf(x.abs() + 0.1, 1.7) - (x.abs() + 0.1).powf(1.7)).abs() < 1e-4);
            assert!((powi(x, 3) - x.powi(3)).abs() <= 1e-5 * x.abs().powi(3).max(1.0));
            assert!((hypot(x, 2.0) - x.hypot(2.0)).abs() < 1e-6);
        }
        assert_eq!(powi(2.0_f64, -2), 0.25);
        assert_eq!(powi(3.0_f32, 0), 1.0);
    }
}
