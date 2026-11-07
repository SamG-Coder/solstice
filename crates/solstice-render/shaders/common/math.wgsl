// Small helpers shared by many passes.

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

// Decodes a signed octahedral vector in [-1, 1]^2 to a unit vector.
fn oct_decode(e: vec2f) -> vec3f {
    var v = vec3f(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if v.z < 0.0 {
        let folded = (vec2f(1.0) - abs(v.yx)) * sign_not_zero(v.xy);
        v = vec3f(folded.x, folded.y, v.z);
    }
    return normalize(v);
}

// Encodes a unit vector to signed octahedral coordinates in [-1, 1]^2.
fn oct_encode(n: vec3f) -> vec2f {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 {
        return (vec2f(1.0) - abs(p.yx)) * sign_not_zero(p);
    }
    return p;
}

fn sign_not_zero(v: vec2f) -> vec2f {
    return vec2f(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
}

// PCG hash: a well-distributed 32-bit integer hash.
fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash to a float in [0, 1).
fn hash_to_unit(v: u32) -> f32 {
    return f32(pcg(v) >> 8u) * (1.0 / 16777216.0);
}

// Interleaved gradient noise: low-discrepancy across neighbouring pixels.
fn interleaved_gradient_noise(pixel: vec2f, frame: u32) -> f32 {
    let p = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

// R2 low-discrepancy sequence, offset by a per-pixel seed.
fn r2_sequence(index: u32, seed: vec2f) -> vec2f {
    return fract(seed + f32(index) * vec2f(0.7548776662, 0.5698402910));
}

// An orthonormal basis around `n` (Duff et al. 2017), returned as tangent, bitangent.
fn orthonormal_basis(n: vec3f) -> mat3x3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat3x3f(
        vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3f(b, s + n.y * n.y * a, -n.y),
        n,
    );
}

// A cosine-weighted direction around +Z of the given basis.
fn cosine_sample(u: vec2f, basis: mat3x3f) -> vec3f {
    let r = sqrt(u.x);
    let phi = TAU * u.y;
    return basis * vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn sq(x: f32) -> f32 {
    return x * x;
}
