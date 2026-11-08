// How water changes a surface's material (docs/wetness.md).
//
// Water wets a surface in two stages (Lagarde 2013, "Water drop 2: wet surfaces").
//   1. It fills the pores of a porous surface: sand, soil, rough stone. Water in the pores lowers
//      the contrast of refractive index between grains and what surrounds them, so light goes
//      deeper and more is absorbed: the surface darkens and its colour deepens (Lekner and Dorf
//      1988). Measured wet sands and soils reflect about half what they do dry (Twomey, Bohren
//      and Mergenthaler 1986). The surface turns a little smoother too.
//   2. Once the pores are full, or at once on a surface with none, such as paint or metal, a
//      film forms on top. It is smooth, so the surface turns glossy; on ground facing up it
//      fills the hollows, and the fine bumps drown under its flat top.
//
// The including shader declares `frame: Frame`.

// A soaked porous surface's albedo is its dry albedo raised to this power: between 1, dry, and
// 2, the limit Lekner and Dorf give for a highly porous one. Sand of albedo 0.35 comes out at
// about half, as measured; darker channels lose a larger share, so colours deepen.
const WET_DARKENING_POWER: f32 = 1.6;
// How much smoother a porous surface turns as its pores fill.
const WET_PORE_SMOOTHING: f32 = 0.7;
// Roughness of a film of water, as GGX roughness.
const WET_FILM_ROUGHNESS: f32 = 0.07;
// Wetness at which a porous surface's pores are full and a film starts to form.
const WET_PORES_FULL: f32 = 0.6;
// For now (docs/wetness.md, step 1), surfaces are wet from the calm sea's level up to this
// height above it, drying towards the top, anywhere in the world, and everything below the calm
// level counts as under water, even inland. Step 2 replaces this with where the water has been.
const WET_BAND: f32 = 0.3;

// How porous a material is, 0 to 1, as it has no such value of its own: rough dielectrics,
// such as sand and stone, are; smooth ones, such as paint or polished stone, are much less;
// metals are not at all.
fn porosity(roughness: f32, metallic: f32) -> f32 {
    return (1.0 - metallic) * smoothstep(0.3, 0.8, roughness);
}

// How wet the surface at `world` is, 0 to 1. Without a sea, nothing is.
fn wetness_at(world: vec3f) -> f32 {
    return 1.0 - smoothstep(0.0, WET_BAND, world.y - frame.sea_level);
}

// How far below the water the surface at `world` lies, 0 to 1. A film's gloss comes from its
// own surface against the air; under the sea there is none, and the sea's surface above gives
// the reflection instead. The pores still fill.
fn submerged_at(world: vec3f) -> f32 {
    return 1.0 - smoothstep(-0.1, 0.0, world.y - frame.sea_level);
}

struct WetSurface {
    albedo: vec3f,
    roughness: f32,
    normal: vec3f,
    // How wet the material was taken to be, cavities included.
    wetness: f32,
}

// The surface with base colour `albedo`, `roughness` and `metallic`, shading normal `normal` and
// the geometry's own normal `surface_normal`, made `wetness` wet, and `submerged` under the
// water. Cavities hold water first.
fn wet_surface(
    albedo: vec3f,
    roughness: f32,
    metallic: f32,
    normal: vec3f,
    surface_normal: vec3f,
    cavity: f32,
    wetness: f32,
    submerged: f32,
) -> WetSurface {
    var out = WetSurface(albedo, roughness, normal, 0.0);
    if wetness <= 0.0 {
        return out;
    }
    let porous = porosity(roughness, metallic);
    // Hollows fill first, so a cavity is a little wetter than the surface around it.
    let w = clamp(wetness + (1.0 - cavity) * 0.25 * wetness, 0.0, 1.0);
    out.wetness = w;

    // Pores filling.
    let pores = clamp(w / WET_PORES_FULL, 0.0, 1.0) * porous;
    let soaked = pow(max(albedo, vec3f(0.0)), vec3f(WET_DARKENING_POWER));
    out.albedo = mix(albedo, soaked, pores);
    out.roughness = roughness * mix(1.0, WET_PORE_SMOOTHING, pores);

    // A film: on porous surfaces once the pores are full, on others as soon as they are wet.
    let film = mix(w, smoothstep(WET_PORES_FULL, 1.0, w), porous) * (1.0 - submerged);
    out.roughness = mix(out.roughness, WET_FILM_ROUGHNESS, film);
    // On ground facing up the film pools in the hollows and drowns the bumps; on steep faces it
    // runs off, thin, and follows them.
    let pooling = film * smoothstep(0.5, 0.9, surface_normal.y);
    out.normal = normalize(mix(normal, surface_normal, pooling));
    return out;
}
