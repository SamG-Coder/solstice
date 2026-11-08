// The layout of the light tiles, shared by the pass that builds them and the passes that read
// them. Twins of the constants in `lighting.rs`.

// Pixels along each side of a tile.
const TILE_SIZE: u32 = 16u;
// Lights a tile can list. Any beyond are left out, and reported as the tile's count stops at
// this; lights overlap this deeply only where dozens stand within one another's range.
const TILE_MAX_LIGHTS: u32 = 127u;
// Words per tile: the count, then the indices.
const TILE_WORDS: u32 = 128u;

// Where the tile holding pixel `pixel` of a `size` viewport starts in the tile buffer.
fn light_tile_base(tile: vec2u, size: vec2u) -> u32 {
    let tiles_across = (size.x + TILE_SIZE - 1u) / TILE_SIZE;
    return (tile.y * tiles_across + tile.x) * TILE_WORDS;
}
