# Zorah: NVIDIA's large scene

Solstice can bake [Zorah](https://github.com/nvpro-samples/vk_lod_clusters/blob/main/docs/scenes.md),
the glTF export of NVIDIA's RTX Kit sample shown at GDC 2025, which is far larger than anything
else this engine has read: 3.31 billion unique triangles, 18.9 billion with instancing, 2,812
meshes and 1,514 materials with 4,418 textures.

## Credit and licence

The Zorah scene is © 2025-2026 NVIDIA Corporation & Affiliates, released under the MIT licence,
whose text is in the archive's `LICENSE` file. Its environment map, `pizzo_pernice_4k.hdr`, is by
Andreas Mischok, from Poly Haven (CC0). The scene is not stored in this repository, and no bake of
it is published.

## Getting it

`python scripts/fetch-zorah.py DIR` downloads 75 GB and unpacks 83 GB, with the download resuming
if it is interrupted.

## Baking it

```
bake --large-gltf SCENE.gltf [--region X,Y,Z,RADIUS] [--no-textures] [--glow STRENGTH] [--texture-size N] --output FILE
```

`--large-gltf` works from disk, one mesh at a time, for scenes too large to hold in memory
(`crates/solstice-import/src/streamed.rs`). It builds meshes in parallel with a limit on the
triangles in flight, spills each mesh's vertices and triangles to a temporary file, cuts pages from
that, and writes the usual scene file: the format is unchanged, and the viewer reads it with
`--stream`. It stops cleanly if the output's drive falls below 50 GiB free, and removes its
temporary files however it ends.

It reads two extensions that `--gltf` does not need for ordinary files:
`EXT_meshopt_compression`, decoded in `meshopt.rs` to the specification's bitstream, and
`EXT_mesh_gpu_instancing`.

Textures are KTX2 in BC7 and BC5, zstd-compressed per level, which is what the renderer stores, so
`ktx2.rs` copies their blocks. A square level is an exact copy. Other shapes take the nearest
block. Normal maps have their green channel inverted in the blocks.

Emissive maps are kept at 512 texels, in the file's `GLOW` section, and materials that have one
glow by it. A material that blends and has an emissive map is left out of the clusters and written
to the `CARD` section as world-space triangles: in Zorah, the flames on the 30 standing lamps. Both
are optional sections, so the file version is unchanged and older files still load. glTF doesn't
say how bright an emissive factor of 1 is, so `--glow` multiplies them all. **Use `--glow 5` for
Zorah**: bright at night, still visible by day.

`--region` bakes only what lies within a radius of a point, which is how to try the scene quickly.
A radius of 5 m around NVIDIA's opening camera is 88 million unique triangles, baked in about 11
minutes.

## What it does not reproduce yet

- An alpha cut-off other than 0.5 is drawn at 0.5, since changing it means editing BC7 blocks.
  111 materials use 0.333, so foliage is a little thinner than it should be.
- A material with an emissive map gives the scene's GI light as if a tenth of it glowed, as voxels
  don't read the map.
- Zorah's flames are one still picture each. The shader sways and flickers them, which NVIDIA's
  scene doesn't describe.
- Non-square textures, 348 of the 4,418, are resampled by nearest block.

## Cards

Cards are drawn after the lit scene and water, and before particles (`src/cards.rs`,
`shaders/cards/draw.wgsl`). Scenes with none pay nothing, and a scene with no emissive maps compiles
the same resolve shader and bindings as before.

- **Sorting:** cards are drawn farthest first against each other. Against what is solid and water,
  each pixel is tested against the scene's depth with the water in it. Particles are drawn over all
  cards, so a particle behind a card is wrongly in front of it.
- **Shadows and GI:** cards don't cast or receive shadows, aren't lit, and give the scene no light.
  In Zorah the lamps' fuel has an emissive map and does give GI light.
- **TAA:** cards don't mark the reactive mask or write motion. The flicker is slow and small enough
  that it survives the filter; a fast-moving camera may trail them slightly.
- **Without BC textures** there are no cards, and materials glow evenly by their factors.

## Viewing it

The full scene's cluster table and the page pool it wants are each over wgpu's 2 GiB binding limit, so `viewer --stream --geometry-budget-mb N` splits them between bindings for such a scene (`crates/solstice-render/src/regions.rs`): cluster records in two, and the pool in three by how simplified a page is. **This path is supported only natively on x64 with an RTX 5080.** It needs 11 storage buffers per stage. Scenes that fit in single bindings don't use it, and compile the same shaders with the same limits as before.

To run the renderer's GPU tests through this path on small scenes: `SOLSTICE_FORCE_REGIONS=1 cargo test -p solstice-render --test gpu`.
