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

The whole map runs in the browser, served locally. A browser has 4 GiB of memory, and the scene's manifest unpacks to 5.65 GB, so it needs a second file holding the manifest without its 3.05 GB of cluster records and 2.07 GB of pinned pages, which follow it uncompressed. Make it from a baked scene, which takes about a minute and needs no re-bake:

```
bake --web-split zorah.solstice --output zorah.web
```

Put `zorah.solstice` and `zorah.web` in `dist/assets/`, serve `dist/` and open `?scene=zorah&split&geometry-mb=4500&texture-mb=1000`. The page hands those tables to the GPU in parts, and pages still stream from the scene file. The browser keeps only the 12 bytes of each cluster record that change as pages come and go. The file version doesn't change, and the native viewer refuses the `.web` file.

Tested by hand on one machine: Chrome 152 (Dawn), RTX 5080, Windows 11, at 1600×900, at budgets of 4,500 MiB geometry and 1,000 MiB textures. The tables load in about 12 seconds, and the opening view ran at 125 fps, 30.8 million triangles, with no GPU errors and about 10.7 GB in use on the card. CI doesn't run this path; a GPU test covers the split path on small scenes natively.

The full scene's cluster table and the page pool it wants are each over wgpu's 2 GiB binding limit, so `viewer --stream --geometry-budget-mb N` splits them between bindings for such a scene (`crates/solstice-render/src/regions.rs`): cluster records in two, and the pool in three by how simplified a page is. Pages are stored coarsest first, so the far band fills with the pages every other page depends on. The whole of Zorah has more of those always-loaded pages (19,537) than one binding holds (16,383), so the rest of the far band's pages go in the middle band. A page goes in its own pool if there's room, and otherwise borrows from whichever has the most it isn't using. The path is for an RTX 5080 on x64, natively or in Chrome. It needs 11 storage buffers per stage, and Chrome 152 reports 16 on that card, with a 2,147,483,644-byte storage binding and a 2 GiB buffer limit. Integrated GPUs are out of scope for this scene. Scenes that fit in single bindings don't use it, and compile the same shaders with the same limits as before.

To run the renderer's GPU tests through this path on small scenes: `SOLSTICE_FORCE_REGIONS=1 cargo test -p solstice-render --test gpu`.
