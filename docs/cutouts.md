# Cut-out materials

Status: built. Measured costs are at the end of this file.

glTF `alphaMode: MASK` materials (hair cards, eyebrows, eyelashes, leaves, nets, the ship's sails)
are drawn solid today. `solstice-import` warns and drops the alpha, so nothing later in the
renderer has a mask to test.

## Where a mask has to be honoured

| Stage | Today | Change |
| --- | --- | --- |
| Import (`material.rs`) | alpha dropped; colour A holds cavity | keep the mask in colour A, set a flag |
| Mips (`mip_chain`) | box filter, so coverage fades | coverage-preserving mips for masked layers |
| Bake and scene text file | nothing | none: the mask arrives through the glTF |
| LOD (`dag.rs`, `thin.rs`) | pieces are thinned and enlarged; UV error is costed | none expected; check on real hair |
| Culling (`cull.wgsl`) | two lists, software and hardware | two more lists for masked clusters |
| Visibility raster, hardware and software | no texture access | masked variants that test the mask |
| HZB and occlusion | built from visibility depth | none: holes write no depth, so they occlude nothing |
| `vg/resolve` | colour A read as cavity | cavity is 1 on masked materials |
| Sun cascades, hardware and software depth | no fragment stage | masked variants |
| Voxel GI (`voxelize.wgsl`) | whole triangles marked solid | test the mask per voxel |
| Local light maps (#41) | traced through the voxel volume | none: they follow the voxels |
| Screen GI, TAA | use depth and motion from resolve | none |
| Seabed depth map | hardware depth | none: masked geometry stays solid there |

## Asset format

One new bit, `Material::MASKED = 4`, in the existing `flags` field. On a masked material's texture
layer, colour A holds coverage instead of cavity. The binary layout and file version do not change,
and files that exist today read exactly as before.

The cut-off is not stored. At import, texture alpha is remapped piecewise so that the material's
effective cut-off, `alphaCutoff / baseColorFactor.a`, lands on 0.5. Every later stage tests against
0.5, which is also the value the coverage-preserving mips aim at. A tint's RGB plays no part, and
materials that differ only in RGB tint share a layer. Each distinct effective cut-off on an image
needs its own layer, so atlases should keep the factor's alpha at 1.

An atlas can mix masked and opaque cells, and materials of both kinds can share its layer. A scan
of the document before conversion finds the base colour images any `MASK` material uses, and those
layers keep their coverage for every material on them. An opaque material on such a layer reads its
cell's alpha as cavity, which is 1 where the cell is fully opaque, as it must be. The exception is
an opaque material with its own occlusion map, which gets a separate layer to hold it.

An occlusion map on a masked material is ignored, with a warning, as colour A is taken. `MASK` with
no base colour texture has nothing to cut out: it is opaque if its factor passes the cut-off, and
left out of the scene otherwise, with a warning.

`BLEND` stays opaque in the world importer, with its warning. The character checker already
rejects it.

## Texture preparation

- **Coverage-preserving mips** (Castaño 2010). A box-filtered mask loses or gains coverage at each
  level, so hair thins to nothing or swells into a blob with distance. Each level's alpha is scaled
  so that the fraction of texels above 0.5 matches mip 0. The scale is found per tile, on a 4×4
  grid of the image, because one scale for a whole atlas would average a hair cell with a net cell
  and an opaque one. A 2×2 atlas's cell borders fall on tile borders. A level of a fine regular
  pattern can be one alpha throughout, where no threshold can pass only part of it, so ties are
  parted by a fixed 4×4 pattern in the texture. That is built in at bake time and holds still on
  screen: the runtime test stays a hard 0.5 with no dither.
- **Colour under the holes.** Authored textures are often black where they are transparent, which
  shows as a dark fringe once a bilinear edge lands inside the mask. Colour is dilated into texels
  with zero alpha before mips are built, and mips average colour weighted by alpha. Textures already
  filled under their holes come out much the same.
- **BC7** keeps its current settings. A test checks that the compressed mask stays within a few
  percent of the source's coverage.

## Rendering

**Separate lists, so opaque clusters run the same work.** The visible list becomes
`[opaque software → … ← opaque hardware][masked software → … ← masked hardware]`. Indices stay
global, so `vg/resolve` and the visibility ID format are untouched. The split point is kept in the
last word of the cull state, written once when the list is created and left alone by the per-frame
clear. A scene with no masked material gets an empty masked region and records no masked passes.

**Masked pipelines are built only for scenes that have masked materials.** The software rasterizer
gains two hooks, per vertex and per pixel, that are empty functions in the opaque entry points and
compile away. The masked entry points compute perspective-correct UVs, pick one mip from the
triangle's UV gradient and call `textureSampleLevel` once. Hardware masked draws have a fragment
stage that does the same. The opaque hardware shadow pipeline still has no fragment stage.

**Binding limit.** The renderer requires only 8 storage buffers per stage, and the raster passes
already use all 8 once `materials` is added. `vertices` and `triangles` are the same buffer bound
twice, so they merge into one `array<u32>` binding, which leaves room. Masked passes bind
`materials`, the three colour arrays and the sampler in their own group.

**The mask samples the tier the material is in**, as resolve does, since the base tier's 128-texel
layers are too coarse for hair close up.

**Two-sided** works as it does now, through the instance flag. A new `INSTANCE_MASKED` bit takes one
more bit from the material index, leaving 2^27 materials.

**Views that keep masked geometry in the ordinary lists.** `PassView._pad2` becomes a switch.
Voxelization consumes one list and tests the mask per voxel itself. The seabed map draws masked
geometry solid, which only matters for something like a net lying on the seabed.

**Voxel GI.** Masked triangles sample coverage per voxel, at the mip that matches the voxel's
footprint, and keep the voxel if the coverage beats a hash of its world coordinate. Over an area
that gives roughly the right density, it does not change between rebuilds, and it uses the base
tier, which is all voxelization binds. It is only roughly right because mip alpha is moved to keep
coverage at the 0.5 test, not to equal coverage, so it leans towards half: sparse hair blocks a
little too much light and dense leaves a little too little. Local light shadows follow from this.

**Stability.** A hard 0.5 test on coverage-preserving mips, with no negative mip bias for the mask:
resolve's `TEXTURE_DETAIL` sharpening stays on colour only. A stochastic threshold would resolve
sub-pixel coverage better under TAA but adds noise to depth, motion and the cached shadows, so it
is left out unless the shimmer test below shows a real problem.

## LOD

No builder change was needed. Separate cards are already thinned with their area preserved
(`thin.rs`), and connected hair is held back by the texture-movement error in `simplify.rs`, so
holes stay in place until the error is too small to see. On a real MakeHuman head, the keeper, the
hair and brows keep their shape at 0.3 m and 2 m, and the hair is still a full head of hair at
20 m, as the view falls from 24,000 triangles to 5,000.

## Costs

Estimates, to be replaced with measurements.

- **Scenes with no masked material:** no extra passes, no extra memory, and the same shaders apart
  from the merged binding. Expected to be within noise.
- **Scenes with masked material, opaque work:** two empty-capable indirect passes per view, and a
  larger visible list, 8 bytes per extra entry.
- **Masked work:** one trilinear fetch and a barycentric solve per covered pixel in software. In
  hardware, a fragment stage with helper invocations, on clusters that are large on screen by
  construction. Overdraw from layered hair cards is the main risk, on the integrated Radeon most of
  all.
- **Shadow cascades:** masked hardware clusters gain a fragment stage. Cascades are cached, so this
  is paid when they are redrawn.
- **Bake:** coverage search and dilation on masked layers only.

To be measured and reported for sizing scenes: milliseconds per million covered masked pixels, on
each rasterizer path and in shadows, and how that grows with layers of overdraw. The harbour's
leaves depend on these figures on the integrated Radeon.

## Visible changes

- Anything authored `MASK` stops being solid, including the ship's sails and any scan that uses it.
- Masked materials lose baked cavity from an occlusion map. MakeHuman's have none.
- GI and local light shadows see masked surfaces at their average density, not their exact holes.
  Voxels are 20 cm and up, so a lantern's shadow of a net or a head of hair has no strands in it.
  Sun shadows do.

## Tests

GPU tests in `crates/solstice-render/tests/gpu.rs`, each failing without the fix:

- holes in the primary view, on both the hardware and software paths;
- holes in sun shadows, on both paths;
- a masked wall admits light in GI where a solid one does not;
- a two-sided card shows from behind and a one-sided one does not;
- a surface behind a masked card is not culled by occlusion;
- coverage at a distance matches coverage near, frame to frame, under TAA jitter;
- a scene with no masked material records no masked passes.

Import and mip tests sit beside their code.
