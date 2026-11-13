// The editor's bake, off the page's thread (docs/editor.md, "One bake"): takes a project's
// scene, gives back the scene file the viewer plays. Loaded only when the editor first bakes.
//
// In: `{ id, scene, name, content, quick }`, where `scene` is the scene file's text, `content` the
// folder holding ready-made textures (`bake --surfaces`), and `quick` (the default) writes the
// file quickly, for the editor to show; without it, the file is the one `bake --project` writes.
// Out: `{ id, bytes, ms }`, or `{ id, error }`.

import init, { bake_scene, content_keys, provide_layer } from "../bake-wasm/solstice_bake_web.js";

let ready = null;
/// Surfaces handed over already, by key and size.
const provided = new Set();

/// Hands the bake every layer made ready ahead: the land's surfaces, the woods' bark and leaves,
/// and the meadow's blades, which take minutes to make.
async function provideSurfaces(content, size, scene) {
  // Only what this scene uses: the land's surfaces under terrain, bark and leaves for woods,
  // blades for a meadow.
  const walk = (list) => list.flatMap((e) => [e, ...walk(e.children ?? [])]);
  const woods = walk(scene.entities ?? []).some((e) => e.wood);
  const wanted = (key) => (key.startsWith("surface-") && scene.terrain)
    || ((key.startsWith("bark-") || key.startsWith("leaves-")) && woods)
    || (key.startsWith("grass-atlas-") && scene.meadow);
  await Promise.all(content_keys().split("\n").filter(wanted).map(async (key) => {
    if (provided.has(`${key}@${size}`)) return;
    const response = await fetch(`${content}${key}-${size}.solstice`).catch(() => null);
    // Without one, the bake makes the surface itself: right, but slow.
    if (!response?.ok) return;
    provide_layer(key, new Uint8Array(await response.arrayBuffer()));
    provided.add(`${key}@${size}`);
  }));
}

onmessage = async ({ data: { id, scene, name, content, quick = true } }) => {
  try {
    // A start that failed, such as a fetch that didn't arrive, is tried again next time.
    ready ??= init().catch((error) => {
      ready = null;
      throw error;
    });
    await ready;
    const parsed = JSON.parse(scene);
    await provideSurfaces(content, parsed.texture_size ?? 1024, parsed);
    const start = performance.now();
    const bytes = bake_scene(scene, name, quick);
    postMessage({ id, bytes, ms: performance.now() - start }, [bytes.buffer]);
  } catch (error) {
    postMessage({ id, error: String(error?.message ?? error) });
  }
};
