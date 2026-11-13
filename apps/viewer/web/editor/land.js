// What grows and lies on the land (docs/editor.md, phase 3): woods, clearings, paths, yards and
// ponds. They are entities like any other, but the bake makes what they look like, so a change to
// one bakes the scene again, but for ponds, which the viewer takes as they change.

import { projectToScreen } from "./math.js";
import { kindOf } from "./project.js";

/// The kinds of entity the land is shaped by, with what a new one starts as and the checks the
/// bake makes of each, so a value it would refuse is refused when it is set.
export const LAND_KINDS = {
  wood: {
    label: "Wood",
    description: "Trees (oak, birch and bush) over the grass, in a circle round its position.",
    placed: true,
    starts: { radius: 20, spacing: 7 },
    fields: { radius: { min: 0.5, max: 300, step: 0.5 }, spacing: { min: 0.6, max: 100, step: 0.1 } },
  },
  clear: {
    label: "Clearing",
    description: "Nothing planted within its radius of its position.",
    placed: true,
    starts: { radius: 10 },
    fields: { radius: { min: 0.1, max: 2000, step: 0.5 } },
  },
  pond: {
    label: "Pond",
    description: "Still water over a rectangle of the ground (x, z), at a level.",
    placed: false,
    starts: { min: [-6, -6], max: [6, 6], level: 0.5 },
    fields: { min: { pair: true }, max: { pair: true }, level: { min: -100, max: 500, step: 0.05 } },
  },
  path: {
    label: "Path",
    description: "A dirt path between two points on the ground (x, z).",
    placed: false,
    starts: { from: [-10, 0], to: [10, 0], width: 3 },
    fields: { from: { pair: true }, to: { pair: true }, width: { min: 0.1, max: 100, step: 0.1 } },
  },
  yard: {
    label: "Yard",
    description: "A dirt yard between opposite corners on the ground (x, z).",
    placed: false,
    starts: { from: [-8, -8], to: [8, 8] },
    fields: { from: { pair: true }, to: { pair: true } },
  },
};

/// Checks a land component's fields. Gives back an error's message, or null if it is good.
/// The most trees a wood may plant: the bake throws darts for each and checks each against all
/// planted, so a wood far past this would hold the bake for minutes.
const MOST_TREES = 6000;

export function checkLand(kind, value) {
  const spec = Object.hasOwn(LAND_KINDS, kind) ? LAND_KINDS[kind] : null;
  if (!spec) return `\`${kind}\` is not one of ${Object.keys(LAND_KINDS).join(", ")}`;
  if (!value || typeof value !== "object" || Array.isArray(value)) return `a ${kind} is an object of its fields`;
  for (const key of Object.keys(value)) {
    if (!Object.hasOwn(spec.fields, key) && !(kind === "pond" && key === "water")) {
      return `a ${kind} has no field \`${key}\`; it has ${Object.keys(spec.fields).join(", ")}`;
    }
  }
  for (const [key, field] of Object.entries(spec.fields)) {
    const v = value[key];
    if (field.pair) {
      if (!Array.isArray(v) || v.length !== 2 || !v.every((n) => typeof n === "number" && Number.isFinite(n) && Math.abs(n) < 1e6)) {
        return `\`${kind}.${key}\` is [x, z]`;
      }
    } else if (typeof v !== "number" || !Number.isFinite(v) || v < field.min || v > field.max) {
      return `\`${kind}.${key}\` is a number from ${field.min} to ${field.max}`;
    }
  }
  if (kind === "pond" && (value.min[0] === value.max[0] || value.min[1] === value.max[1])) {
    return "a pond's corners must differ along both x and z";
  }
  if (kind === "pond" && value.water !== undefined
    && !["coastal", "clear", "ocean", "murky"].includes(value.water)) {
    return "a pond's water is one of coastal, clear, ocean, murky";
  }
  if (kind === "wood" && (Math.PI * value.radius * value.radius) / (value.spacing * value.spacing) > MOST_TREES) {
    return `a wood that wide and that close would plant over ${MOST_TREES} trees: make it smaller or its trees further apart`;
  }
  return null;
}

/// What the bake makes of the land, as text: when it changes, the scene is baked again. Ponds are
/// left out: the viewer takes them as they change.
export function landSignature(scene, entities) {
  const land = [];
  for (const entity of entities.values()) {
    const kind = kindOf(entity);
    // A pond's water is baked; its outline and level are shown as they change.
    if (kind === "pond") land.push([entity.id, entity.pond.water ?? null]);
    else if (Object.hasOwn(LAND_KINDS, kind)) {
      land.push([entity.id, kind, entity[kind], entity.transform?.position ?? null]);
    }
  }
  return JSON.stringify([scene.texture_size, scene.terrain, scene.land, scene.meadow, scene.sea,
    scene.water, land]);
}

/// The scene's ponds as the viewer takes them: `x0,z0,x1,z1,level`, with `;` between.
export function pondsText(entities) {
  const ponds = [];
  for (const entity of entities.values()) {
    if (!entity.pond) continue;
    const { min, max, level } = entity.pond;
    ponds.push([Math.min(min[0], max[0]), Math.min(min[1], max[1]), Math.max(min[0], max[0]),
      Math.max(min[1], max[1]), level].join(","));
  }
  return ponds.join(";");
}

const COLOURS = { wood: "#46a758", clear: "#e8e8e8", pond: "#5cc8e2", path: "#c9a26b", yard: "#c9a26b" };

/// Draws the land's shapes on the ground, as both editors draw volumes and splines: woods and
/// clearings as circles, ponds and yards as rectangles, paths as their two edges.
export function drawLand(context, entities, camera, selection, hidden) {
  const { cssWidth: w, cssHeight: h } = camera;
  const toScreen = (x, y, z) => projectToScreen(camera.viewProjection, [x, y, z], w, h);
  const polyline = (points, closed) => {
    const screen = points.map(([x, y, z]) => toScreen(x, y, z));
    if (screen.some((p) => !p.front)) return;
    context.beginPath();
    screen.forEach((p, i) => (i ? context.lineTo(p.x, p.y) : context.moveTo(p.x, p.y)));
    if (closed) context.closePath();
    context.stroke();
  };
  context.save();
  for (const [id, entity] of entities) {
    const kind = kindOf(entity);
    if (!(kind in COLOURS) || hidden.has(id)) continue;
    const selected = selection.has(id);
    context.strokeStyle = selected ? "#ff9f1a" : COLOURS[kind];
    context.lineWidth = selected ? 2.5 : 1.5;
    context.globalAlpha = selected ? 1 : 0.75;
    context.setLineDash(kind === "clear" ? [6, 5] : []);
    const c = entity[kind];
    if (kind === "wood" || kind === "clear") {
      const [x, y, z] = entity.transform?.position ?? [0, 0, 0];
      const points = [];
      for (let i = 0; i < 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        points.push([x + Math.cos(a) * c.radius, y, z + Math.sin(a) * c.radius]);
      }
      polyline(points, true);
    } else if (kind === "pond" || kind === "yard") {
      const [a, b] = kind === "pond" ? [c.min, c.max] : [c.from, c.to];
      const y = kind === "pond" ? c.level : 0;
      polyline([[a[0], y, a[1]], [b[0], y, a[1]], [b[0], y, b[1]], [a[0], y, b[1]]], true);
    } else if (kind === "path") {
      const [dx, dz] = [c.to[0] - c.from[0], c.to[1] - c.from[1]];
      const length = Math.hypot(dx, dz) || 1;
      const [nx, nz] = [(-dz / length) * c.width / 2, (dx / length) * c.width / 2];
      polyline([[c.from[0] + nx, 0, c.from[1] + nz], [c.to[0] + nx, 0, c.to[1] + nz]], false);
      polyline([[c.from[0] - nx, 0, c.from[1] - nz], [c.to[0] - nx, 0, c.to[1] - nz]], false);
    }
  }
  context.restore();
}
