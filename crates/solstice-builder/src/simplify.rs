//! Quadric-error simplification by half-edge collapse.
//!
//! Half-edge collapses are used rather than optimal placement: every surviving
//! vertex is an original vertex, so the 16-bit position grid stays valid and no
//! new attributes have to be invented.
//!
//! Topology is tracked on welded *positions*, while attributes live on triangle
//! corners. A UV seam or hard edge is therefore not a topological boundary: the
//! surface collapses straight across it and each corner keeps a matching attribute.

use glam::{DVec3, Vec2, Vec3};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Shading attributes of a welded vertex.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Attribute {
    pub normal: Vec3,
    pub uv: Vec2,
}

/// A triangle corner: which welded position and which attribute it uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Corner {
    pub position: u32,
    pub attribute: u32,
}

pub type Triangle = [Corner; 3];

/// Result of [`simplify`].
pub struct Simplified {
    pub triangles: Vec<Triangle>,
    /// Largest geometric deviation introduced, in world units.
    pub error: f32,
}

/// Rejects a collapse that would rotate a face by more than about 78 degrees.
const MIN_NORMAL_AGREEMENT: f64 = 0.2;
/// Weight of the edge-length term that breaks ties on flat regions, relative to
/// squared length. Without it flat areas collapse in arbitrary order into slivers.
const EDGE_LENGTH_WEIGHT: f64 = 1.0e-4;
/// How much a normal discontinuity across an edge raises its collapse cost.
const NORMAL_DIFFERENCE_WEIGHT: f64 = 0.02;

/// Reduces `triangles` towards `target_triangles`.
///
/// Vertices for which `is_locked` returns true never move, and neither do the ends
/// of any edge that is not shared by exactly two triangles.
pub fn simplify(
    triangles: &[Triangle],
    positions: &[Vec3],
    attributes: &[Attribute],
    target_triangles: usize,
    is_locked: impl Fn(u32) -> bool,
) -> Simplified {
    Simplifier::new(triangles, positions, attributes, &is_locked).run(target_triangles)
}

/// A symmetric 4x4 error quadric plus the accumulated area weight.
#[derive(Clone, Copy, Default)]
struct Quadric {
    aa: f64,
    ab: f64,
    ac: f64,
    ad: f64,
    bb: f64,
    bc: f64,
    bd: f64,
    cc: f64,
    cd: f64,
    dd: f64,
    weight: f64,
}

impl Quadric {
    fn from_plane(normal: DVec3, d: f64, weight: f64) -> Self {
        let (a, b, c) = (normal.x, normal.y, normal.z);
        Self {
            aa: a * a * weight,
            ab: a * b * weight,
            ac: a * c * weight,
            ad: a * d * weight,
            bb: b * b * weight,
            bc: b * c * weight,
            bd: b * d * weight,
            cc: c * c * weight,
            cd: c * d * weight,
            dd: d * d * weight,
            weight,
        }
    }

    fn add(&mut self, o: &Self) {
        self.aa += o.aa;
        self.ab += o.ab;
        self.ac += o.ac;
        self.ad += o.ad;
        self.bb += o.bb;
        self.bc += o.bc;
        self.bd += o.bd;
        self.cc += o.cc;
        self.cd += o.cd;
        self.dd += o.dd;
        self.weight += o.weight;
    }

    /// Weighted mean squared distance from `p` to the accumulated planes.
    fn mean_squared_distance(&self, p: DVec3) -> f64 {
        if self.weight <= 0.0 {
            return 0.0;
        }
        let value = self.aa * p.x * p.x
            + 2.0 * self.ab * p.x * p.y
            + 2.0 * self.ac * p.x * p.z
            + 2.0 * self.ad * p.x
            + self.bb * p.y * p.y
            + 2.0 * self.bc * p.y * p.z
            + 2.0 * self.bd * p.y
            + self.cc * p.z * p.z
            + 2.0 * self.cd * p.z
            + self.dd;
        value.max(0.0) / self.weight
    }
}

/// A candidate collapse of `from` into `to`, valid while both stamps match.
struct Candidate {
    cost: f64,
    from: u32,
    to: u32,
    from_stamp: u32,
    to_stamp: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    /// Reversed on cost so that `BinaryHeap` pops the cheapest collapse. Vertex ids
    /// break ties so the result does not depend on heap internals.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.from.cmp(&self.from))
            .then_with(|| other.to.cmp(&self.to))
    }
}

struct Simplifier<'a> {
    attributes: &'a [Attribute],
    /// Global position id per local vertex.
    global: Vec<u32>,
    positions: Vec<DVec3>,
    locked: Vec<bool>,
    quadrics: Vec<Quadric>,
    stamps: Vec<u32>,
    /// Alive triangles touching each local vertex.
    vertex_triangles: Vec<Vec<u32>>,
    /// Triangles whose corners' `position` holds local vertex ids.
    triangles: Vec<Triangle>,
    alive: Vec<bool>,
    alive_count: usize,
    heap: BinaryHeap<Candidate>,
    max_error: f64,
}

impl<'a> Simplifier<'a> {
    fn new(
        input: &[Triangle],
        positions: &[Vec3],
        attributes: &'a [Attribute],
        is_locked: &dyn Fn(u32) -> bool,
    ) -> Self {
        let mut remap: FxHashMap<u32, u32> = FxHashMap::default();
        let mut global = Vec::new();
        let triangles: Vec<Triangle> = input
            .iter()
            .map(|t| {
                t.map(|corner| {
                    let local = *remap.entry(corner.position).or_insert_with(|| {
                        global.push(corner.position);
                        global.len() as u32 - 1
                    });
                    Corner {
                        position: local,
                        attribute: corner.attribute,
                    }
                })
            })
            .collect();

        let vertex_count = global.len();
        let local_positions: Vec<DVec3> = global
            .iter()
            .map(|&g| positions[g as usize].as_dvec3())
            .collect();
        let mut locked: Vec<bool> = global.iter().map(|&g| is_locked(g)).collect();
        let mut vertex_triangles = vec![Vec::new(); vertex_count];
        let mut quadrics = vec![Quadric::default(); vertex_count];
        let mut edge_uses: FxHashMap<u64, u32> = FxHashMap::default();

        for (index, t) in triangles.iter().enumerate() {
            let [a, b, c] = t.map(|corner| corner.position);
            let (pa, pb, pc) = (
                local_positions[a as usize],
                local_positions[b as usize],
                local_positions[c as usize],
            );
            let cross = (pb - pa).cross(pc - pa);
            let double_area = cross.length();
            if double_area > 0.0 {
                let normal = cross / double_area;
                let quadric = Quadric::from_plane(normal, -normal.dot(pa), double_area * 0.5);
                for v in [a, b, c] {
                    quadrics[v as usize].add(&quadric);
                }
            }
            for (v, next) in [(a, b), (b, c), (c, a)] {
                vertex_triangles[v as usize].push(index as u32);
                let key = u64::from(v.min(next)) << 32 | u64::from(v.max(next));
                *edge_uses.entry(key).or_insert(0) += 1;
            }
        }

        // Borders and non-manifold edges stay fixed. A border here is either the
        // group's outer boundary or a genuine hole in the source mesh.
        for (key, uses) in edge_uses {
            if uses != 2 {
                locked[(key >> 32) as usize] = true;
                locked[(key & 0xffff_ffff) as usize] = true;
            }
        }

        let alive_count = triangles.len();
        let alive = vec![true; alive_count];
        Self {
            attributes,
            global,
            positions: local_positions,
            locked,
            quadrics,
            stamps: vec![0; vertex_count],
            vertex_triangles,
            triangles,
            alive,
            alive_count,
            heap: BinaryHeap::new(),
            max_error: 0.0,
        }
    }

    fn run(mut self, target: usize) -> Simplified {
        for v in 0..self.positions.len() as u32 {
            self.push_edges_around(v);
        }
        while self.alive_count > target {
            let Some(candidate) = self.heap.pop() else {
                break;
            };
            if self.stamps[candidate.from as usize] != candidate.from_stamp
                || self.stamps[candidate.to as usize] != candidate.to_stamp
            {
                continue;
            }
            if self.can_collapse(candidate.from, candidate.to) {
                self.collapse(candidate.from, candidate.to);
            }
        }
        self.finish()
    }

    fn neighbors(&self, v: u32) -> Vec<u32> {
        let mut result: Vec<u32> = self.vertex_triangles[v as usize]
            .iter()
            .flat_map(|&t| self.triangles[t as usize])
            .map(|corner| corner.position)
            .filter(|&p| p != v)
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }

    fn push_edges_around(&mut self, v: u32) {
        for neighbor in self.neighbors(v) {
            for (from, to) in [(v, neighbor), (neighbor, v)] {
                if !self.locked[from as usize] {
                    let candidate = Candidate {
                        cost: self.collapse_cost(from, to),
                        from,
                        to,
                        from_stamp: self.stamps[from as usize],
                        to_stamp: self.stamps[to as usize],
                    };
                    self.heap.push(candidate);
                }
            }
        }
    }

    fn collapse_cost(&self, from: u32, to: u32) -> f64 {
        let mut quadric = self.quadrics[from as usize];
        quadric.add(&self.quadrics[to as usize]);
        let target = self.positions[to as usize];
        let length_squared = self.positions[from as usize].distance_squared(target);
        quadric.mean_squared_distance(target)
            + length_squared
                * (EDGE_LENGTH_WEIGHT + NORMAL_DIFFERENCE_WEIGHT * self.normal_difference(from, to))
    }

    /// How much the shading normals disagree between the two ends, in `0..=1`.
    fn normal_difference(&self, from: u32, to: u32) -> f64 {
        let normal_at = |v: u32| {
            self.vertex_triangles[v as usize]
                .iter()
                .flat_map(|&t| self.triangles[t as usize])
                .find(|corner| corner.position == v)
                .map_or(Vec3::Y, |corner| {
                    self.attributes[corner.attribute as usize].normal
                })
        };
        f64::from((1.0 - normal_at(from).dot(normal_at(to))) * 0.5)
    }

    fn can_collapse(&self, from: u32, to: u32) -> bool {
        // Link condition: the only vertices adjacent to both ends must be the two
        // opposite the shared edge, otherwise the collapse pinches the surface.
        let from_neighbors = self.neighbors(from);
        let to_neighbors = self.neighbors(to);
        let shared = from_neighbors
            .iter()
            .filter(|n| to_neighbors.binary_search(n).is_ok())
            .count();
        if shared != 2 {
            return false;
        }

        // Reject collapses that flip or degenerate any surviving triangle.
        let new_position = self.positions[to as usize];
        for &triangle in &self.vertex_triangles[from as usize] {
            let corners = self.triangles[triangle as usize].map(|c| c.position);
            if corners.contains(&to) {
                continue;
            }
            let old = corners.map(|v| self.positions[v as usize]);
            let new = corners.map(|v| {
                if v == from {
                    new_position
                } else {
                    self.positions[v as usize]
                }
            });
            let old_normal = (old[1] - old[0]).cross(old[2] - old[0]);
            let new_normal = (new[1] - new[0]).cross(new[2] - new[0]);
            let new_length = new_normal.length();
            if new_length <= 0.0 {
                return false;
            }
            if old_normal.dot(new_normal) < MIN_NORMAL_AGREEMENT * old_normal.length() * new_length
            {
                return false;
            }
        }
        true
    }

    fn collapse(&mut self, from: u32, to: u32) {
        let target = self.positions[to as usize];
        let mut merged = self.quadrics[to as usize];
        merged.add(&self.quadrics[from as usize]);
        self.max_error = self.max_error.max(merged.mean_squared_distance(target));
        self.quadrics[to as usize] = merged;

        let target_wedges = self.wedges_at(to);
        let from_triangles = std::mem::take(&mut self.vertex_triangles[from as usize]);
        for triangle in from_triangles {
            let index = triangle as usize;
            if !self.alive[index] {
                continue;
            }
            let corners = self.triangles[index];
            if corners.iter().any(|c| c.position == to) {
                self.alive[index] = false;
                self.alive_count -= 1;
                for corner in corners {
                    if corner.position != from {
                        self.vertex_triangles[corner.position as usize].retain(|&t| t != triangle);
                    }
                }
            } else {
                for corner in &mut self.triangles[index] {
                    if corner.position == from {
                        corner.position = to;
                        corner.attribute =
                            best_wedge(self.attributes, corner.attribute, &target_wedges);
                    }
                }
                self.vertex_triangles[to as usize].push(triangle);
            }
        }

        // Invalidate every queued collapse touching the changed neighbourhood.
        self.stamps[from as usize] += 1;
        self.stamps[to as usize] += 1;
        let neighbors = self.neighbors(to);
        for &n in &neighbors {
            self.stamps[n as usize] += 1;
        }
        self.push_edges_around(to);
        for n in neighbors {
            self.push_edges_around(n);
        }
    }

    fn wedges_at(&self, v: u32) -> Vec<u32> {
        let mut wedges: Vec<u32> = self.vertex_triangles[v as usize]
            .iter()
            .flat_map(|&t| self.triangles[t as usize])
            .filter(|corner| corner.position == v)
            .map(|corner| corner.attribute)
            .collect();
        wedges.sort_unstable();
        wedges.dedup();
        wedges
    }

    fn finish(self) -> Simplified {
        let global = &self.global;
        let triangles = self
            .triangles
            .iter()
            .zip(&self.alive)
            .filter(|(_, &alive)| alive)
            .map(|(t, _)| {
                t.map(|corner| Corner {
                    position: global[corner.position as usize],
                    attribute: corner.attribute,
                })
            })
            .collect();
        Simplified {
            triangles,
            error: self.max_error.sqrt() as f32,
        }
    }
}

/// Picks the attribute at the collapse target that best continues `current`.
///
/// Returns `current` when nothing is close: the corner is on the far side of a
/// hard edge or UV seam, and borrowing an attribute would smear across it.
fn best_wedge(attributes: &[Attribute], current: u32, wedges: &[u32]) -> u32 {
    let reference = attributes[current as usize];
    wedges
        .iter()
        .copied()
        .map(|wedge| {
            let candidate = attributes[wedge as usize];
            let normal_score = candidate.normal.dot(reference.normal);
            let uv_distance = candidate.uv.distance(reference.uv);
            (wedge, normal_score, uv_distance)
        })
        .filter(|&(_, normal_score, _)| normal_score > 0.7)
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .map_or(current, |(wedge, _, _)| wedge)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed UV sphere made of `segments * rings * 2`-ish triangles.
    fn closed_sphere(segments: u32, rings: u32) -> (Vec<Triangle>, Vec<Vec3>, Vec<Attribute>) {
        let mut positions = vec![Vec3::Y, -Vec3::Y];
        for ring in 1..rings {
            let theta = std::f32::consts::PI * ring as f32 / rings as f32;
            for segment in 0..segments {
                let phi = std::f32::consts::TAU * segment as f32 / segments as f32;
                positions.push(Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                ));
            }
        }
        let attributes: Vec<Attribute> = positions
            .iter()
            .map(|p| Attribute {
                normal: p.normalize(),
                uv: Vec2::ZERO,
            })
            .collect();
        let at = |ring: u32, segment: u32| 2 + (ring - 1) * segments + segment % segments;
        let mut indices: Vec<[u32; 3]> = Vec::new();
        for s in 0..segments {
            indices.push([0, at(1, s + 1), at(1, s)]);
            indices.push([1, at(rings - 1, s), at(rings - 1, s + 1)]);
            for r in 1..rings - 1 {
                indices.push([at(r, s), at(r, s + 1), at(r + 1, s)]);
                indices.push([at(r, s + 1), at(r + 1, s + 1), at(r + 1, s)]);
            }
        }
        let triangles = indices
            .into_iter()
            .map(|t| {
                t.map(|v| Corner {
                    position: v,
                    attribute: v,
                })
            })
            .collect();
        (triangles, positions, attributes)
    }

    #[test]
    fn reaches_target_on_a_closed_surface() {
        let (triangles, positions, attributes) = closed_sphere(32, 16);
        let target = triangles.len() / 2;
        let result = simplify(&triangles, &positions, &attributes, target, |_| false);
        assert!(result.triangles.len() <= target);
    }

    #[test]
    fn locked_vertices_survive() {
        let (triangles, positions, attributes) = closed_sphere(32, 16);
        let locked: Vec<u32> = (0..positions.len() as u32).step_by(3).collect();
        let result = simplify(&triangles, &positions, &attributes, 20, |v| {
            locked.contains(&v)
        });
        for &v in &locked {
            assert!(
                result.triangles.iter().flatten().any(|c| c.position == v),
                "locked vertex {v} disappeared"
            );
        }
    }

    #[test]
    fn simplified_surface_stays_closed() {
        let (triangles, positions, attributes) = closed_sphere(32, 16);
        let result = simplify(&triangles, &positions, &attributes, 100, |_| false);
        let mut uses: FxHashMap<(u32, u32), i32> = FxHashMap::default();
        for t in &result.triangles {
            for side in 0..3 {
                let (a, b) = (t[side].position, t[(side + 1) % 3].position);
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        assert!(uses.values().all(|&count| count == 2));
    }

    #[test]
    fn error_grows_as_the_target_shrinks() {
        let (triangles, positions, attributes) = closed_sphere(32, 16);
        let gentle = simplify(&triangles, &positions, &attributes, 500, |_| false);
        let harsh = simplify(&triangles, &positions, &attributes, 60, |_| false);
        assert!(harsh.error > gentle.error);
        assert!(gentle.error > 0.0);
    }
}
