//! Builds the cluster LOD DAG and packs it for the GPU.

use crate::bounds::Sphere;
use crate::group::{group_clusters, GroupInput};
use crate::hierarchy::{build_hierarchy, Leaf};
use crate::mesh::Mesh;
use crate::partition::{partition_triangles, ClusterLimits};
use crate::simplify::{simplify, Attribute, Corner, Triangle};
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use solstice_asset::gpu::{
    encode_uv_density, pack_triangle, CLUSTER_UV_DENSITY_SHIFT, MAX_CLUSTER_TRIANGLES,
    MAX_CLUSTER_VERTICES, ROOT_PARENT_ERROR,
};
use solstice_asset::paged::{ClusterSet, ROOT_ROUND};
use solstice_asset::{PackedCluster, PackedVertex, VirtualMesh};
use std::ops::Range;

/// Tuning parameters for [`build_virtual_mesh`].
#[derive(Clone, Copy, Debug)]
pub struct BuildOptions {
    pub max_cluster_triangles: usize,
    pub max_cluster_vertices: usize,
    /// A group that cannot shrink below this fraction of its triangles is left
    /// as is for this level and retried in a differently shaped group next level.
    pub min_reduction: f32,
    /// Fraction of triangles a group tries to keep per level.
    pub target_ratio: f32,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            max_cluster_triangles: MAX_CLUSTER_TRIANGLES,
            max_cluster_vertices: MAX_CLUSTER_VERTICES,
            min_reduction: 0.85,
            target_ratio: 0.5,
        }
    }
}

/// A cluster during the build. LOD links are filled in as the DAG grows.
struct BuildCluster {
    triangles: Vec<Triangle>,
    level: u32,
    bounds: Sphere,
    /// Sphere and error of the group this cluster was generated from.
    lod_bounds: Sphere,
    lod_error: f32,
    /// Whether some group has been simplified from this cluster.
    has_parent: bool,
}

/// Clusters that swap for the same parents, on their way to becoming one hierarchy leaf.
struct SiblingSet {
    members: Vec<u32>,
    parent_bounds: Sphere,
    parent_error: f32,
    round: u32,
    generated: Range<u32>,
}

/// A group of clusters that was successfully simplified.
struct BuiltGroup {
    lod_bounds: Sphere,
    error: f32,
    children: Vec<u32>,
    /// Which pass of simplification made this group.
    round: u32,
    /// The clusters simplified from this group.
    generated: Range<u32>,
}

/// Mesh data after welding onto the 16-bit grid.
struct WeldedMesh {
    positions: Vec<Vec3>,
    attributes: Vec<Attribute>,
    triangles: Vec<Triangle>,
    origin: Vec3,
    step: f32,
    uv_origin: Vec2,
    uv_step: Vec2,
}

/// Converts a triangle mesh to a cluster LOD DAG.
pub fn build_virtual_mesh(mesh: &Mesh, options: &BuildOptions) -> VirtualMesh {
    let welded = weld(mesh);
    let source_triangles = welded.triangles.len() as u64;
    let limits = ClusterLimits {
        max_triangles: options.max_cluster_triangles,
        max_vertices: options.max_cluster_vertices,
    };

    let mut clusters = initial_clusters(&welded, limits);
    let mut groups: Vec<BuiltGroup> = Vec::new();
    let mut active: Vec<u32> = (0..clusters.len() as u32).collect();
    let mut level = 0;

    while active.len() > 1 {
        level += 1;
        active = build_level(
            &welded,
            &mut clusters,
            &mut groups,
            active,
            level,
            limits,
            options,
        );
    }

    pack(&welded, clusters, &groups, source_triangles)
}

/// Runs one round of group, simplify, re-split. Returns the clusters that stay
/// active for the next round, or an empty list if this round made no progress.
fn build_level(
    welded: &WeldedMesh,
    clusters: &mut Vec<BuildCluster>,
    groups: &mut Vec<BuiltGroup>,
    active: Vec<u32>,
    level: u32,
    limits: ClusterLimits,
    options: &BuildOptions,
) -> Vec<u32> {
    let cluster_triangles: Vec<Vec<[u32; 3]>> = active
        .iter()
        .map(|&c| {
            clusters[c as usize]
                .triangles
                .iter()
                .map(|t| t.map(|corner| corner.position))
                .collect()
        })
        .collect();
    let cluster_bounds: Vec<Sphere> = active
        .iter()
        .map(|&c| clusters[c as usize].bounds)
        .collect();
    let partition = group_clusters(&GroupInput {
        triangles: &cluster_triangles,
        bounds: &cluster_bounds,
    });

    // A vertex used by two different groups is on a boundary and must not move.
    let mut owner = vec![u32::MAX; welded.positions.len()];
    let mut shared = vec![false; welded.positions.len()];
    for (group_index, group) in partition.iter().enumerate() {
        for &member in group {
            for triangle in &cluster_triangles[member as usize] {
                for &position in triangle {
                    let slot = &mut owner[position as usize];
                    if *slot == u32::MAX {
                        *slot = group_index as u32;
                    } else if *slot != group_index as u32 {
                        shared[position as usize] = true;
                    }
                }
            }
        }
    }

    struct GroupResult {
        members: Vec<u32>,
        lod_bounds: Sphere,
        error: f32,
        clusters: Vec<Vec<Triangle>>,
    }

    let clusters_view: &[BuildCluster] = clusters;
    let results: Vec<Option<GroupResult>> = partition
        .par_iter()
        .map(|group| {
            let members: Vec<u32> = group.iter().map(|&m| active[m as usize]).collect();
            let triangles: Vec<Triangle> = members
                .iter()
                .flat_map(|&c| clusters_view[c as usize].triangles.iter().copied())
                .collect();
            let target = ((triangles.len() as f32 * options.target_ratio) as usize).max(1);
            let simplified = simplify(
                &triangles,
                &welded.positions,
                &welded.attributes,
                target,
                |v| shared[v as usize],
            );

            if simplified.triangles.is_empty()
                || simplified.triangles.len() as f32
                    > triangles.len() as f32 * options.min_reduction
            {
                return None;
            }

            // Spheres and errors must grow monotonically up the DAG. That is what
            // makes the runtime LOD test give a consistent, hole-free cut.
            let lod_bounds = Sphere::merged_all(
                members
                    .iter()
                    .map(|&c| clusters_view[c as usize].lod_bounds),
            );
            let child_error = members
                .iter()
                .map(|&c| clusters_view[c as usize].lod_error)
                .fold(0.0, f32::max);
            // The small increment keeps parents strictly coarser than children
            // even when a collapse on a perfectly flat region costs nothing.
            let error = child_error + simplified.error + welded.step * 0.5;

            let new_clusters =
                split_into_clusters(&simplified.triangles, &welded.positions, limits);
            Some(GroupResult {
                members,
                lod_bounds,
                error,
                clusters: new_clusters,
            })
        })
        .collect();

    let mut next_active = Vec::new();
    let mut progressed = false;
    for (group, result) in partition.iter().zip(results) {
        let Some(result) = result else {
            // Keep these clusters in play; next level they form different groups.
            next_active.extend(group.iter().map(|&m| active[m as usize]));
            continue;
        };
        progressed = true;
        for &member in &result.members {
            clusters[member as usize].has_parent = true;
        }
        let first_generated = clusters.len() as u32;
        groups.push(BuiltGroup {
            lod_bounds: result.lod_bounds,
            error: result.error,
            children: result.members,
            round: level,
            generated: first_generated..first_generated + result.clusters.len() as u32,
        });
        for triangles in result.clusters {
            let bounds = triangle_bounds(&triangles, &welded.positions);
            next_active.push(clusters.len() as u32);
            clusters.push(BuildCluster {
                triangles,
                level,
                bounds,
                lod_bounds: result.lod_bounds,
                lod_error: result.error,
                has_parent: false,
            });
        }
    }

    if progressed {
        next_active
    } else {
        Vec::new()
    }
}

fn initial_clusters(welded: &WeldedMesh, limits: ClusterLimits) -> Vec<BuildCluster> {
    split_into_clusters(&welded.triangles, &welded.positions, limits)
        .into_iter()
        .map(|triangles| {
            let bounds = triangle_bounds(&triangles, &welded.positions);
            BuildCluster {
                triangles,
                level: 0,
                bounds,
                lod_bounds: bounds,
                lod_error: 0.0,
                has_parent: false,
            }
        })
        .collect()
}

fn split_into_clusters(
    triangles: &[Triangle],
    positions: &[Vec3],
    limits: ClusterLimits,
) -> Vec<Vec<Triangle>> {
    let by_position: Vec<[u32; 3]> = triangles.iter().map(|t| t.map(|c| c.position)).collect();
    let mut clusters = Vec::new();
    for members in partition_triangles(&by_position, positions, limits) {
        let cluster: Vec<Triangle> = members.iter().map(|&t| triangles[t as usize]).collect();
        split_until_indexable(cluster, limits.max_vertices, &mut clusters);
    }
    clusters
}

/// Partitioning counts welded positions, but a cluster stores one vertex for each distinct
/// position and attribute pair. A model with many UV seams or hard edges can therefore stay
/// within the position limit and still exceed what 8-bit indices can address. Halving keeps
/// neighbouring triangles together, because partitioning emits them in the order they grew.
fn split_until_indexable(
    cluster: Vec<Triangle>,
    max_vertices: usize,
    out: &mut Vec<Vec<Triangle>>,
) {
    let mut corners: Vec<Corner> = cluster.iter().flatten().copied().collect();
    corners.sort_unstable_by_key(|c| (c.position, c.attribute));
    corners.dedup();
    if corners.len() <= max_vertices {
        out.push(cluster);
        return;
    }
    let mut first = cluster;
    let second = first.split_off(first.len() / 2);
    split_until_indexable(first, max_vertices, out);
    split_until_indexable(second, max_vertices, out);
}

fn triangle_bounds(triangles: &[Triangle], positions: &[Vec3]) -> Sphere {
    Sphere::from_points(
        triangles
            .iter()
            .flatten()
            .map(|corner| positions[corner.position as usize]),
    )
}

/// Snaps vertices to the mesh-wide 16-bit grid and welds them.
///
/// Welding *after* snapping guarantees one grid position has one topological
/// vertex. Triangles that collapse to a line under quantisation are dropped.
fn weld(mesh: &Mesh) -> WeldedMesh {
    let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    let (mut uv_min, mut uv_max) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for vertex in &mesh.vertices {
        min = min.min(vertex.position);
        max = max.max(vertex.position);
        uv_min = uv_min.min(vertex.uv);
        uv_max = uv_max.max(vertex.uv);
    }
    if mesh.vertices.is_empty() {
        (min, max, uv_min, uv_max) = (Vec3::ZERO, Vec3::ZERO, Vec2::ZERO, Vec2::ZERO);
    }

    let step = ((max - min).max_element() / 65535.0).max(1.0e-9);
    let uv_step = ((uv_max - uv_min) / 65535.0).max(Vec2::splat(1.0e-9));

    let mut position_ids: FxHashMap<[u16; 3], u32> = FxHashMap::default();
    let mut attribute_ids: FxHashMap<[u32; 4], u32> = FxHashMap::default();
    let mut positions = Vec::new();
    let mut attributes = Vec::new();

    let corners: Vec<Corner> = mesh
        .vertices
        .iter()
        .map(|vertex| {
            let grid = ((vertex.position - min) / step).round().as_uvec3();
            let key = [grid.x as u16, grid.y as u16, grid.z as u16];
            let position = *position_ids.entry(key).or_insert_with(|| {
                positions.push(min + Vec3::new(key[0] as f32, key[1] as f32, key[2] as f32) * step);
                positions.len() as u32 - 1
            });

            let octahedral = encode_octahedral(vertex.normal);
            let uv = ((vertex.uv - uv_min) / uv_step).round().as_uvec2();
            let attribute_key = [
                u32::from(octahedral[0] as u16),
                u32::from(octahedral[1] as u16),
                uv.x,
                uv.y,
            ];
            let attribute = *attribute_ids.entry(attribute_key).or_insert_with(|| {
                attributes.push(Attribute {
                    normal: vertex.normal.normalize_or(Vec3::Y),
                    uv: vertex.uv,
                });
                attributes.len() as u32 - 1
            });
            Corner {
                position,
                attribute,
            }
        })
        .collect();

    let triangles = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            [
                corners[t[0] as usize],
                corners[t[1] as usize],
                corners[t[2] as usize],
            ]
        })
        .filter(|t| {
            t[0].position != t[1].position
                && t[1].position != t[2].position
                && t[2].position != t[0].position
        })
        .collect();

    WeldedMesh {
        positions,
        attributes,
        triangles,
        origin: min,
        step,
        uv_origin: uv_min,
        uv_step,
    }
}

/// Encodes a unit vector to 16-bit signed octahedral coordinates.
fn encode_octahedral(normal: Vec3) -> [i16; 2] {
    let n = normal.normalize_or(Vec3::Y);
    let p = Vec2::new(n.x, n.y) / (n.x.abs() + n.y.abs() + n.z.abs());
    let folded = if n.z < 0.0 {
        Vec2::new(
            (1.0 - p.y.abs()) * p.x.signum(),
            (1.0 - p.x.abs()) * p.y.signum(),
        )
    } else {
        p
    };
    [
        (folded.x.clamp(-1.0, 1.0) * 32767.0).round() as i16,
        (folded.y.clamp(-1.0, 1.0) * 32767.0).round() as i16,
    ]
}

/// Orders clusters by sibling set and emits the GPU representation.
fn pack(
    welded: &WeldedMesh,
    clusters: Vec<BuildCluster>,
    groups: &[BuiltGroup],
    source_triangles: u64,
) -> VirtualMesh {
    let mut out = VirtualMesh {
        quant_origin: welded.origin,
        quant_step: welded.step,
        uv_origin: welded.uv_origin.to_array(),
        uv_step: welded.uv_step.to_array(),
        source_triangles,
        ..Default::default()
    };
    if clusters.is_empty() {
        return out;
    }

    // Sibling sets: children of one group, then parentless clusters in small runs.
    let mut sibling_sets: Vec<SiblingSet> = groups
        .iter()
        .map(|group| SiblingSet {
            members: group.children.clone(),
            parent_bounds: group.lod_bounds,
            parent_error: group.error,
            round: group.round,
            generated: group.generated.clone(),
        })
        .collect();
    let roots: Vec<u32> = (0..clusters.len() as u32)
        .filter(|&c| !clusters[c as usize].has_parent)
        .collect();
    for run in roots.chunks(16) {
        sibling_sets.push(SiblingSet {
            members: run.to_vec(),
            parent_bounds: Sphere::merged_all(run.iter().map(|&c| clusters[c as usize].lod_bounds)),
            parent_error: ROOT_PARENT_ERROR,
            round: ROOT_ROUND,
            generated: 0..0,
        });
    }

    let mut packed_index = vec![0u32; clusters.len()];
    let mut leaves = Vec::with_capacity(sibling_sets.len());
    for set in &sibling_sets {
        let first_cluster = out.clusters.len() as u32;
        for &member in &set.members {
            packed_index[member as usize] = out.clusters.len() as u32;
            let packed = pack_cluster(
                welded,
                &clusters[member as usize],
                &mut out.vertices,
                &mut out.triangles,
            );
            out.clusters.push(packed);
        }
        leaves.push(Leaf {
            level: clusters[set.members[0] as usize].level,
            first_cluster,
            cluster_count: set.members.len() as u32,
            bounds: Sphere::merged_all(set.members.iter().map(|&m| clusters[m as usize].bounds)),
            parent_lod_bounds: set.parent_bounds,
            parent_error: set.parent_error,
        });
    }

    // Now that clusters have their final positions, record which ones replace each set.
    for (set, leaf) in sibling_sets.iter().zip(&leaves) {
        let first_generated = out.generated.len() as u32;
        out.generated
            .extend(set.generated.clone().map(|c| packed_index[c as usize]));
        out.sets.push(ClusterSet {
            first_cluster: leaf.first_cluster,
            cluster_count: leaf.cluster_count,
            first_generated,
            generated_count: set.generated.len() as u32,
            round: set.round,
            ..Default::default()
        });
    }

    let bounds = Sphere::merged_all(leaves.iter().map(|leaf| leaf.bounds));
    out.bounds_center = bounds.center;
    out.bounds_radius = bounds.radius;

    let hierarchy = build_hierarchy(leaves);
    out.nodes = hierarchy.nodes;
    out.root_node = hierarchy.root;
    out.hierarchy_depth = hierarchy.depth;
    out
}

fn pack_cluster(
    welded: &WeldedMesh,
    cluster: &BuildCluster,
    vertices: &mut Vec<PackedVertex>,
    triangles: &mut Vec<u32>,
) -> PackedCluster {
    let vertex_offset = vertices.len() as u32;
    let triangle_offset = triangles.len() as u32;
    let mut local: FxHashMap<Corner, u8> = FxHashMap::default();

    for triangle in &cluster.triangles {
        let indices = triangle.map(|corner| {
            *local.entry(corner).or_insert_with(|| {
                vertices.push(pack_vertex(welded, corner));
                u8::try_from(vertices.len() as u32 - 1 - vertex_offset)
                    .expect("cluster exceeds 8-bit vertex indexing")
            })
        });
        triangles.push(pack_triangle(indices[0], indices[1], indices[2]));
    }

    PackedCluster {
        center: cluster.bounds.center.to_array(),
        radius: cluster.bounds.radius,
        lod_center: cluster.lod_bounds.center.to_array(),
        lod_radius: cluster.lod_bounds.radius,
        lod_error: cluster.lod_error,
        vertex_offset,
        triangle_offset,
        counts: local.len() as u32
            | (cluster.triangles.len() as u32) << 8
            | encode_uv_density(uv_density(welded, &cluster.triangles)) << CLUSTER_UV_DENSITY_SHIFT,
    }
}

/// Texture repeats per object-space unit across some triangles. It's an area ratio, so seams
/// and slivers with wild UVs count only as much as their small area.
fn uv_density(welded: &WeldedMesh, triangles: &[Triangle]) -> f32 {
    let (mut uv_area, mut area) = (0.0, 0.0);
    for triangle in triangles {
        let p = triangle.map(|corner| welded.positions[corner.position as usize]);
        let uv = triangle.map(|corner| welded.attributes[corner.attribute as usize].uv);
        area += (p[1] - p[0]).cross(p[2] - p[0]).length();
        uv_area += (uv[1] - uv[0]).perp_dot(uv[2] - uv[0]).abs();
    }
    (uv_area / area.max(1.0e-12)).sqrt()
}

fn pack_vertex(welded: &WeldedMesh, corner: Corner) -> PackedVertex {
    let position = welded.positions[corner.position as usize];
    let attribute = welded.attributes[corner.attribute as usize];
    let grid = ((position - welded.origin) / welded.step)
        .round()
        .as_uvec3();
    let uv = ((attribute.uv - welded.uv_origin) / welded.uv_step)
        .round()
        .as_uvec2();
    let normal = encode_octahedral(attribute.normal);
    PackedVertex {
        data: [
            grid.x as u16,
            grid.y as u16,
            grid.z as u16,
            normal[0] as u16,
            normal[1] as u16,
            uv.x as u16,
            uv.y as u16,
            0,
        ],
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::mesh::Vertex;
    use solstice_asset::gpu::unpack_triangle;

    /// A closed subdivided icosphere with bumps, so simplification has real
    /// geometric error to measure.
    pub(crate) fn bumpy_sphere(subdivisions: u32) -> Mesh {
        let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let mut positions: Vec<Vec3> = [
            [-1.0, t, 0.0],
            [1.0, t, 0.0],
            [-1.0, -t, 0.0],
            [1.0, -t, 0.0],
            [0.0, -1.0, t],
            [0.0, 1.0, t],
            [0.0, -1.0, -t],
            [0.0, 1.0, -t],
            [t, 0.0, -1.0],
            [t, 0.0, 1.0],
            [-t, 0.0, -1.0],
            [-t, 0.0, 1.0],
        ]
        .map(|p| Vec3::from_array(p).normalize())
        .to_vec();
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..subdivisions {
            let mut midpoints: FxHashMap<(u32, u32), u32> = FxHashMap::default();
            let mut next = Vec::with_capacity(faces.len() * 4);
            for [a, b, c] in faces {
                let mut midpoint = |i: u32, j: u32| {
                    *midpoints.entry((i.min(j), i.max(j))).or_insert_with(|| {
                        positions.push((positions[i as usize] + positions[j as usize]).normalize());
                        positions.len() as u32 - 1
                    })
                };
                let (ab, bc, ca) = (midpoint(a, b), midpoint(b, c), midpoint(c, a));
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            faces = next;
        }
        let vertices = positions
            .iter()
            .map(|&p| {
                let bump =
                    1.0 + 0.05 * (p.x * 19.0).sin() * (p.y * 23.0).cos() * (p.z * 17.0).sin();
                Vertex {
                    position: p * bump,
                    normal: p,
                    uv: Vec2::new(p.x, p.y),
                }
            })
            .collect();
        let mut mesh = Mesh {
            vertices,
            indices: faces.into_iter().flatten().collect(),
        };
        mesh.compute_smooth_normals();
        mesh
    }

    /// Selects a DAG cut with the same rule as the GPU culling shader, walking the
    /// real node hierarchy. `error_of` projects a (sphere, error) pair to pixels.
    fn select_cut(
        mesh: &VirtualMesh,
        error_of: &dyn Fn(Sphere, f32) -> f32,
        threshold: f32,
    ) -> Vec<u32> {
        let mut selected = Vec::new();
        let mut stack = vec![mesh.root_node];
        while let Some(node_index) = stack.pop() {
            let node = mesh.nodes[node_index as usize];
            let parent_sphere = Sphere {
                center: node.lod_center.into(),
                radius: node.lod_radius,
            };
            if error_of(parent_sphere, node.max_parent_error) <= threshold {
                continue;
            }
            if node.is_leaf == 0 {
                stack.extend(node.first..node.first + node.count);
                continue;
            }
            for index in node.first..node.first + node.count {
                let cluster = mesh.clusters[index as usize];
                let sphere = Sphere {
                    center: cluster.lod_center.into(),
                    radius: cluster.lod_radius,
                };
                if error_of(sphere, cluster.lod_error) <= threshold {
                    selected.push(index);
                }
            }
        }
        selected
    }

    /// Counts edges in the cut that are not shared by exactly two triangles.
    pub(crate) fn open_edges(mesh: &VirtualMesh, selected: &[u32]) -> usize {
        let mut uses: FxHashMap<([u16; 3], [u16; 3]), u32> = FxHashMap::default();
        for &index in selected {
            let cluster = mesh.clusters[index as usize];
            for t in 0..cluster.triangle_count() {
                let packed = mesh.triangles[(cluster.triangle_offset + t) as usize];
                let grid = unpack_triangle(packed).map(|local| {
                    let data = mesh.vertices[cluster.vertex_offset as usize + local as usize].data;
                    [data[0], data[1], data[2]]
                });
                for side in 0..3 {
                    let (a, b) = (grid[side], grid[(side + 1) % 3]);
                    *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
        }
        uses.values().filter(|&&count| count != 2).count()
    }

    fn triangle_total(mesh: &VirtualMesh, selected: &[u32]) -> u32 {
        selected
            .iter()
            .map(|&c| mesh.clusters[c as usize].triangle_count())
            .sum()
    }

    fn built(subdivisions: u32) -> VirtualMesh {
        build_virtual_mesh(&bumpy_sphere(subdivisions), &BuildOptions::default())
    }

    /// A flat 2 by 2 quad with UVs running 0 to 1 repeats once per 2 units.
    #[test]
    fn uv_density_measures_repeats_per_unit() {
        let welded = WeldedMesh {
            positions: vec![Vec3::ZERO, Vec3::X * 2.0, Vec3::Y * 2.0],
            attributes: vec![
                Attribute {
                    normal: Vec3::Z,
                    uv: Vec2::ZERO,
                },
                Attribute {
                    normal: Vec3::Z,
                    uv: Vec2::X,
                },
                Attribute {
                    normal: Vec3::Z,
                    uv: Vec2::Y,
                },
            ],
            triangles: Vec::new(),
            origin: Vec3::ZERO,
            step: 1.0,
            uv_origin: Vec2::ZERO,
            uv_step: Vec2::ONE,
        };
        let corner = |index| Corner {
            position: index,
            attribute: index,
        };
        let density = uv_density(&welded, &[[corner(0), corner(1), corner(2)]]);
        assert!((density - 0.5).abs() < 1.0e-5, "{density}");
    }

    #[test]
    fn clusters_respect_format_limits() {
        let mesh = built(5);
        for cluster in &mesh.clusters {
            assert!(cluster.triangle_count() as usize <= MAX_CLUSTER_TRIANGLES);
            assert!(cluster.vertex_count() as usize <= MAX_CLUSTER_VERTICES);
            assert!(cluster.triangle_count() > 0);
        }
    }

    #[test]
    fn dag_reduces_to_a_small_root_set() {
        let mesh = built(6);
        let coarsest = select_cut(&mesh, &|_, error| error, 1.0e20);
        assert!(
            triangle_total(&mesh, &coarsest) < 1000,
            "coarsest cut still has {} triangles of {}",
            triangle_total(&mesh, &coarsest),
            mesh.source_triangles
        );
    }

    #[test]
    fn finest_cut_is_exactly_the_source_mesh() {
        let mesh = built(5);
        let finest = select_cut(&mesh, &|_, error| error, 0.0);
        assert_eq!(
            u64::from(triangle_total(&mesh, &finest)),
            mesh.source_triangles
        );
    }

    /// The defining property of the whole system: any uniform error threshold gives
    /// a closed surface, even though clusters come from many different LOD levels.
    #[test]
    fn every_uniform_cut_is_watertight() {
        let mesh = built(6);
        let mut previous_triangles = u32::MAX;
        for threshold in [
            0.0, 1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 10.0,
        ] {
            let cut = select_cut(&mesh, &|_, error| error, threshold);
            assert!(!cut.is_empty(), "threshold {threshold} selected nothing");
            assert_eq!(
                open_edges(&mesh, &cut),
                0,
                "cracks at threshold {threshold}"
            );
            let triangles = triangle_total(&mesh, &cut);
            assert!(
                triangles <= previous_triangles,
                "coarser cut got denser at {threshold}"
            );
            previous_triangles = triangles;
        }
    }

    /// The realistic case: the threshold varies across the mesh with distance from
    /// the camera, so a single frame mixes levels. It must still be watertight.
    #[test]
    fn perspective_cuts_are_watertight_from_any_viewpoint() {
        let mesh = built(6);
        let viewpoints = [
            Vec3::new(0.0, 0.0, 1.5),
            Vec3::new(3.0, 0.5, 0.0),
            Vec3::new(-1.2, 1.2, 1.2),
            Vec3::new(0.0, -8.0, 0.0),
            Vec3::new(40.0, 40.0, 40.0),
        ];
        for viewpoint in viewpoints {
            let project = |sphere: Sphere, error: f32| {
                let distance = (sphere.center.distance(viewpoint) - sphere.radius).max(0.01);
                1000.0 * error / distance
            };
            let cut = select_cut(&mesh, &project, 1.0);
            assert!(!cut.is_empty());
            assert_eq!(open_edges(&mesh, &cut), 0, "cracks seen from {viewpoint}");
        }
    }

    #[test]
    fn nearer_viewpoints_select_more_triangles() {
        let mesh = built(6);
        let triangles_at = |distance: f32| {
            let viewpoint = Vec3::Z * distance;
            let project = |sphere: Sphere, error: f32| {
                1000.0 * error / (sphere.center.distance(viewpoint) - sphere.radius).max(0.01)
            };
            triangle_total(&mesh, &select_cut(&mesh, &project, 1.0))
        };
        assert!(triangles_at(2.0) > triangles_at(50.0));
    }

    #[test]
    fn error_and_bounds_grow_monotonically_toward_the_root() {
        let mesh = built(5);
        fn walk(mesh: &VirtualMesh, node: u32) {
            let packed = mesh.nodes[node as usize];
            let parent = Sphere {
                center: packed.lod_center.into(),
                radius: packed.lod_radius,
            };
            if packed.is_leaf != 0 {
                for c in packed.first..packed.first + packed.count {
                    let cluster = mesh.clusters[c as usize];
                    assert!(cluster.lod_error <= packed.max_parent_error);
                    if packed.max_parent_error < ROOT_PARENT_ERROR {
                        let own = Sphere {
                            center: cluster.lod_center.into(),
                            radius: cluster.lod_radius,
                        };
                        assert!(
                            parent.contains(own),
                            "parent LOD sphere does not contain child"
                        );
                    }
                }
                return;
            }
            for child in packed.first..packed.first + packed.count {
                walk(mesh, child);
            }
        }
        walk(&mesh, mesh.root_node);
    }

    #[test]
    fn empty_mesh_builds_an_empty_result() {
        let mesh = build_virtual_mesh(&Mesh::default(), &BuildOptions::default());
        assert!(mesh.clusters.is_empty());
        assert!(mesh.nodes.is_empty());
    }

    /// Every triangle has its own attributes, as on a flat-shaded or heavily seamed model,
    /// so a full cluster holds three vertices per triangle: more than 8 bits can index.
    #[test]
    fn models_made_of_seams_build_and_stay_watertight() {
        let source = bumpy_sphere(5);
        let mut mesh = Mesh::default();
        for (face, triangle) in source.indices.chunks_exact(3).enumerate() {
            for &index in triangle {
                let mut vertex = source.vertices[index as usize];
                vertex.uv = Vec2::new(face as f32, (face % 7) as f32);
                mesh.indices.push(mesh.vertices.len() as u32);
                mesh.vertices.push(vertex);
            }
        }
        let built = build_virtual_mesh(&mesh, &BuildOptions::default());
        for cluster in &built.clusters {
            assert!(cluster.vertex_count() as usize <= MAX_CLUSTER_VERTICES);
        }
        for threshold in [0.0, 0.01, 0.1, 10.0] {
            let cut = select_cut(&built, &|_, error| error, threshold);
            assert_eq!(
                open_edges(&built, &cut),
                0,
                "cracks at threshold {threshold}"
            );
        }
    }

    #[test]
    fn single_triangle_builds() {
        let vertex = |x: f32, y: f32| Vertex {
            position: Vec3::new(x, y, 0.0),
            normal: Vec3::Z,
            uv: Vec2::ZERO,
        };
        let mesh = Mesh {
            vertices: vec![vertex(0.0, 0.0), vertex(1.0, 0.0), vertex(0.0, 1.0)],
            indices: vec![0, 1, 2],
        };
        let built = build_virtual_mesh(&mesh, &BuildOptions::default());
        assert_eq!(built.clusters.len(), 1);
        assert_eq!(built.source_triangles, 1);
    }
}
