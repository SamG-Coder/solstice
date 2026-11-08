//! Divides a finished scene into pages for on-demand loading.
//!
//! Page ids run coarse to fine, and a page only holds sets from one simplification round. A
//! set's parents always come from a later round, so page dependencies can't form a cycle, and
//! they always point at a lower id.

use solstice_asset::{
    gpu::{CLUSTER_CHILDREN_MISSING, NODE_LEAF},
    paged::{
        upper_mip_count, ClusterSet, LoadedScene, MeshInfo, PageInfo, PagedScene, PAGE_BYTES,
        ROOT_ROUND,
    },
    PackedNode, PackedVertex, SceneAsset, TextureLayer, VirtualMesh,
};

/// One set on its way into a page.
struct Item {
    round: u32,
    mesh: u32,
    /// Position along a space-filling curve within the mesh, so a page covers one area.
    curve: u32,
    set: ClusterSet,
    /// Where this mesh's clusters begin in the scene.
    cluster_base: u32,
}

/// Divides `scene` into pages. Textures split into the base levels and those above.
pub fn paginate(scene: &SceneAsset) -> LoadedScene {
    let mut out = PagedScene {
        materials: scene.materials.clone(),
        instances: scene.instances.clone(),
        lights: scene.lights.clone(),
        viewpoint: scene.viewpoint,
        sea: scene.sea.clone(),
        gameplay: scene.gameplay.clone(),
        hierarchy_depth: 1,
        source_triangles: scene.total_source_triangles(),
        unique_triangles: scene.unique_source_triangles(),
        ..Default::default()
    };

    let mut items = Vec::new();
    for (mesh_index, mesh) in scene.meshes.iter().enumerate() {
        let node_base = out.nodes.len() as u32;
        let cluster_base = out.clusters.len() as u32;
        let generated_base = out.generated.len() as u32;

        out.meshes.push(MeshInfo {
            root_node: (!mesh.clusters.is_empty()).then_some(mesh.root_node + node_base),
            quant_origin: mesh.quant_origin,
            quant_step: mesh.quant_step,
            uv_origin: mesh.uv_origin,
            uv_step: mesh.uv_step,
        });
        out.hierarchy_depth = out.hierarchy_depth.max(mesh.hierarchy_depth);
        out.nodes.extend(mesh.nodes.iter().map(|node| PackedNode {
            first: node.first
                + if node.is_leaf != 0 {
                    cluster_base
                } else {
                    node_base
                },
            ..*node
        }));
        out.clusters.extend_from_slice(&mesh.clusters);
        out.generated
            .extend(mesh.generated.iter().map(|&c| c + cluster_base));

        let leaf_of_set = leaf_nodes(mesh);
        let curve = space_curve(mesh, &leaf_of_set);
        for (index, set) in mesh.sets.iter().enumerate() {
            // A set with no parents to fall back on must always be there.
            let round = if set.generated_count == 0 {
                ROOT_ROUND
            } else {
                set.round
            };
            items.push(Item {
                round,
                mesh: mesh_index as u32,
                curve: curve[index],
                cluster_base,
                set: ClusterSet {
                    first_cluster: set.first_cluster + cluster_base,
                    first_generated: set.first_generated + generated_base,
                    leaf_node: leaf_of_set[index] + node_base,
                    round,
                    ..*set
                },
            });
        }
    }

    // Coarsest first. The roots lead, as `ROOT_ROUND` is the largest round.
    items.sort_by_key(|item| (std::cmp::Reverse(item.round), item.mesh, item.curve));

    let mut pages: Vec<Vec<u8>> = Vec::new();
    let mut vertices: Vec<PackedVertex> = Vec::new();
    let mut triangles: Vec<u32> = Vec::new();
    let mut current = PageInfo::default();
    let mut current_round = None;
    for item in &items {
        let mesh = &scene.meshes[item.mesh as usize];
        let local = |c: u32| &mesh.clusters[(c - item.cluster_base) as usize];
        let range = item.set.first_cluster..item.set.first_cluster + item.set.cluster_count;
        let extra_vertices: u32 = range.clone().map(|c| local(c).vertex_count()).sum();
        let extra_triangles: u32 = range.clone().map(|c| local(c).triangle_count()).sum();
        let extra_bytes = extra_vertices as usize * std::mem::size_of::<PackedVertex>()
            + extra_triangles as usize * 4;
        assert!(
            extra_bytes <= PAGE_BYTES,
            "a sibling set must fit in a page"
        );

        let raw = PageInfo {
            vertex_count: current.vertex_count + extra_vertices,
            triangle_count: current.triangle_count + extra_triangles,
            ..Default::default()
        };
        if current_round.is_some_and(|round| round != item.round) || raw.raw_bytes() > PAGE_BYTES {
            finish_page(
                &mut out,
                &mut pages,
                &mut current,
                &mut vertices,
                &mut triangles,
            );
        }
        current_round = Some(item.round);

        for c in range {
            let cluster = *local(c);
            let (v, t) = (
                cluster.vertex_offset as usize,
                cluster.triangle_offset as usize,
            );
            let packed = &mut out.clusters[c as usize];
            packed.vertex_offset = vertices.len() as u32;
            packed.triangle_offset = triangles.len() as u32;
            vertices.extend_from_slice(&mesh.vertices[v..v + cluster.vertex_count() as usize]);
            triangles.extend_from_slice(&mesh.triangles[t..t + cluster.triangle_count() as usize]);
        }
        current.vertex_count = vertices.len() as u32;
        current.triangle_count = triangles.len() as u32;
        current.set_count += 1;
        out.sets.push(ClusterSet {
            page: out.pages.len() as u32,
            ..item.set
        });
        if item.round == ROOT_ROUND {
            out.pinned_pages = out.pages.len() as u32 + 1;
        }
    }
    if current.set_count > 0 {
        finish_page(
            &mut out,
            &mut pages,
            &mut current,
            &mut vertices,
            &mut triangles,
        );
    }

    // Nothing is installed yet: leaves know their page, and generated clusters stand in for
    // the sets simplified into them.
    for set in &out.sets {
        out.nodes[set.leaf_node as usize].is_leaf = NODE_LEAF | set.page;
    }
    for &cluster in &out.generated {
        out.clusters[cluster as usize].counts |= CLUSTER_CHILDREN_MISSING;
    }

    let pinned = out.pinned_pages as usize;
    out.pinned_data = pages[..pinned].concat();
    let upper_textures = split_textures(scene, &mut out);
    LoadedScene {
        scene: out,
        pages: pages.split_off(pinned),
        upper_textures,
        collision: crate::collision::build_collision(scene),
    }
}

fn finish_page(
    out: &mut PagedScene,
    pages: &mut Vec<Vec<u8>>,
    current: &mut PageInfo,
    vertices: &mut Vec<PackedVertex>,
    triangles: &mut Vec<u32>,
) {
    let mut bytes = Vec::with_capacity(current.raw_bytes());
    bytes.extend_from_slice(bytemuck::cast_slice(vertices));
    bytes.extend_from_slice(bytemuck::cast_slice(triangles));
    pages.push(bytes);
    current.first_set = out.sets.len() as u32 - current.set_count;
    out.pages.push(*current);
    *current = PageInfo::default();
    vertices.clear();
    triangles.clear();
}

/// Finds the hierarchy leaf of each sibling set.
fn leaf_nodes(mesh: &VirtualMesh) -> Vec<u32> {
    let mut set_of_first = rustc_hash::FxHashMap::default();
    for (index, set) in mesh.sets.iter().enumerate() {
        set_of_first.insert(set.first_cluster, index);
    }
    let mut leaves = vec![u32::MAX; mesh.sets.len()];
    for (index, node) in mesh.nodes.iter().enumerate() {
        if node.is_leaf != 0 {
            leaves[set_of_first[&node.first]] = index as u32;
        }
    }
    assert!(
        leaves.iter().all(|&leaf| leaf != u32::MAX),
        "every sibling set has a leaf"
    );
    leaves
}

/// Places each set on a Morton curve through the mesh's bounds.
fn space_curve(mesh: &VirtualMesh, leaf_of_set: &[u32]) -> Vec<u32> {
    fn spread(mut x: u32) -> u32 {
        x &= 0x3ff;
        x = (x | x << 16) & 0x030000ff;
        x = (x | x << 8) & 0x0300f00f;
        x = (x | x << 4) & 0x030c30c3;
        (x | x << 2) & 0x09249249
    }
    let centers: Vec<[f32; 3]> = leaf_of_set
        .iter()
        .map(|&leaf| mesh.nodes[leaf as usize].center)
        .collect();
    let mut low = [f32::MAX; 3];
    let mut high = [f32::MIN; 3];
    for center in &centers {
        for axis in 0..3 {
            low[axis] = low[axis].min(center[axis]);
            high[axis] = high[axis].max(center[axis]);
        }
    }
    let extent = (0..3)
        .map(|axis| high[axis] - low[axis])
        .fold(1.0e-6, f32::max);
    centers
        .iter()
        .map(|center| {
            let cell = |axis: usize| ((center[axis] - low[axis]) / extent * 1023.0) as u32;
            spread(cell(0)) | spread(cell(1)) << 1 | spread(cell(2)) << 2
        })
        .collect()
}

/// Moves each layer's levels from [`BASE_TEXTURE_EDGE`](solstice_asset::paged::BASE_TEXTURE_EDGE)
/// down to the manifest, and returns those above it.
fn split_textures(scene: &SceneAsset, out: &mut PagedScene) -> Vec<TextureLayer> {
    out.texture_size = scene.texture_size;
    out.texture_encoding = scene.texture_encoding;
    let upper_count = upper_mip_count(scene.texture_size);
    let mut upper = Vec::with_capacity(scene.textures.len());
    for layer in &scene.textures {
        upper.push(TextureLayer {
            color_mips: layer.color_mips[..upper_count].to_vec(),
            normal_mips: layer.normal_mips[..upper_count].to_vec(),
            material_mips: layer.material_mips[..upper_count].to_vec(),
        });
        out.base_textures.push(TextureLayer {
            color_mips: layer.color_mips[upper_count..].to_vec(),
            normal_mips: layer.normal_mips[upper_count..].to_vec(),
            material_mips: layer.material_mips[upper_count..].to_vec(),
        });
    }
    upper
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::tests::{bumpy_sphere, open_edges};
    use crate::Sphere;
    use crate::{build_virtual_mesh, BuildOptions};
    use glam::{Mat4, Vec3};
    use solstice_asset::gpu::NODE_PAGE_MASK;
    use solstice_asset::{unpack_triangle, Instance, Material};

    fn scene(subdivisions: &[u32]) -> SceneAsset {
        let meshes: Vec<VirtualMesh> = subdivisions
            .iter()
            .map(|&s| build_virtual_mesh(&bumpy_sphere(s), &BuildOptions::default()))
            .collect();
        SceneAsset {
            instances: (0..meshes.len() as u32)
                .map(|mesh| Instance {
                    mesh,
                    material: 0,
                    transform: Mat4::IDENTITY,
                })
                .collect(),
            meshes,
            materials: vec![Material::default()],
            ..Default::default()
        }
    }

    #[test]
    fn every_cluster_lands_in_one_page_with_its_own_data() {
        let source = scene(&[5, 2, 3]);
        let loaded = paginate(&source);
        let paged = &loaded.scene;
        let mut seen = vec![0; paged.clusters.len()];
        for set in &paged.sets {
            let page = paged.pages[set.page as usize];
            for c in set.first_cluster..set.first_cluster + set.cluster_count {
                seen[c as usize] += 1;
                let cluster = paged.clusters[c as usize];
                assert!(cluster.vertex_offset + cluster.vertex_count() <= page.vertex_count);
                assert!(cluster.triangle_offset + cluster.triangle_count() <= page.triangle_count);
            }
        }
        assert!(seen.iter().all(|&count| count == 1));

        // The triangles a page yields are the triangles the mesh built, in position.
        let mut base = 0u32;
        for mesh in &source.meshes {
            for (local, original) in mesh.clusters.iter().enumerate() {
                let cluster = paged.clusters[base as usize + local];
                let set = paged
                    .sets
                    .iter()
                    .find(|s| {
                        (s.first_cluster..s.first_cluster + s.cluster_count)
                            .contains(&(base + local as u32))
                    })
                    .unwrap();
                let bytes = loaded.page(set.page);
                let page = paged.pages[set.page as usize];
                let vertex_bytes = page.vertex_count as usize * 16;
                let (v, t) = bytes.split_at(vertex_bytes);
                let v: &[PackedVertex] = bytemuck::cast_slice(v);
                let t: Vec<u32> = t
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                    .collect();
                for i in 0..original.triangle_count() {
                    let a = t[(cluster.triangle_offset + i) as usize];
                    let b = mesh.triangles[(original.triangle_offset + i) as usize];
                    assert_eq!(a, b);
                    for corner in unpack_triangle(a) {
                        assert_eq!(
                            v[cluster.vertex_offset as usize + corner as usize],
                            mesh.vertices[original.vertex_offset as usize + corner as usize]
                        );
                    }
                }
            }
            base += mesh.clusters.len() as u32;
        }
    }

    #[test]
    fn pages_fit_and_hold_one_round() {
        let loaded = paginate(&scene(&[6, 2]));
        let paged = &loaded.scene;
        assert!(paged.pages.len() > 2);
        for (index, page) in paged.pages.iter().enumerate() {
            assert!(page.raw_bytes() <= PAGE_BYTES);
            assert_eq!(loaded.page(index as u32).len(), page.raw_bytes());
            let sets =
                &paged.sets[page.first_set as usize..(page.first_set + page.set_count) as usize];
            assert!(sets
                .iter()
                .all(|s| s.page == index as u32 && s.round == sets[0].round));
        }
    }

    #[test]
    fn roots_are_pinned_and_dependencies_only_point_coarser() {
        let loaded = paginate(&scene(&[6, 4]));
        let paged = &loaded.scene;
        assert!(paged.pinned_pages > 0);
        assert!(paged.pinned_pages as usize <= paged.pages.len());
        for set in &paged.sets {
            assert_eq!(set.page < paged.pinned_pages, set.round == ROOT_ROUND);
        }
        let dependencies = paged.page_dependencies();
        for (page, list) in dependencies.iter().enumerate() {
            assert!(
                list.iter().all(|&d| (d as usize) < page),
                "page {page}: {list:?}"
            );
            if page < paged.pinned_pages as usize {
                assert!(list.is_empty(), "pinned pages depend on nothing");
            } else {
                assert!(!list.is_empty(), "other pages have parents");
            }
        }
    }

    #[test]
    fn nodes_and_clusters_start_uninstalled() {
        let loaded = paginate(&scene(&[4]));
        let paged = &loaded.scene;
        for set in &paged.sets {
            assert_eq!(
                paged.nodes[set.leaf_node as usize].is_leaf,
                NODE_LEAF | set.page
            );
            assert_eq!(paged.nodes[set.leaf_node as usize].first, set.first_cluster);
        }
        let finest = paged.clusters.iter().filter(|c| c.lod_error == 0.0).count();
        let waiting = paged
            .clusters
            .iter()
            .filter(|c| c.counts & CLUSTER_CHILDREN_MISSING != 0)
            .count();
        assert_eq!(finest + waiting, paged.clusters.len());
    }

    /// Selects a cut with the same rule as `vg/cull.wgsl`, given which pages are installed. A
    /// one-mesh scene keeps the mesh's cluster numbering, so the mesh's data can be read back.
    fn select_resident_cut(
        paged: &PagedScene,
        installed: &[bool],
        error_of: &dyn Fn(Sphere, f32) -> f32,
        threshold: f32,
    ) -> Vec<u32> {
        let mut children_missing = vec![false; paged.clusters.len()];
        for set in &paged.sets {
            if !installed[set.page as usize] {
                let range = set.first_generated as usize
                    ..(set.first_generated + set.generated_count) as usize;
                for &parent in &paged.generated[range] {
                    children_missing[parent as usize] = true;
                }
            }
        }

        let mut selected = Vec::new();
        let mut stack = vec![paged.meshes[0].root_node.unwrap()];
        while let Some(index) = stack.pop() {
            let node = paged.nodes[index as usize];
            let sphere = Sphere {
                center: node.lod_center.into(),
                radius: node.lod_radius,
            };
            if error_of(sphere, node.max_parent_error) <= threshold {
                continue;
            }
            if node.is_leaf == 0 {
                stack.extend(node.first..node.first + node.count);
                continue;
            }
            if !installed[(node.is_leaf & NODE_PAGE_MASK) as usize] {
                continue;
            }
            for c in node.first..node.first + node.count {
                let cluster = paged.clusters[c as usize];
                let sphere = Sphere {
                    center: cluster.lod_center.into(),
                    radius: cluster.lod_radius,
                };
                if error_of(sphere, cluster.lod_error) <= threshold || children_missing[c as usize]
                {
                    selected.push(c);
                }
            }
        }
        selected
    }

    /// Picks a random set of pages that includes every dependency of each page it holds.
    fn random_closed_pages(paged: &PagedScene, next: &mut impl FnMut() -> u64) -> Vec<bool> {
        let dependencies = paged.page_dependencies();
        let keep = next() % 100;
        let mut installed = vec![false; paged.pages.len()];
        // Dependencies have lower ids, so a parent has been decided before its children.
        for page in 0..paged.pages.len() {
            installed[page] = page < paged.pinned_pages as usize
                || (next() % 100 < keep
                    && dependencies[page].iter().all(|&d| installed[d as usize]));
        }
        installed
    }

    fn perspective(viewpoint: Vec3) -> impl Fn(Sphere, f32) -> f32 {
        move |sphere: Sphere, error: f32| {
            let distance = (sphere.center.distance(viewpoint) - sphere.radius).max(0.01);
            1000.0 * error / distance
        }
    }

    /// The heart of streaming: whichever valid set of pages is loaded, from wherever it's seen,
    /// there are no holes, cracks or overlaps.
    #[test]
    fn partly_loaded_scenes_are_watertight_from_any_viewpoint() {
        let source = scene(&[6]);
        let loaded = paginate(&source);
        let paged = &loaded.scene;
        assert!(
            paged.pages.len() > 4,
            "this test needs a mesh of many pages"
        );

        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let viewpoints = [
            Vec3::new(0.0, 0.0, 1.5),
            Vec3::new(3.0, 0.5, 0.0),
            Vec3::new(-1.2, 1.2, 1.2),
            Vec3::new(0.0, -8.0, 0.0),
            Vec3::new(40.0, 40.0, 40.0),
        ];
        let mut partial_cuts = 0;
        for round in 0..200 {
            let installed = random_closed_pages(paged, &mut next);
            let project = perspective(viewpoints[round % viewpoints.len()]);
            let cut = select_resident_cut(paged, &installed, &project, 1.0);
            assert!(!cut.is_empty());
            assert_eq!(open_edges(&source.meshes[0], &cut), 0, "round {round}");
            if installed.iter().any(|&page| !page) {
                partial_cuts += 1;
            }
        }
        assert!(partial_cuts > 100, "too few sets were actually partial");
    }

    #[test]
    fn pinned_pages_alone_cover_the_whole_mesh_and_all_pages_change_nothing() {
        let source = scene(&[6]);
        let paged = &paginate(&source).scene;
        let near = perspective(Vec3::new(0.0, 0.0, 1.5));

        let mut pinned = vec![false; paged.pages.len()];
        pinned[..paged.pinned_pages as usize].fill(true);
        let coarse = select_resident_cut(paged, &pinned, &near, 1.0);
        assert_eq!(open_edges(&source.meshes[0], &coarse), 0);

        let all = vec![true; paged.pages.len()];
        let fine = select_resident_cut(paged, &all, &near, 1.0);
        let triangles = |cut: &[u32]| -> u32 {
            cut.iter()
                .map(|&c| paged.clusters[c as usize].triangle_count())
                .sum()
        };
        assert!(triangles(&coarse) < triangles(&fine));

        // With everything loaded, the flag never overrides the plain LOD rule.
        for &c in &fine {
            let cluster = paged.clusters[c as usize];
            let sphere = Sphere {
                center: cluster.lod_center.into(),
                radius: cluster.lod_radius,
            };
            assert!(near(sphere, cluster.lod_error) <= 1.0);
        }
    }

    /// Loading pages parent-first, never skipping one, keeps every step watertight.
    #[test]
    fn each_step_of_loading_is_watertight() {
        let source = scene(&[6]);
        let paged = &paginate(&source).scene;
        let near = perspective(Vec3::new(0.0, 0.0, 1.5));
        let mut installed = vec![false; paged.pages.len()];
        installed[..paged.pinned_pages as usize].fill(true);
        let mut previous = 0;
        for page in paged.pinned_pages as usize..paged.pages.len() {
            installed[page] = true;
            let cut = select_resident_cut(paged, &installed, &near, 1.0);
            assert_eq!(open_edges(&source.meshes[0], &cut), 0, "page {page}");
            let triangles: u32 = cut
                .iter()
                .map(|&c| paged.clusters[c as usize].triangle_count())
                .sum();
            assert!(triangles >= previous, "loading made the view coarser");
            previous = triangles;
        }
    }
}
