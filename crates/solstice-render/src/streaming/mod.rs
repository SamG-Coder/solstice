//! Moves pages of geometry into and out of the GPU's page pool.
//!
//! Culling records which pages each frame wants ([`residency`] decides what that means), the host
//! fetches them however suits its platform, and this module puts them on the GPU. Installing
//! rewrites records the GPU already holds, between frames, so a frame never sees half a page.

pub mod residency;

use crate::scene::GpuScene;
use crate::Gpu;
use residency::{Install, Residency};
use solstice_asset::gpu::{CLUSTER_CHILDREN_MISSING, NODE_INSTALLED, NODE_LEAF};
use solstice_asset::paged::{ClusterSet, PageInfo, PagedScene, PAGE_VERTICES, PAGE_WORDS};
use solstice_asset::{PackedCluster, PackedNode};
use std::collections::HashMap;

/// Byte offset of `is_leaf` within a hierarchy node.
const NODE_FLAGS_OFFSET: u64 = std::mem::offset_of!(PackedNode, is_leaf) as u64;
/// Close runs of changed cluster records are written as one, as each write has a cost.
const MERGE_GAP_CLUSTERS: usize = 8;

/// Keeps the GPU's cluster and node records in step with which pages are installed.
pub struct Streamer {
    pages: Vec<PageInfo>,
    sets: Vec<ClusterSet>,
    generated: Vec<u32>,
    /// The GPU's cluster records. An uninstalled cluster's offsets are local to its page.
    clusters: Vec<PackedCluster>,
    residency: Residency,
    /// Page data that has come but hasn't gone to the GPU.
    arrived: HashMap<u32, Vec<u8>>,
    dirty_clusters: Vec<u32>,
    /// Pool slot of each installed page.
    slots: Vec<u32>,
    frame: u32,
    /// Counts up by one for each change to what's installed.
    generation: u32,
}

impl Streamer {
    /// Starts with nothing installed and a pool of `slots`.
    pub fn new(scene: &PagedScene, slots: u32) -> Self {
        Self {
            pages: scene.pages.clone(),
            sets: scene.sets.clone(),
            generated: scene.generated.clone(),
            clusters: scene.clusters.clone(),
            residency: Residency::new(scene.page_dependencies(), scene.pinned_pages, slots),
            arrived: HashMap::new(),
            dirty_clusters: Vec::new(),
            slots: vec![u32::MAX; scene.pages.len()],
            frame: 0,
            generation: 0,
        }
    }

    pub fn residency(&self) -> &Residency {
        &self.residency
    }

    /// Changes each time pages come or go, so cached results can tell they've gone stale.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Installs a page at once, which must have all its dependencies. For scene setup.
    pub fn install_now(&mut self, gpu: &Gpu, scene: &GpuScene, page: u32, data: &[u8]) {
        let slot = self.residency.install_now(page);
        self.apply_install(gpu, scene, page, slot, data);
    }

    /// Records one frame's requests, and returns the pages the host should start fetching.
    pub fn plan(&mut self, requests: &[u32], max_fetching: u32) -> Vec<u32> {
        self.frame = self.frame.wrapping_add(1);
        self.residency.plan(self.frame, requests, max_fetching)
    }

    /// Takes an unpacked page from the host. It installs once its parents are in and there's
    /// room. Returns false if the page isn't wanted any more or is the wrong size.
    pub fn deliver(&mut self, page: u32, data: Vec<u8>) -> bool {
        let right_size = self
            .pages
            .get(page as usize)
            .is_some_and(|info| info.raw_bytes() == data.len());
        if !right_size || !self.residency.arrived(page) {
            return false;
        }
        self.arrived.insert(page, data);
        true
    }

    /// A fetch failed. The page may be requested again.
    pub fn fetch_failed(&mut self, page: u32) {
        self.residency.failed(page);
    }

    /// Installs waiting pages, at most `limit` of them, then writes all changed records.
    pub fn install_waiting(&mut self, gpu: &Gpu, scene: &GpuScene, limit: u32) {
        self.arrived
            .retain(|&page, _| self.residency.is_waiting(page));
        for _ in 0..limit {
            let Some(Install {
                page,
                slot,
                evicted,
            }) = self.residency.next_install(self.frame)
            else {
                break;
            };
            if let Some(victim) = evicted {
                self.apply_eviction(gpu, scene, victim);
            }
            let data = self
                .arrived
                .remove(&page)
                .expect("a page installs only after its data arrived");
            self.apply_install(gpu, scene, page, slot, &data);
        }
        self.flush(gpu, scene);
    }

    fn apply_install(&mut self, gpu: &Gpu, scene: &GpuScene, page: u32, slot: u32, data: &[u8]) {
        self.generation = self.generation.wrapping_add(1);
        self.slots[page as usize] = slot;
        gpu.queue.write_buffer(
            &scene.vertices,
            u64::from(slot) * u64::from(PAGE_WORDS) * 4,
            data,
        );
        let info = self.pages[page as usize];
        let (vertex_base, triangle_base) = slot_bases(slot, info.vertex_count);
        for index in info.first_set..info.first_set + info.set_count {
            let set = self.sets[index as usize];
            for c in set.first_cluster..set.first_cluster + set.cluster_count {
                let cluster = &mut self.clusters[c as usize];
                cluster.vertex_offset += vertex_base;
                cluster.triangle_offset += triangle_base;
                self.dirty_clusters.push(c);
            }
            for g in set.first_generated..set.first_generated + set.generated_count {
                let parent = self.generated[g as usize];
                self.clusters[parent as usize].counts &= !CLUSTER_CHILDREN_MISSING;
                self.dirty_clusters.push(parent);
            }
            write_node_flags(gpu, scene, set.leaf_node, NODE_LEAF | NODE_INSTALLED | page);
        }
    }

    fn apply_eviction(&mut self, gpu: &Gpu, scene: &GpuScene, page: u32) {
        self.generation = self.generation.wrapping_add(1);
        let slot = self.slots[page as usize];
        let info = self.pages[page as usize];
        let (vertex_base, triangle_base) = slot_bases(slot, info.vertex_count);
        for index in info.first_set..info.first_set + info.set_count {
            let set = self.sets[index as usize];
            // The GPU can't reach these any more, so only this copy needs changing.
            for c in set.first_cluster..set.first_cluster + set.cluster_count {
                let cluster = &mut self.clusters[c as usize];
                cluster.vertex_offset -= vertex_base;
                cluster.triangle_offset -= triangle_base;
            }
            for g in set.first_generated..set.first_generated + set.generated_count {
                let parent = self.generated[g as usize];
                self.clusters[parent as usize].counts |= CLUSTER_CHILDREN_MISSING;
                self.dirty_clusters.push(parent);
            }
            write_node_flags(gpu, scene, set.leaf_node, NODE_LEAF | page);
        }
    }

    /// Writes all cluster records changed since the last call.
    pub fn flush(&mut self, gpu: &Gpu, scene: &GpuScene) {
        for (first, last) in merged_runs(&mut self.dirty_clusters) {
            gpu.queue.write_buffer(
                &scene.clusters,
                u64::from(first) * std::mem::size_of::<PackedCluster>() as u64,
                bytemuck::cast_slice(&self.clusters[first as usize..=last as usize]),
            );
        }
        self.dirty_clusters.clear();
    }

    /// Whether every page has been installed.
    pub fn all_installed(&self) -> bool {
        self.residency.installed_count() as usize == self.residency.page_count()
    }
}

fn write_node_flags(gpu: &Gpu, scene: &GpuScene, node: u32, flags: u32) {
    gpu.queue.write_buffer(
        &scene.nodes,
        u64::from(node) * std::mem::size_of::<PackedNode>() as u64 + NODE_FLAGS_OFFSET,
        bytemuck::bytes_of(&flags),
    );
}

/// Where a slot's vertices and triangles begin, in units of vertices and of triangle words.
/// Triangles start right after the page's own vertices.
fn slot_bases(slot: u32, vertex_count: u32) -> (u32, u32) {
    (
        slot * PAGE_VERTICES,
        slot * PAGE_WORDS
            + vertex_count * (std::mem::size_of::<solstice_asset::PackedVertex>() as u32 / 4),
    )
}

/// Sorts `items` and merges them into inclusive runs, swallowing small gaps.
fn merged_runs(items: &mut [u32]) -> Vec<(u32, u32)> {
    items.sort_unstable();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &item in items.iter() {
        match runs.last_mut() {
            Some((_, last)) if (item - *last) as usize <= MERGE_GAP_CLUSTERS => *last = item,
            _ => runs.push((item, item)),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearby_changes_merge_into_one_write() {
        let mut items = vec![40, 3, 5, 3, 100, 42];
        assert_eq!(merged_runs(&mut items), vec![(3, 5), (40, 42), (100, 100)]);
        assert!(merged_runs(&mut []).is_empty());
    }

    #[test]
    fn triangles_follow_a_pages_vertices_within_its_slot() {
        assert_eq!(slot_bases(0, 10), (0, 40));
        assert_eq!(slot_bases(2, 10), (2 * PAGE_VERTICES, 2 * PAGE_WORDS + 40));
    }
}
