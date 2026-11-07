//! Moves pages of geometry into and out of the GPU's page pool.
//!
//! Culling records which pages each frame wants ([`residency`] decides what that means), the host
//! fetches them however suits its platform, and this module puts them on the GPU. Installing
//! rewrites records the GPU already holds, between frames, so a frame never sees half a page.

mod feedback;
pub mod residency;
pub mod textures;

use crate::scene::{
    starting_materials, unpack_bc5, unpack_bc7, GpuScene, TierArrays, UNPACKED_MAX_EDGE,
};
use crate::Gpu;
use feedback::RequestFeedback;
use residency::{Install, Residency};
use solstice_asset::gpu::{CLUSTER_CHILDREN_MISSING, NODE_INSTALLED, NODE_LEAF};
use solstice_asset::paged::{
    upper_mip_count, ClusterSet, PageInfo, PagedScene, BASE_TEXTURE_EDGE, PAGE_VERTICES, PAGE_WORDS,
};
use solstice_asset::{Material, PackedCluster, PackedNode, TextureEncoding};
use std::collections::HashMap;
use textures::{MipFetch, TextureResidency, TierPlan, TIER_MIDDLE, TIER_TOP};

/// Byte offset of `is_leaf` within a hierarchy node.
const NODE_FLAGS_OFFSET: u64 = std::mem::offset_of!(PackedNode, is_leaf) as u64;
/// Close runs of changed cluster records are written as one, as each write has a cost.
const MERGE_GAP_CLUSTERS: usize = 8;
/// Pages to install per frame. Each costs little, but many at once would be felt as a hitch.
const INSTALLS_PER_FRAME: u32 = 8;
/// Frames between redraws of cached views, which a stream of arriving pages would otherwise
/// make redraw constantly.
const REFRESH_INTERVAL_FRAMES: u32 = 30;

/// Levels of texture to fetch at once. They're much larger than pages, and fewer.
const MAX_TEXTURE_FETCHES: u32 = 4;
/// Texture levels to write per frame.
const TEXTURE_WRITES_PER_FRAME: usize = 2;
/// Sharper than a texel per pixel, matching `TEXTURE_DETAIL` in the renderer. Textures load a
/// little before they would be seen to blur.
const TEXTURE_DEMAND_SCALE: f32 = 1.0 / 0.6;
/// Marks a request's id as a texture level rather than a page.
const TEXTURE_REQUEST_BIT: u32 = 1 << 31;

/// Something for the host to fetch: a range of the scene file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamRequest {
    /// What to pass back to [`crate::Renderer::deliver`].
    pub id: u32,
    /// First byte in the file.
    pub start: u64,
    /// One past the last byte.
    pub end: u64,
    /// Whether the range must be inflated before delivery.
    pub deflated: bool,
}

/// How streaming is going, for on-screen readouts and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamingStats {
    pub pages: u32,
    pub installed: u32,
    /// Size of the page pool.
    pub slots: u32,
    /// Being fetched, or fetched and waiting to install.
    pub in_transit: u32,
    /// Wanted but with no room in the pool.
    pub starved: u32,
    pub evictions: u64,
    /// Bytes of the GPU's page pool.
    pub pool_bytes: u64,
    /// Layers usable at the middle and top tiers.
    pub middle_layers: u32,
    pub top_layers: u32,
    /// Texture levels being fetched.
    pub texture_fetches: u32,
}

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
    feedback: RequestFeedback,
    /// The latest requests read back, not yet planned for.
    fresh_requests: Option<Vec<u32>>,
    slot_count: u32,
    data_offset: u64,
    textures: Option<TextureStreamer>,
    evictions: u64,
    /// Whether what's installed changed since cached views last refreshed.
    changed_since_refresh: bool,
    frames_since_refresh: u32,
}

impl Streamer {
    /// Starts with nothing installed and a pool of `slots`. Textures stream only if a plan is
    /// given, and otherwise stay as they were created.
    pub fn new(gpu: &Gpu, scene: &PagedScene, slots: u32, plan: Option<TierPlan>) -> Self {
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
            feedback: RequestFeedback::new(
                gpu,
                (scene.pages.len() + scene.base_textures.len()) as u32,
            ),
            fresh_requests: None,
            slot_count: slots,
            data_offset: scene.data_offset,
            textures: plan
                .filter(|_| !scene.base_textures.is_empty())
                .map(|plan| TextureStreamer::new(scene, plan)),
            evictions: 0,
            changed_since_refresh: false,
            frames_since_refresh: 0,
        }
    }

    /// The buffer that culling records requests in.
    pub fn requests_buffer(&self) -> &wgpu::Buffer {
        self.feedback.buffer()
    }

    /// Clears the requests for this frame's culling, and finishes textures that are complete.
    pub fn begin_frame(&mut self, gpu: &Gpu, scene: &GpuScene, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(self.feedback.buffer(), 0, None);
        if let Some(textures) = &mut self.textures {
            textures.begin_frame(gpu, scene, encoder);
        }
    }

    /// Queues a copy of this frame's requests, if the last copy has been dealt with.
    pub fn capture_requests(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.feedback.capture(encoder);
    }

    /// Starts reading back requests captured this frame. Call after submitting.
    pub fn map_requests(&mut self) {
        self.feedback.request_map();
    }

    /// One frame's work: picks up requests read back and installs waiting pages.
    pub fn update(&mut self, gpu: &Gpu, scene: &GpuScene) {
        self.frame = self.frame.wrapping_add(1);
        self.frames_since_refresh = self.frames_since_refresh.saturating_add(1);
        if let Some(requests) = self.feedback.take_latest() {
            self.fresh_requests = Some(requests);
        }
        self.install_waiting(gpu, scene, INSTALLS_PER_FRAME);
        if let Some(textures) = &mut self.textures {
            textures.write_waiting(gpu, scene);
        }
    }

    /// What to start fetching now, from the latest requests read back. At most `max_fetching`
    /// pages and a few texture levels are ever outstanding.
    pub fn take_fetches(&mut self, max_fetching: u32) -> Vec<StreamRequest> {
        let Some(requests) = self.fresh_requests.take() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for page in self.plan(&requests, max_fetching) {
            let info = &self.pages[page as usize];
            let start = self.data_offset + info.offset;
            out.push(StreamRequest {
                id: page,
                start,
                end: start + u64::from(info.stored_bytes),
                deflated: true,
            });
        }
        if let Some(textures) = &mut self.textures {
            out.extend(textures.plan(self.frame, &requests));
        }
        out
    }

    /// Takes the bytes of a request: unpacked, if it was deflated. Returns false if they aren't
    /// wanted any more or are the wrong size.
    pub fn deliver_request(&mut self, id: u32, data: Vec<u8>) -> bool {
        if id & TEXTURE_REQUEST_BIT != 0 {
            match &mut self.textures {
                Some(textures) => textures.deliver(id, data),
                None => false,
            }
        } else {
            self.deliver(id, data)
        }
    }

    /// A fetch failed. What it was for may be requested again.
    pub fn request_failed(&mut self, id: u32) {
        if id & TEXTURE_REQUEST_BIT != 0 {
            if let Some(textures) = &mut self.textures {
                textures.failed(id);
            }
        } else {
            self.fetch_failed(id);
        }
    }

    /// Whether shadows and GI should redraw now, because finer geometry has arrived. They
    /// keep what they have otherwise, so this also says when they last changed.
    pub fn should_refresh_cached_views(&mut self) -> bool {
        if !self.changed_since_refresh || self.frames_since_refresh < REFRESH_INTERVAL_FRAMES {
            return false;
        }
        self.changed_since_refresh = false;
        self.frames_since_refresh = 0;
        true
    }

    pub fn stats(&self) -> StreamingStats {
        StreamingStats {
            pages: self.pages.len() as u32,
            installed: self.residency.installed_count(),
            slots: self.slot_count,
            in_transit: self.residency.in_transit(),
            starved: self.residency.starved(),
            evictions: self.evictions,
            pool_bytes: u64::from(self.slot_count) * u64::from(PAGE_WORDS) * 4,
            middle_layers: self.textures.as_ref().map_or(0, |t| t.residency.counts().0),
            top_layers: self.textures.as_ref().map_or(0, |t| t.residency.counts().1),
            texture_fetches: self.textures.as_ref().map_or(0, |t| t.residency.fetching()),
        }
    }

    pub fn residency(&self) -> &Residency {
        &self.residency
    }

    /// Installs a page at once, which must have all its dependencies. For scene setup.
    pub fn install_now(&mut self, gpu: &Gpu, scene: &GpuScene, page: u32, data: &[u8]) {
        let slot = self.residency.install_now(page);
        self.apply_install(gpu, scene, page, slot, data);
    }

    /// Records one frame's requests, and returns the pages the host should start fetching.
    fn plan(&mut self, requests: &[u32], max_fetching: u32) -> Vec<u32> {
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
    fn install_waiting(&mut self, gpu: &Gpu, scene: &GpuScene, limit: u32) {
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
            self.changed_since_refresh = true;
            let data = self
                .arrived
                .remove(&page)
                .expect("a page installs only after its data arrived");
            self.apply_install(gpu, scene, page, slot, &data);
        }
        self.flush(gpu, scene);
    }

    fn apply_install(&mut self, gpu: &Gpu, scene: &GpuScene, page: u32, slot: u32, data: &[u8]) {
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
        self.evictions += 1;
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
}

/// Chooses tier sizes for a scene, on this GPU, within a budget in bytes.
pub fn choose_tiers(gpu: &Gpu, scene: &PagedScene, budget: u64) -> TierPlan {
    let stored_as_blocks = scene.texture_encoding == TextureEncoding::Block;
    let block = stored_as_blocks && gpu.block_compressed_textures;
    let unpack = stored_as_blocks && !gpu.block_compressed_textures;
    TierPlan::choose(
        scene.texture_size,
        scene.base_textures.len() as u32,
        BASE_TEXTURE_EDGE,
        if block {
            TextureEncoding::Block
        } else {
            TextureEncoding::Raw
        },
        budget,
        if unpack { UNPACKED_MAX_EDGE } else { u32::MAX },
    )
}

/// Moves texture levels between the file and the tiers.
struct TextureStreamer {
    residency: TextureResidency,
    /// Each layer's materials, whose tier and slot change together.
    materials: Vec<Material>,
    size: u32,
    encoding: TextureEncoding,
    upper_count: usize,
    /// For each layer, each upper level: its range in the file.
    ranges: Vec<(u64, u64)>,
    /// Fetched levels waiting for their turn to be written.
    waiting: Vec<(MipFetch, Vec<u8>)>,
    /// Layers whose top levels are written, for which lower levels should copy and then draw.
    complete: Vec<(u32, u32)>,
    /// Written tier changes, to apply to materials, whose buffer may be written any time.
    dirty_materials: bool,
}

impl TextureStreamer {
    fn new(scene: &PagedScene, plan: TierPlan) -> Self {
        let upper_count = upper_mip_count(scene.texture_size);
        let layers = scene.base_textures.len();
        let mut ranges = Vec::with_capacity(layers * upper_count);
        for layer in 0..layers {
            for mip in 0..upper_count {
                ranges.push(
                    solstice_asset::file::upper_mip_range(scene, layer, mip).unwrap_or((0, 0)),
                );
            }
        }
        // The unpacked path's base tier might start lower than a tier's scale would suggest,
        // but levels are always addressed from the file's largest.
        Self {
            residency: TextureResidency::new(plan, scene.texture_size, layers as u32),
            materials: starting_materials(scene),
            size: scene.texture_size,
            encoding: scene.texture_encoding,
            upper_count,
            ranges,
            waiting: Vec::new(),
            complete: Vec::new(),
            dirty_materials: false,
        }
    }

    fn plan(&mut self, frame: u32, requests: &[u32]) -> Vec<StreamRequest> {
        let layers = self.ranges.len().checked_div(self.upper_count).unwrap_or(0);
        if layers == 0 || requests.len() < layers {
            return Vec::new();
        }
        // These are stored from the end of the array, so that pages need no offset.
        let demands: Vec<u32> = (0..layers)
            .map(|layer| {
                let edge = requests[requests.len() - 1 - layer];
                (edge as f32 * TEXTURE_DEMAND_SCALE) as u32
            })
            .collect();
        self.residency
            .plan(frame, &demands, MAX_TEXTURE_FETCHES)
            .into_iter()
            .filter_map(|fetch| {
                let index = fetch.layer as usize * self.upper_count + fetch.mip as usize;
                let &(start, end) = self.ranges.get(index)?;
                Some(StreamRequest {
                    id: TEXTURE_REQUEST_BIT | fetch.layer << 8 | fetch.mip,
                    start,
                    end,
                    deflated: false,
                })
            })
            .collect()
    }

    fn decode_id(id: u32) -> MipFetch {
        MipFetch {
            layer: (id & !TEXTURE_REQUEST_BIT) >> 8,
            mip: id & 0xff,
        }
    }

    fn deliver(&mut self, id: u32, data: Vec<u8>) -> bool {
        let fetch = Self::decode_id(id);
        let edge = self.size >> fetch.mip;
        let expected = self.encoding.color_mip_bytes(edge) + 2 * self.encoding.data_mip_bytes(edge);
        if data.len() != expected {
            self.residency.failed(fetch);
            return false;
        }
        self.waiting.push((fetch, data));
        true
    }

    fn failed(&mut self, id: u32) {
        self.residency.failed(Self::decode_id(id));
    }

    /// Writes a few fetched levels to their arrays.
    fn write_waiting(&mut self, gpu: &Gpu, scene: &GpuScene) {
        for _ in 0..TEXTURE_WRITES_PER_FRAME.min(self.waiting.len()) {
            let (fetch, data) = self.waiting.remove(0);
            let Some(target) = self.residency.arrived(fetch) else {
                continue;
            };
            let tier: &TierArrays = match target.tier {
                TIER_TOP => &scene.textures.top,
                _ => &scene.textures.middle,
            };
            let edge = self.size >> fetch.mip;
            let color_bytes = self.encoding.color_mip_bytes(edge);
            let data_bytes = self.encoding.data_mip_bytes(edge);
            let color = &data[..color_bytes];
            let normal = &data[color_bytes..color_bytes + data_bytes];
            let material = &data[color_bytes + data_bytes..];
            let block = scene.textures.block;
            if scene.textures.unpack {
                tier.write_level(
                    gpu,
                    block,
                    target.slot,
                    target.tier_mip,
                    [
                        &unpack_bc7(edge, color),
                        &unpack_bc5(edge, normal),
                        &unpack_bc5(edge, material),
                    ],
                );
            } else {
                tier.write_level(
                    gpu,
                    block,
                    target.slot,
                    target.tier_mip,
                    [color, normal, material],
                );
            }
            if target.completes {
                self.complete.push((fetch.layer, target.tier));
            }
        }
    }

    /// Copies lower levels up for tiers that have all their own, then lets materials use them.
    fn begin_frame(&mut self, gpu: &Gpu, scene: &GpuScene, encoder: &mut wgpu::CommandEncoder) {
        for (layer, tier) in std::mem::take(&mut self.complete) {
            let arrays = &scene.textures;
            if tier == TIER_MIDDLE {
                arrays
                    .middle
                    .copy_lower_levels(encoder, &arrays.base, arrays.block, layer, layer);
            } else {
                let slot = self.residency.top_slot(layer).unwrap_or(0);
                arrays
                    .top
                    .copy_lower_levels(encoder, &arrays.middle, arrays.block, layer, slot);
            }
            self.residency.commit(layer, tier);
        }
        for change in self.residency.take_changes() {
            for material in &mut self.materials {
                if material.texture_layer == change.layer as i32 {
                    material.texture_tier = change.tier;
                    material.texture_slot = change.slot;
                    self.dirty_materials = true;
                }
            }
        }
        if self.dirty_materials {
            self.dirty_materials = false;
            gpu.queue
                .write_buffer(&scene.materials, 0, bytemuck::cast_slice(&self.materials));
        }
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
