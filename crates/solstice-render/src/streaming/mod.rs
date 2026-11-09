//! Moves pages of geometry into and out of the GPU's page pool.
//!
//! Culling records which pages each frame wants ([`residency`] decides what that means), the host
//! fetches them however suits its platform, and this module puts them on the GPU. Installing
//! rewrites records the GPU already holds, between frames, so a frame never sees half a page.

mod feedback;
pub mod residency;
pub mod textures;

use crate::regions::{binding_slots, page_bands, RegionPlan};
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
const INSTALLS_PER_FRAME: u32 = 32;
/// The same for a scene with pools by level of detail, for which a new view is thousands of pages.
const INSTALLS_PER_FRAME_LARGE: u32 = 192;
/// Frames between plans of such a scene. Each looks over more than a million pages, and new
/// requests a frame or two later change little.
const FRAMES_PER_PLAN_LARGE: u32 = 3;
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
    /// For a page of a scene that keeps its cluster records outside its manifest: where the
    /// page's records are among them, as a range of bytes. They go with the page, to
    /// [`crate::Renderer::deliver_page`].
    pub clusters: Option<(u64, u64)>,
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
    /// For a scene with regions: pages installed and slots in each band, nearest first.
    pub bands: Option<[(u32, u32); crate::regions::BANDS]>,
}

/// Keeps the GPU's cluster and node records in step with which pages are installed.
pub struct Streamer {
    pages: Vec<PageInfo>,
    sets: Vec<ClusterSet>,
    generated: Vec<u32>,
    /// The GPU's cluster records. An uninstalled cluster's offsets are local to its page.
    clusters: ClusterCopy,
    residency: Residency,
    /// Page data that has come but hasn't gone to the GPU.
    arrived: HashMap<u32, (Vec<u8>, Vec<u8>)>,
    dirty_clusters: Vec<u32>,
    /// Pool slot of each installed page.
    slots: Vec<u32>,
    regions: Option<RegionPlan>,
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
    last_plan_frame: u32,
}

/// The host's copy of the cluster records, which installing pages changes.
enum ClusterCopy {
    /// Every record, as the GPU has them, in a scene that holds them all.
    Whole(Vec<PackedCluster>),
    /// Only the records of pages on the GPU, in a scene too large to hold them all.
    Resident(Resident),
}

/// Cluster records for only the pages on the GPU. Each pool slot has room for as many records as
/// a page can have, at `slot * stride`, and leaf nodes are pointed there as their pages install.
struct Resident {
    stride: u32,
    /// Each slot's records as the GPU has them. Empty for a slot with no page.
    records: Vec<Vec<PackedCluster>>,
    /// How many records come before each page's, in the order pages hold them, with one more
    /// entry for the total.
    page_first: Vec<u32>,
    /// The sets by their first cluster, to find the set that holds a cluster.
    by_cluster: Vec<u32>,
}

impl Resident {
    fn new(scene: &PagedScene, slots: u32) -> Self {
        let mut page_first = Vec::with_capacity(scene.pages.len() + 1);
        let mut total = 0u32;
        for page in 0..scene.pages.len() as u32 {
            page_first.push(total);
            total += scene.page_cluster_count(page);
        }
        page_first.push(total);
        let mut by_cluster: Vec<u32> = (0..scene.sets.len() as u32).collect();
        by_cluster.sort_unstable_by_key(|&set| {
            let set = &scene.sets[set as usize];
            (set.first_cluster, set.cluster_count)
        });
        Self {
            stride: scene.max_page_clusters().max(1),
            records: vec![Vec::new(); slots as usize],
            page_first,
            by_cluster,
        }
    }

    /// The page holding a cluster, and the cluster's place among that page's records.
    fn locate(&self, sets: &[ClusterSet], pages: &[PageInfo], cluster: u32) -> (u32, u32) {
        let after = self
            .by_cluster
            .partition_point(|&set| sets[set as usize].first_cluster <= cluster);
        let index = self.by_cluster[after - 1];
        let set = &sets[index as usize];
        let first_set = pages[set.page as usize].first_set;
        let before: u32 = sets[first_set as usize..index as usize]
            .iter()
            .map(|earlier| earlier.cluster_count)
            .sum();
        (set.page, before + cluster - set.first_cluster)
    }

    /// A page's records among all of them, as a range of bytes.
    fn byte_range(&self, page: u32) -> (u64, u64) {
        let bytes = std::mem::size_of::<PackedCluster>() as u64;
        (
            u64::from(self.page_first[page as usize]) * bytes,
            u64::from(self.page_first[page as usize + 1]) * bytes,
        )
    }

    /// Sets or clears a flag on a cluster that is on the GPU, and notes that it changed.
    fn mark(
        &mut self,
        sets: &[ClusterSet],
        pages: &[PageInfo],
        slots: &[u32],
        cluster: u32,
        missing: bool,
        dirty: &mut Vec<u32>,
    ) {
        let (page, place) = self.locate(sets, pages, cluster);
        let slot = slots[page as usize];
        let record = self
            .records
            .get_mut(slot as usize)
            .and_then(|records| records.get_mut(place as usize));
        if let Some(record) = record {
            if missing {
                record.counts |= CLUSTER_CHILDREN_MISSING;
            } else {
                record.counts &= !CLUSTER_CHILDREN_MISSING;
            }
            dirty.push(slot * self.stride + place);
        }
    }
}

/// Writes cluster records to wherever the GPU keeps them, starting at record `first`.
fn write_clusters(gpu: &Gpu, scene: &GpuScene, chunk: u32, first: u32, records: &[PackedCluster]) {
    let bytes = std::mem::size_of::<PackedCluster>() as u64;
    let head = (chunk.saturating_sub(first) as usize).min(records.len());
    if head > 0 {
        gpu.queue.write_buffer(
            &scene.clusters,
            u64::from(first) * bytes,
            bytemuck::cast_slice(&records[..head]),
        );
    }
    if head < records.len() {
        let more = scene
            .clusters_more
            .as_ref()
            .expect("a chunk means a second binding");
        let at = first.max(chunk) - chunk;
        gpu.queue.write_buffer(
            more,
            u64::from(at) * bytes,
            bytemuck::cast_slice(&records[head..]),
        );
    }
}

/// Where a node's first child or cluster is. Its count and flags follow.
const NODE_FIRST_OFFSET: u64 = std::mem::offset_of!(PackedNode, first) as u64;

/// Points a leaf node at its set's records, and sets its flags.
fn write_leaf(gpu: &Gpu, scene: &GpuScene, node: u32, first: u32, count: u32, flags: u32) {
    gpu.queue.write_buffer(
        &scene.nodes,
        u64::from(node) * std::mem::size_of::<PackedNode>() as u64 + NODE_FIRST_OFFSET,
        bytemuck::cast_slice(&[first, count, flags]),
    );
}

impl Streamer {
    /// Starts with nothing installed and a pool of `slots`. Textures stream only if a plan is
    /// given, and otherwise stay as they were created.
    pub fn new(
        gpu: &Gpu,
        scene: &PagedScene,
        slots: u32,
        plan: Option<TierPlan>,
        regions: Option<RegionPlan>,
    ) -> Self {
        let residency = match regions {
            Some(regions) => Residency::with_bands(
                scene.page_dependencies(),
                scene.pinned_pages,
                page_bands(scene, binding_slots(&gpu.limits)),
                &regions.band_slots,
            ),
            None => Residency::new(scene.page_dependencies(), scene.pinned_pages, slots),
        };
        let slot_count = regions.map_or(slots, |r| r.band_slots.iter().sum());
        Self {
            pages: scene.pages.clone(),
            sets: scene.sets.clone(),
            generated: scene.generated.clone(),
            clusters: match scene.external {
                Some(_) => ClusterCopy::Resident(Resident::new(scene, slot_count)),
                None => ClusterCopy::Whole(scene.clusters.clone()),
            },
            residency,
            regions,
            arrived: HashMap::new(),
            dirty_clusters: Vec::new(),
            slots: vec![u32::MAX; scene.pages.len()],
            frame: 0,
            feedback: RequestFeedback::new(
                gpu,
                (scene.pages.len() + scene.base_textures.len()) as u32,
            ),
            fresh_requests: None,
            slot_count,
            data_offset: scene.data_offset,
            textures: plan
                .filter(|_| !scene.base_textures.is_empty())
                .map(|plan| TextureStreamer::new(scene, plan)),
            evictions: 0,
            changed_since_refresh: false,
            frames_since_refresh: 0,
            last_plan_frame: 0,
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
        let installs = if self.regions.is_some() {
            INSTALLS_PER_FRAME_LARGE
        } else {
            INSTALLS_PER_FRAME
        };
        self.install_waiting(gpu, scene, installs);
        if let Some(textures) = &mut self.textures {
            textures.write_waiting(gpu, scene);
        }
    }

    /// What to start fetching now, from the latest requests read back. At most `max_fetching`
    /// pages and a few texture levels are ever outstanding.
    pub fn take_fetches(&mut self, max_fetching: u32) -> Vec<StreamRequest> {
        let due = self.regions.is_none()
            || self.frame.wrapping_sub(self.last_plan_frame) >= FRAMES_PER_PLAN_LARGE;
        let requests = if due {
            self.fresh_requests.take()
        } else {
            None
        };
        if requests.is_some() {
            self.last_plan_frame = self.frame;
        }
        let pages = match &requests {
            Some(requests) => self.plan(requests, max_fetching),
            None => self.residency.refill(max_fetching),
        };
        let mut out = Vec::new();
        for page in pages {
            let info = &self.pages[page as usize];
            let start = self.data_offset + info.offset;
            out.push(StreamRequest {
                id: page,
                start,
                end: start + u64::from(info.stored_bytes),
                deflated: true,
                clusters: match &self.clusters {
                    ClusterCopy::Resident(resident) => Some(resident.byte_range(page)),
                    ClusterCopy::Whole(_) => None,
                },
            });
        }
        if let (Some(textures), Some(requests)) = (&mut self.textures, &requests) {
            out.extend(textures.plan(self.frame, requests));
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
            bands: self.regions.map(|regions| {
                let installed = self.residency.installed_by_band();
                std::array::from_fn(|band| (installed[band], regions.band_slots[band]))
            }),
        }
    }

    pub fn residency(&self) -> &Residency {
        &self.residency
    }

    /// Installs a page at once, which must have all its dependencies. For scene setup.
    pub fn install_now(
        &mut self,
        gpu: &Gpu,
        scene: &GpuScene,
        page: u32,
        data: &[u8],
        clusters: &[u8],
    ) {
        let slot = self.residency.install_now(page);
        self.apply_install(gpu, scene, page, slot, data, clusters);
    }

    /// Records one frame's requests, and returns the pages the host should start fetching.
    fn plan(&mut self, requests: &[u32], max_fetching: u32) -> Vec<u32> {
        self.residency.plan(self.frame, requests, max_fetching)
    }

    /// Takes an unpacked page from the host. It installs once its parents are in and there's
    /// room. Returns false if the page isn't wanted any more or is the wrong size.
    pub fn deliver(&mut self, page: u32, data: Vec<u8>) -> bool {
        self.deliver_page(page, data, Vec::new())
    }

    /// As [`Self::deliver`], with the page's cluster records, for a scene that keeps them outside
    /// its manifest: the bytes [`StreamRequest::clusters`] names.
    pub fn deliver_page(&mut self, page: u32, data: Vec<u8>, clusters: Vec<u8>) -> bool {
        let right_size = self
            .pages
            .get(page as usize)
            .is_some_and(|info| info.raw_bytes() == data.len());
        let right_records = match &self.clusters {
            ClusterCopy::Resident(resident) if right_size => {
                let (start, end) = resident.byte_range(page);
                (end - start) as usize == clusters.len()
            }
            _ => true,
        };
        if !right_size || !right_records || !self.residency.arrived(page) {
            return false;
        }
        self.arrived.insert(page, (data, clusters));
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
            let install = self.residency.next_install(self.frame);
            // Even if nothing could install: the pages let go are no longer on the GPU's books.
            for released in self.residency.take_released() {
                self.apply_eviction(gpu, scene, released);
            }
            let Some(Install {
                page,
                slot,
                evicted,
            }) = install
            else {
                break;
            };
            if let Some(victim) = evicted {
                self.apply_eviction(gpu, scene, victim);
            }
            self.changed_since_refresh = true;
            let (data, clusters) = self
                .arrived
                .remove(&page)
                .expect("a page installs only after its data arrived");
            self.apply_install(gpu, scene, page, slot, &data, &clusters);
        }
        self.flush(gpu, scene);
    }

    fn apply_install(
        &mut self,
        gpu: &Gpu,
        scene: &GpuScene,
        page: u32,
        slot: u32,
        data: &[u8],
        clusters: &[u8],
    ) {
        self.slots[page as usize] = slot;
        let (band, local) = locate_slot(self.regions.as_ref(), slot);
        gpu.queue.write_buffer(
            scene.pool(band),
            u64::from(local) * u64::from(PAGE_WORDS) * 4,
            data,
        );
        let info = self.pages[page as usize];
        let (vertex_base, triangle_base) = slot_bases(band, local, info.vertex_count);
        let chunk = self.regions.map_or(u32::MAX, |r| r.cluster_chunk);
        let installed = NODE_LEAF | NODE_INSTALLED | page;
        match &mut self.clusters {
            ClusterCopy::Whole(records) => {
                for index in info.first_set..info.first_set + info.set_count {
                    let set = self.sets[index as usize];
                    for c in set.first_cluster..set.first_cluster + set.cluster_count {
                        let cluster = &mut records[c as usize];
                        cluster.vertex_offset += vertex_base;
                        cluster.triangle_offset += triangle_base;
                        self.dirty_clusters.push(c);
                    }
                    for g in set.first_generated..set.first_generated + set.generated_count {
                        let parent = self.generated[g as usize];
                        records[parent as usize].counts &= !CLUSTER_CHILDREN_MISSING;
                        self.dirty_clusters.push(parent);
                    }
                    write_node_flags(gpu, scene, set.leaf_node, installed);
                }
            }
            ClusterCopy::Resident(resident) => {
                let mut records: Vec<PackedCluster> = clusters
                    .chunks_exact(std::mem::size_of::<PackedCluster>())
                    .map(bytemuck::pod_read_unaligned)
                    .collect();
                for cluster in &mut records {
                    cluster.vertex_offset += vertex_base;
                    cluster.triangle_offset += triangle_base;
                }
                let mut first = slot * resident.stride;
                write_clusters(gpu, scene, chunk, first, &records);
                resident.records[slot as usize] = records;
                for index in info.first_set..info.first_set + info.set_count {
                    let set = self.sets[index as usize];
                    for g in set.first_generated..set.first_generated + set.generated_count {
                        let parent = self.generated[g as usize];
                        resident.mark(
                            &self.sets,
                            &self.pages,
                            &self.slots,
                            parent,
                            false,
                            &mut self.dirty_clusters,
                        );
                    }
                    write_leaf(
                        gpu,
                        scene,
                        set.leaf_node,
                        first,
                        set.cluster_count,
                        installed,
                    );
                    first += set.cluster_count;
                }
            }
        }
    }

    fn apply_eviction(&mut self, gpu: &Gpu, scene: &GpuScene, page: u32) {
        self.evictions += 1;
        let slot = self.slots[page as usize];
        // Before its parents are marked, so that nothing can find this slot through the page.
        self.slots[page as usize] = u32::MAX;
        let info = self.pages[page as usize];
        let (band, local) = locate_slot(self.regions.as_ref(), slot);
        let (vertex_base, triangle_base) = slot_bases(band, local, info.vertex_count);
        if let ClusterCopy::Resident(resident) = &mut self.clusters {
            resident.records[slot as usize] = Vec::new();
        }
        for index in info.first_set..info.first_set + info.set_count {
            let set = self.sets[index as usize];
            match &mut self.clusters {
                ClusterCopy::Whole(records) => {
                    // The GPU can't reach these any more, so only this copy needs changing.
                    for c in set.first_cluster..set.first_cluster + set.cluster_count {
                        let cluster = &mut records[c as usize];
                        cluster.vertex_offset -= vertex_base;
                        cluster.triangle_offset -= triangle_base;
                    }
                    for g in set.first_generated..set.first_generated + set.generated_count {
                        let parent = self.generated[g as usize];
                        records[parent as usize].counts |= CLUSTER_CHILDREN_MISSING;
                        self.dirty_clusters.push(parent);
                    }
                }
                ClusterCopy::Resident(resident) => {
                    for g in set.first_generated..set.first_generated + set.generated_count {
                        let parent = self.generated[g as usize];
                        resident.mark(
                            &self.sets,
                            &self.pages,
                            &self.slots,
                            parent,
                            true,
                            &mut self.dirty_clusters,
                        );
                    }
                }
            }
            write_node_flags(gpu, scene, set.leaf_node, NODE_LEAF | page);
        }
    }

    /// Writes all cluster records changed since the last call.
    pub fn flush(&mut self, gpu: &Gpu, scene: &GpuScene) {
        let chunk = self.regions.map_or(u32::MAX, |r| r.cluster_chunk);
        let runs = merged_runs(&mut self.dirty_clusters);
        match &self.clusters {
            ClusterCopy::Whole(records) => {
                for (first, last) in runs {
                    let run = &records[first as usize..=last as usize];
                    write_clusters(gpu, scene, chunk, first, run);
                }
            }
            ClusterCopy::Resident(resident) => {
                for (first, last) in runs {
                    // A run may pass from one slot's records into the next's.
                    let mut at = first;
                    while at <= last {
                        let slot = at / resident.stride;
                        let end = last.min((slot + 1) * resident.stride - 1);
                        let records = &resident.records[slot as usize];
                        let from = (at % resident.stride) as usize;
                        let to = ((end % resident.stride) as usize + 1).min(records.len());
                        if from < to {
                            write_clusters(gpu, scene, chunk, at, &records[from..to]);
                        }
                        at = end + 1;
                    }
                }
            }
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
                    clusters: None,
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

/// Which band a pool slot is in, and its place within that band.
fn locate_slot(regions: Option<&RegionPlan>, slot: u32) -> (usize, u32) {
    let Some(regions) = regions else {
        return (0, slot);
    };
    let mut first = 0;
    for (band, &slots) in regions.band_slots.iter().enumerate() {
        if slot < first + slots {
            return (band, slot - first);
        }
        first += slots;
    }
    panic!("slot {slot} is outside the pool");
}

/// Where a slot's vertices and triangles begin, in units of vertices and of triangle words.
/// Triangles start right after the page's own vertices. The band goes in the top two bits, which
/// is zero in a scene with one pool.
fn slot_bases(band: usize, local: u32, vertex_count: u32) -> (u32, u32) {
    let band = (band as u32) << crate::regions::BAND_SHIFT;
    (
        band | (local * PAGE_VERTICES),
        band | (local * PAGE_WORDS
            + vertex_count * (std::mem::size_of::<solstice_asset::PackedVertex>() as u32 / 4)),
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
        assert_eq!(slot_bases(0, 0, 10), (0, 40));
        assert_eq!(
            slot_bases(0, 2, 10),
            (2 * PAGE_VERTICES, 2 * PAGE_WORDS + 40)
        );
    }

    #[test]
    fn a_band_is_carried_above_the_place_in_its_buffer() {
        let (vertices, triangles) = slot_bases(2, 3, 10);
        use crate::regions::{BANDS, BAND_SHIFT};
        let low = (1 << BAND_SHIFT) - 1;
        assert_eq!(vertices >> BAND_SHIFT, 2);
        assert_eq!(vertices & low, 3 * PAGE_VERTICES);
        assert_eq!(triangles >> BAND_SHIFT, 2);
        assert_eq!(triangles & low, 3 * PAGE_WORDS + 40);

        // The last slot of the last band a binding can hold still fits beside its band, with
        // vertex numbers counted in words as the shaders do.
        let last = (crate::regions::binding_slots(&wgpu::Limits {
            max_storage_buffer_binding_size: 2_147_483_644,
            max_buffer_size: 1 << 31,
            ..Default::default()
        })) - 1;
        let (vertices, triangles) = slot_bases(BANDS - 1, last, PAGE_VERTICES);
        assert_eq!(vertices >> BAND_SHIFT, BANDS as u32 - 1);
        assert!(u64::from(vertices & low) * 4 + u64::from(PAGE_WORDS) <= 1 << BAND_SHIFT);
        assert_eq!(triangles >> BAND_SHIFT, BANDS as u32 - 1);
    }

    #[test]
    fn slots_are_numbered_through_the_bands() {
        let plan = RegionPlan {
            cluster_chunk: 1,
            band_slots: [2, 3, 4, 1, 0, 0, 0, 0],
        };
        assert_eq!(locate_slot(Some(&plan), 1), (0, 1));
        assert_eq!(locate_slot(Some(&plan), 2), (1, 0));
        assert_eq!(locate_slot(Some(&plan), 4), (1, 2));
        assert_eq!(locate_slot(Some(&plan), 8), (2, 3));
        assert_eq!(locate_slot(None, 8), (0, 8));
    }
}
