//! Built meshes that don't depend on the scene, such as a species' trees and a meadow's patches
//! of grass, kept once built. A process that bakes many times, as the editor's bake in the browser
//! does each time a scene changes, builds each once: they are most of a wood's or a meadow's
//! cost. A native bake builds each once anyway, so for it nothing changes.

use solstice_asset::VirtualMesh;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

fn built() -> &'static Mutex<HashMap<String, Arc<VirtualMesh>>> {
    static BUILT: OnceLock<Mutex<HashMap<String, Arc<VirtualMesh>>>> = OnceLock::new();
    BUILT.get_or_init(Default::default)
}

/// The mesh for `key`: the one built before, or else `make`'s, kept for next time. A key names
/// all that the mesh is made from, so the same key always gives the same mesh.
pub fn mesh(key: &str, make: impl FnOnce() -> VirtualMesh) -> VirtualMesh {
    let found = built()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .cloned();
    let mesh = found.unwrap_or_else(|| {
        let made = Arc::new(make());
        built()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.to_owned(), made.clone());
        made
    });
    (*mesh).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mesh_is_built_once_for_its_key() {
        let mut builds = 0;
        let mut make = || {
            builds += 1;
            VirtualMesh {
                root_node: 7,
                ..Default::default()
            }
        };
        assert_eq!(mesh("test-built-once", &mut make).root_node, 7);
        assert_eq!(mesh("test-built-once", &mut make).root_node, 7);
        assert_eq!(builds, 1);
    }
}
