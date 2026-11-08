//! Gameplay scripts in Starlark, a Python-like language (docs/gameplay.md, step 3).
//!
//! * [`ScriptHost`]: loads scripts, reloads them while the game runs, and runs their handlers.
//! * [`Script`]: the component that gives an actor a script, its properties and its own values.
//! * [`ScriptEvent`]: events between scripts, and from Rust to scripts.
//! * [`ScriptValue`]: what a script keeps from one step to the next.

pub mod actor;
mod api;
pub mod host;
pub mod value;

pub use host::{
    api_reference, Limits, Name, Script, ScriptError, ScriptEvent, ScriptHost, ScriptLog,
};
pub use value::ScriptValue;
