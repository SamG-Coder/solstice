//! Values that outlive a script call, and the Starlark types for vectors and entities.
//!
//! Starlark values live on a heap that is dropped when a step's script calls end. Anything a
//! script keeps, in a property or in `self.vars`, is stored as a [`ScriptValue`] instead: plain
//! data, `Send + Sync`, which the world can hold as a component.

use allocative::Allocative;
use glam::Vec3;
use solstice_world::Entity;
use starlark::any::ProvidesStaticType;
use starlark::collections::SmallMap;
use starlark::starlark_simple_value;
use starlark::values::dict::{AllocDict, DictRef};
use starlark::values::float::StarlarkFloat;
use starlark::values::list::{AllocList, ListRef};
use starlark::values::none::NoneType;
use starlark::values::{Heap, NoSerialize, StarlarkValue, Value, ValueLike};
use starlark_derive::starlark_value;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// A value a script keeps from one step to the next.
#[derive(Clone, Debug, PartialEq)]
pub enum ScriptValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
    Vec3(Vec3),
    Entity(Entity),
    List(Vec<ScriptValue>),
    Dict(BTreeMap<String, ScriptValue>),
}

impl ScriptValue {
    /// The value on a Starlark heap.
    pub fn to_starlark<'v>(&self, heap: Heap<'v>) -> Value<'v> {
        match self {
            Self::None => Value::new_none(),
            Self::Bool(b) => Value::new_bool(*b),
            Self::Int(i) => heap.alloc(*i),
            Self::Float(f) => heap.alloc(StarlarkFloat(*f)),
            Self::Str(s) => heap.alloc(&**s),
            Self::Vec3(v) => heap.alloc(Vec3Value(*v)),
            Self::Entity(e) => heap.alloc(EntityValue(*e)),
            Self::List(items) => heap.alloc(AllocList(items.iter().map(|i| i.to_starlark(heap)))),
            Self::Dict(map) => heap.alloc(AllocDict(
                map.iter()
                    .map(|(k, v)| (heap.alloc(k.as_str()), v.to_starlark(heap))),
            )),
        }
    }

    /// The value a Starlark value holds, if it is one a script can keep: none, a bool, an int
    /// that fits 64 bits, a float, a string, a vector, an entity, or a list, tuple or dict of
    /// those, with string keys.
    pub fn from_starlark(value: Value) -> Result<Self, String> {
        if value.is_none() {
            return Ok(Self::None);
        }
        if let Some(b) = value.unpack_bool() {
            return Ok(Self::Bool(b));
        }
        if let Some(i) = value.unpack_i32() {
            return Ok(Self::Int(i64::from(i)));
        }
        if let Some(f) = value.downcast_ref::<StarlarkFloat>() {
            return Ok(Self::Float(f.0));
        }
        if let Some(s) = value.unpack_str() {
            return Ok(Self::Str(Arc::from(s)));
        }
        if let Some(v) = value.downcast_ref::<Vec3Value>() {
            return Ok(Self::Vec3(v.0));
        }
        if let Some(e) = value.downcast_ref::<EntityValue>() {
            return Ok(Self::Entity(e.0));
        }
        if let Some(list) = ListRef::from_value(value) {
            return list
                .iter()
                .map(Self::from_starlark)
                .collect::<Result<_, _>>()
                .map(Self::List);
        }
        if let Some(tuple) = starlark::values::tuple::TupleRef::from_value(value) {
            return tuple
                .iter()
                .map(Self::from_starlark)
                .collect::<Result<_, _>>()
                .map(Self::List);
        }
        if let Some(dict) = DictRef::from_value(value) {
            let mut map = BTreeMap::new();
            for (key, item) in dict.iter() {
                let key = key.unpack_str().ok_or_else(|| {
                    format!("a dict kept by a script needs string keys, not {key}")
                })?;
                map.insert(key.to_owned(), Self::from_starlark(item)?);
            }
            return Ok(Self::Dict(map));
        }
        // A big int, or one that only fits as i64.
        if value.get_type() == "int" {
            if let Ok(i) = value.to_str().parse::<i64>() {
                return Ok(Self::Int(i));
            }
        }
        Err(format!(
            "a {} can't be kept from one step to the next",
            value.get_type()
        ))
    }
}

impl From<f64> for ScriptValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<i64> for ScriptValue {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<bool> for ScriptValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<&str> for ScriptValue {
    fn from(value: &str) -> Self {
        Self::Str(Arc::from(value))
    }
}

impl From<Vec3> for ScriptValue {
    fn from(value: Vec3) -> Self {
        Self::Vec3(value)
    }
}

impl From<Entity> for ScriptValue {
    fn from(value: Entity) -> Self {
        Self::Entity(value)
    }
}

/// A vector, as scripts see it: `v.x`, `v.y`, `v.z`, `+`, `-`, and `*` or `/` by a number.
#[derive(Clone, Copy, Debug, PartialEq, ProvidesStaticType, NoSerialize, Allocative)]
pub struct Vec3Value(#[allocative(skip)] pub Vec3);

starlark_simple_value!(Vec3Value);

impl fmt::Display for Vec3Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vec3({}, {}, {})", self.0.x, self.0.y, self.0.z)
    }
}

/// A number a vector can be scaled by.
fn number(value: Value) -> Option<f32> {
    value
        .unpack_i32()
        .map(|i| i as f32)
        .or_else(|| value.downcast_ref::<StarlarkFloat>().map(|f| f.0 as f32))
}

#[starlark_value(type = "vec3")]
impl<'v> StarlarkValue<'v> for Vec3Value {
    fn get_attr(&self, attribute: &str, heap: Heap<'v>) -> Option<Value<'v>> {
        let component = match attribute {
            "x" => self.0.x,
            "y" => self.0.y,
            "z" => self.0.z,
            _ => return None,
        };
        Some(heap.alloc(StarlarkFloat(f64::from(component))))
    }

    fn dir_attr(&self) -> Vec<String> {
        ["x", "y", "z"].map(String::from).to_vec()
    }

    fn equals(&self, other: Value<'v>) -> starlark::Result<bool> {
        Ok(other.downcast_ref::<Self>().is_some_and(|o| o.0 == self.0))
    }

    fn add(&self, rhs: Value<'v>, heap: Heap<'v>) -> Option<starlark::Result<Value<'v>>> {
        let rhs = rhs.downcast_ref::<Self>()?;
        Some(Ok(heap.alloc(Self(self.0 + rhs.0))))
    }

    fn sub(&self, rhs: Value<'v>, heap: Heap<'v>) -> starlark::Result<Value<'v>> {
        match rhs.downcast_ref::<Self>() {
            Some(rhs) => Ok(heap.alloc(Self(self.0 - rhs.0))),
            None => starlark::values::ValueError::unsupported_with(self, "-", rhs),
        }
    }

    fn mul(&self, rhs: Value<'v>, heap: Heap<'v>) -> Option<starlark::Result<Value<'v>>> {
        number(rhs).map(|n| Ok(heap.alloc(Self(self.0 * n))))
    }

    fn rmul(&self, lhs: Value<'v>, heap: Heap<'v>) -> Option<starlark::Result<Value<'v>>> {
        number(lhs).map(|n| Ok(heap.alloc(Self(self.0 * n))))
    }

    fn div(&self, rhs: Value<'v>, heap: Heap<'v>) -> starlark::Result<Value<'v>> {
        match number(rhs) {
            Some(n) => Ok(heap.alloc(Self(self.0 / n))),
            None => starlark::values::ValueError::unsupported_with(self, "/", rhs),
        }
    }

    fn minus(&self, heap: Heap<'v>) -> starlark::Result<Value<'v>> {
        Ok(heap.alloc(Self(-self.0)))
    }
}

/// An entity, as scripts see it: something to send events to, or to look up.
#[derive(Clone, Copy, Debug, PartialEq, ProvidesStaticType, NoSerialize, Allocative)]
pub struct EntityValue(#[allocative(skip)] pub Entity);

starlark_simple_value!(EntityValue);

impl fmt::Display for EntityValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "entity({})", self.0.id())
    }
}

#[starlark_value(type = "entity")]
impl<'v> StarlarkValue<'v> for EntityValue {
    fn equals(&self, other: Value<'v>) -> starlark::Result<bool> {
        Ok(other.downcast_ref::<Self>().is_some_and(|o| o.0 == self.0))
    }
}

/// Keeps `NoneType` and `SmallMap` in use for the types other modules build.
#[allow(dead_code)]
type Unused = (NoneType, SmallMap<String, ()>);

#[cfg(test)]
mod tests {
    use super::*;
    use starlark::environment::Module;

    #[test]
    fn values_come_back_as_they_went_in() {
        let mut dict = BTreeMap::new();
        dict.insert("speed".to_owned(), ScriptValue::Float(2.5));
        dict.insert(
            "path".to_owned(),
            ScriptValue::List(vec![ScriptValue::Vec3(Vec3::X)]),
        );
        let values = [
            ScriptValue::None,
            ScriptValue::Bool(true),
            ScriptValue::Int(-7),
            ScriptValue::Int(1 << 40),
            ScriptValue::Float(0.25),
            ScriptValue::from("gate"),
            ScriptValue::Vec3(Vec3::new(1.0, 2.0, 3.0)),
            ScriptValue::List(vec![ScriptValue::Int(1), ScriptValue::from("two")]),
            ScriptValue::Dict(dict),
        ];
        Module::with_temp_heap(|module| {
            for value in &values {
                let back = ScriptValue::from_starlark(value.to_starlark(module.heap()));
                assert_eq!(back.as_ref(), Ok(value));
            }
        });
    }
}
