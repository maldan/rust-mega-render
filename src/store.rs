use std::marker::PhantomData;

#[derive(PartialEq, Eq, Hash)]
pub struct Handle<T> {
    pub index: u32,
    pub generation: u32,
    _m: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> Handle<T> {
    pub fn key(self) -> (u32, u32) {
        (self.index, self.generation)
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

pub struct Store<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
}

impl<T> Default for Store<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }
}

impl<T> Store<T> {
    pub fn insert(&mut self, value: T) -> Handle<T> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            return Handle {
                index,
                generation: slot.generation,
                _m: PhantomData,
            };
        }
        let index = self.slots.len() as u32;
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        Handle {
            index,
            generation: 0,
            _m: PhantomData,
        }
    }

    pub fn get(&self, h: Handle<T>) -> Option<&T> {
        self.slots
            .get(h.index as usize)
            .filter(|s| s.generation == h.generation)
            .and_then(|s| s.value.as_ref())
    }

    pub fn get_mut(&mut self, h: Handle<T>) -> Option<&mut T> {
        self.slots
            .get_mut(h.index as usize)
            .filter(|s| s.generation == h.generation)
            .and_then(|s| s.value.as_mut())
    }

    pub fn remove(&mut self, h: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(h.index as usize)?;
        if slot.generation != h.generation {
            return None;
        }
        let v = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(h.index);
        Some(v)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.value.as_ref().map(|v| {
                (
                    Handle {
                        index: i as u32,
                        generation: s.generation,
                        _m: PhantomData,
                    },
                    v,
                )
            })
        })
    }
}
