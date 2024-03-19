/*
* Custom Implementation of Binary Heap
* Note: This can (and should) be redone to be totally functional
*/
use std::collections::HashMap;

pub(crate) trait HeapInterface {
    fn get_key(&self) -> u64;
    fn is_less_than(&self, other: &Self) -> bool; // caller is less than callee
    fn no_dependencies_remaining(&self) -> bool;
    fn decrease(&mut self);

    fn decrease_num_children(&mut self);
}
pub struct BinaryPQ<T: HeapInterface> {
    data: Vec<T>,
    lookup: HashMap<u64, usize>,
}

impl<T: HeapInterface> BinaryPQ<T> {
    pub(crate) fn new() -> Self {
        BinaryPQ {
            data: Vec::with_capacity(20),
            lookup: HashMap::with_capacity(20),
        }
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    pub fn insert(&mut self, item: T) {
        self.data.push(item);
        self.bubble_up(self.len() - 1);
    }

    fn remove_min(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let min = self.data.swap_remove(0);
        if !self.is_empty() {
            self.bubble_down(0);
        }

        Some(min)
    }

    fn bubble_up(&mut self, mut index: usize) {
        let mut parent = (index - 1) / 2;
        while index > 0 && self.data[index].is_less_than(&self.data[parent]) {
            self.data.swap(index, parent);
            index = parent;
            parent = (index - 1) / 2;
        }
    }

    fn bubble_down(&mut self, mut index: usize) {
        let len = self.len();
        let mut min_child = index;

        loop {
            let left = 2 * index + 1;
            let right = left + 1;

            if left < len && self.data[left].is_less_than(&self.data[min_child]) {
                min_child = left;
            }

            if right < len && self.data[right].is_less_than(&self.data[min_child]) {
                min_child = right;
            }

            if min_child == index {
                break;
            }

            self.data.swap(index, min_child);
            index = min_child;
        }
    }

    fn get_min(&self) -> Option<&T> {
        self.data.first()
    }

    fn decrease_key(&mut self, index: usize, new_value: T) {
        self.data[index] = new_value;
        self.bubble_up(index);
    }

    fn decrease_num_children(&self, name: u64) {
        let idx = match self.lookup.get(&name) {
            None => return,
            Some(idx) => idx,
        };
        let node = &mut self.data[*idx];
        node.decrease_num_children();
    }

    fn increment_num_dependencies(&self, child: u64, parent: u64) {
        
    }

    fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }

        let len = self.len();
        self.data.swap(index, len - 1);
        let removed = self.data.pop().unwrap();

        if index < len - 1 {
            self.bubble_down(index);
        }

        Some(removed)
    }
    fn is_heap_valid(&self) -> bool {
        self.is_heap_valid_helper(0)
    }

    fn is_heap_valid_helper(&self, index: usize) -> bool {
        let left = 2 * index + 1;
        let right = left + 1;

        if left < self.len() && self.data[left].is_less_than(&self.data[index]) {
            return false;
        }

        if right < self.len() && self.data[right].is_less_than(&self.data[index]) {
            return false;
        }

        if left < self.len() && !self.is_heap_valid_helper(left) {
            return false;
        }

        if right < self.len() && !self.is_heap_valid_helper(right) {
            return false;
        }

        true
    }
}
