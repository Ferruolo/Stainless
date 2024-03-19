use std::cell::RefCell;
use std::cmp::max;
use std::collections::HashMap;
use std::rc::Rc;

use crate::classes::ItemLoc;
use crate::object::{Object, ObjectInterface};
use crate::PriorityHeap::HeapInterface;

/*
* Dep Tree Functionality:
*   - Tracks if any dependencies still need calculation (won't get sent to calculation until they are)
*   - Does NOT deadbolt
*   - Releases parent tress upon calculation
*/

pub trait DepTreeInterface {
    fn init(obj: &Object, lookup: &HashMap<u64, DepTree>) -> Self;
    fn get_height(&self) -> usize;
    fn increment_num_dependencies(&mut self);
    fn get_name(&self) -> u64;
    fn get_num_dependencies(&self) -> usize;
    fn set_parent(&mut self, parent: u64);
    fn decrease_num_children(&mut self);

    fn get_parents(&self) -> Vec<u64>;

    fn get_node(&self) -> Object;
}

struct DepTreeInner {
    node: Object,             // The Object that this DepTree Object references
    location: ItemLoc,        // The location of the object in memory
    height: usize,            // Height (number of items that need to be calculated before this one
    num_dependencies: usize,  // Number of objects that depend on this one
    name: u64,                // Node Name
    parents: Vec<u64>,        // Names of all parents for this dep tree
    num_live_children: usize, // Number of live objects that this one depends on
}

impl DepTreeInterface for DepTreeInner {
    fn init(obj: &Object, lookup: &HashMap<u64, DepTree>) -> Self {
        let name = obj.get_name();

        let mut height: usize = 0;
        let mut num_live_children: usize = 0;
        let child_handler = |child: Option<Object>, height, num_live_children| match child {
            None => (height, num_live_children),
            Some(c) => {
                let name = c.get_name();
                match lookup.get(&name) {
                    None => (height, num_live_children),
                    Some(c) => (max(height, c.get_height()), num_live_children + 1),
                }
            }
        };
        (height, num_live_children) = child_handler(obj.get_left(), height, num_live_children);
        (height, num_live_children) = child_handler(obj.get_right(), height, num_live_children);

        return Self {
            node: obj.clone(),
            location: obj.get_loc(),
            height,
            num_dependencies: 0,
            name,
            parents: Vec::new(),
            num_live_children,
        };
    }

    fn get_height(&self) -> usize {
        return self.height;
    }

    fn increment_num_dependencies(&mut self) {
        self.num_dependencies += 1;
    }

    fn get_name(&self) -> u64 {
        return self.name;
    }

    fn get_num_dependencies(&self) -> usize {
        return self.num_dependencies;
    }

    fn set_parent(&mut self, parent: u64) {
        self.parents.push(parent);
        self.increment_num_dependencies();
    }

    fn decrease_num_children(&mut self) {
        self.num_live_children -= 1
    }

    fn get_parents(&self) -> Vec<u64> {
        return self.parents.clone();
    }

    fn get_node(&self) -> Object {
        return self.node.clone();
    }
}

#[derive(Clone)]
pub struct DepTree {
    inner: Rc<RefCell<DepTreeInner>>,
}

impl DepTreeInterface for DepTree {
    fn init(obj: &Object, lookup: &HashMap<u64, DepTree>) -> Self {
        return Self {
            inner: Rc::new(RefCell::new(DepTreeInner::init(obj, lookup))),
        };
    }

    fn get_height(&self) -> usize {
        return self.inner.borrow().get_height();
    }

    fn increment_num_dependencies(&mut self) {
        self.inner.borrow_mut().increment_num_dependencies();
    }

    fn get_name(&self) -> u64 {
        return self.inner.borrow().get_name();
    }

    fn get_num_dependencies(&self) -> usize {
        return self.inner.borrow().get_num_dependencies();
    }

    fn set_parent(&mut self, parent: u64) {
        self.inner.borrow_mut().set_parent(parent);
        self.increment_num_dependencies();
    }

    fn decrease_num_children(&mut self) {
        self.inner.borrow_mut().decrease_num_children();
    }

    fn get_parents(&self) -> Vec<u64> {
        return self.inner.borrow().get_parents();
    }

    fn get_node(&self) -> Object {
        return self.inner.borrow().get_node();
    }
}

impl HeapInterface for DepTree {
    fn get_key(&self) -> u64 {
        todo!()
    }

    fn is_less_than(&self, other: &Self) -> bool {
        let num_live_children = |x: &Self| x.inner.borrow().num_live_children;

        let (l_live_children, r_live_children) =
            (num_live_children(self), num_live_children(other));
        let (l_num_dependencies, r_num_dependencies) =
            (self.get_num_dependencies(), other.get_num_dependencies());
        let (l_height, r_height) = (self.get_height(), other.get_height());
        let (l_name, r_name) = (self.get_name(), other.get_name());

        return if l_live_children != r_live_children {
            l_live_children < r_live_children
        } else if l_num_dependencies != r_num_dependencies {
            l_num_dependencies > r_num_dependencies
        } else if l_height != r_height {
            l_height < r_height
        } else {
            l_name < r_name
        }
    }

    fn no_dependencies_remaining(&self) -> bool {
        self.inner.borrow().num_live_children == 0
    }

    fn decrease(&mut self) {
        todo!()
    }

    fn decrease_num_children_heap(&mut self) {
        self.inner.borrow_mut().decrease_num_children()
    }

    fn set_parent_heap(&mut self, parent: u64) {
        self.set_parent(parent);
    }
}
