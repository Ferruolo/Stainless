use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::classes::{ItemLoc, Operation};

pub(crate) struct Object {
    loc: ItemLoc,
    name: u64,
    shape: Vec<u64>,
    dependency_tree: Option<Arc<Mutex<DepTree>>>,
    left: Option<Arc<Mutex<Object>>>,
    right: Option<Arc<Mutex<Object>>>,
    track_deps: bool,
    forge_op: Operation,
}

impl Object {
    pub(crate) fn init(
        name: u64,
        shape: &Vec<u64>,
        track_deps: bool,
        forge_op: Operation,
        left: Option<Arc<Mutex<Object>>>,
        right: Option<Arc<Mutex<Object>>>,
    ) -> Self {
        Self {
            loc: ItemLoc::CPU,
            name,
            shape: shape.clone(),
            dependency_tree: None,
            left,
            right,
            track_deps,
            forge_op,
        }
    }

    pub fn set_dependency(&mut self, dep: &Arc<Mutex<DepTree>>) {
        self.dependency_tree = Some(Arc::clone(dep));
    }

    pub fn remove_dependency(&mut self) {
        self.dependency_tree = None;
    }

    pub fn get_shape(&self) -> &Vec<u64> {
        &self.shape
    }

    pub fn get_op(&self) -> Operation {
        self.forge_op
    }

    pub fn get_left(&self) -> &Option<Arc<Mutex<Object>>> {
        return &self.left;
    }
    pub fn get_right(&self) -> &Option<Arc<Mutex<Object>>> {
        return &self.right;
    }

    pub fn get_dep(&self) -> Option<Arc<Mutex<DepTree>>> {
        if let Some(d) = &self.dependency_tree {
            return Some(Arc::clone(d));
        } else {
            return None;
        }
    }

    pub fn get_name(&self) -> u64 {
        return self.name;
    }

    pub fn get_loc(&self) -> ItemLoc {
        return self.loc;
    }
}

pub struct DepTree {
    node: Arc<Mutex<Object>>,
    forge_op: Operation,
    children: Vec<Arc<Mutex<DepTree>>>,
    height: usize,
    shape: Vec<u64>,
    num_parents: u8,
    parents: HashMap<u64, Arc<Mutex<Object>>>,
    loc: ItemLoc,
}

impl DepTree {
    pub(crate) fn init(refers_to: Arc<Mutex<Object>>) -> Self {
        let mut children = Vec::new();
        {
            let unwrapped = &refers_to.lock().unwrap();
            let unwrapped_children = [unwrapped.get_right(), unwrapped.get_left()];
            for c in unwrapped_children {
                if let Some(child) = c {
                    if let Some(dep) = child.lock().unwrap().get_dep() {
                        children.push(Arc::clone(&dep))
                    }
                }
            }
        }

        let max_height = children
            .iter()
            .map(|c| c.lock().unwrap().height + 1)
            .max()
            .unwrap_or(0);
        let shape = refers_to.lock().unwrap().shape.clone();
        let loc = refers_to.lock().unwrap().get_loc();
        Self {
            node: Arc::clone(&refers_to),
            forge_op: refers_to.lock().unwrap().get_op(),
            children,
            height: max_height,
            shape: shape,
            num_parents: 0,
            parents: HashMap::new(),
            loc,
        }
    }

    pub fn merge(&self, other: &Arc<Mutex<DepTree>>) -> Option<Arc<Mutex<Object>>> {
        let len = other.lock().unwrap().children.len();
        let o_forge_op = other.lock().unwrap().forge_op;

        if len == 0 || len != self.children.len() || o_forge_op != self.forge_op {
            return None;
        }

        let other_children = other.lock().unwrap().children.clone();
        for (s, o) in self.children.iter().zip(other_children.iter()) {
            let s_name = s.lock().unwrap().node.lock().unwrap().get_name();
            let o_name = o.lock().unwrap().node.lock().unwrap().get_name();
            if s_name != o_name {
                return None;
            }
        }

        return Some(Arc::clone(&self.node));
    }

    pub fn get_num_children(&self) -> usize {
        return self.children.len();
    }

    pub fn get_children(&self) -> Vec<Arc<Mutex<DepTree>>> {
        return self.children.clone();
    }

    pub fn get_height(&self) -> usize {
        return self.height;
    }

    pub fn get_name(&self) -> u64 {
        return self.node.lock().unwrap().get_name();
    }

    pub fn get_forge_op(&self) -> Operation {
        return self.forge_op;
    }

    pub fn get_num_use(&self) -> u8 {
        return self.num_parents;
    }

    pub fn get_shape(&self) -> &Vec<u64> {
        return &self.shape;
    }

    pub fn get_node(&self) -> Arc<Mutex<Object>> {
        return Arc::clone(&self.node);
    }

    pub fn increment_num_parents(&mut self) {
        self.num_parents += 1
    }

    pub fn get_num_parents(&self) -> u8 {
        return self.num_parents;
    }

    pub fn add_parent(&mut self, parent: &Arc<Mutex<Object>>) {
        let name = parent.lock().unwrap().get_name();
        self.parents.insert(name, Arc::clone(parent));
    }

    pub fn erase(&self) {
        // self.node.lock().unwrap().remove_dependency();
        // for (_, parent) in self.parents {
        //     parent.lock().unwrap().remove_child(self.get_name());
        // }
        // for child in self.children {
        //     child.remove_parent(self.get_name());
        // }
        todo!();
    }

    pub fn get_loc(&self) -> ItemLoc {
        return self.loc;
    }
}
