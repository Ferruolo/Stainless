use std::ptr;
use crate::classes::{ItemLoc, Operations};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub(crate) struct Object {

    loc: ItemLoc,
    name: u64,
    shape: Vec<u8>,
    dependency_tree: Option<Rc<DepTree>>,
    left: Option<Arc<Mutex<Object>>>,
    right: Option<Arc<Mutex<Object>>>,
    track_deps: bool,
    forge_op: Operations,
}

impl Object {

    pub(crate) fn init(
        name: u64,
        shape: &Vec<u8>,
        track_deps: bool,
        forge_op: Operations,
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

    pub fn set_dependency(&mut self, dep: &Rc<DepTree>) {
        self.dependency_tree = Some(Rc::clone(dep));
    }

    pub fn get_shape(&self) -> &Vec<u8> {
        &self.shape
    }

    pub fn get_op(&self) -> Operations {
        self.forge_op
    }

    pub fn get_left(&self) -> &Option<Arc<Mutex<Object>>> {
        return &self.left;
    }
    pub fn get_right(&self) -> &Option<Arc<Mutex<Object>>> {
        return &self.right;
    }

    pub fn get_dep(&self) -> Option<Rc<DepTree>> {
        if let Some(d) = &self.dependency_tree {
            return Some(Rc::clone(d));
        } else {
            return None;
        }
    }

    pub fn get_name(&self) -> u64 {
        return self.name;
    }
}


pub struct DepTree{
    node: Arc<Mutex<Object>>,
    forge_op: Operations,
    children: Vec<Rc<DepTree>>,
    height: usize
}

impl  DepTree {
    pub(crate) fn init(refers_to: Arc<Mutex<Object>>) -> Self {
        let mut children = Vec::new();
        {
            let unwrapped = &refers_to.lock().unwrap();
            let unwrapped_children = [unwrapped.get_right(), unwrapped.get_left()];
            for c in unwrapped_children {
                if let Some(child) = c {
                    if let Some(dep) = child.lock().unwrap().get_dep() {
                        children.push(Rc::clone(&dep))
                    }
                }
            }
        }

        let max_height = children.iter().map(|c| c.height + 1).max();

        Self {
            node: Arc::clone(&refers_to),
            forge_op: refers_to.lock().unwrap().get_op(),
            children,
            height: max_height.unwrap_or(0)
        }
    }

    pub fn merge(&self, other: &Rc<DepTree>) -> Option<Arc<Mutex<Object>>> {
        if other.children.len() == 0 
            || other.children.len() != self.children.len() 
            || other.forge_op != self.forge_op {
            return None;
        }

        for (s, o) in self.children.iter().zip(other.children.iter()) {
            let s_name = s.node.lock().unwrap().get_name();
            let o_name = o.node.lock().unwrap().get_name();
            if s_name != o_name {
                return None
            }
        }
        
        return Some(Arc::clone(&self.node))
    }


    pub fn get_height(&self) -> usize {
        return self.height;
    }

    pub fn get_name(&self) -> u64 {
        return self.node.lock().unwrap().get_name();
    }
}


