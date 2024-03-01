extern crate queues;

use std::ops::DerefMut;
use std::rc::Rc;
use crate::array::{DepTree, Object};
use std::sync::{Arc, Mutex};
// use crate::array::{DepTree, Object};

pub struct TaskQueue {
    dependency_graph: Vec<Rc<DepTree>>,
}

impl TaskQueue {
    pub(crate) fn init() -> Self {
        Self {
            dependency_graph: Vec::new(),
        }
    }

    pub fn push_object(&mut self, item: Arc<Mutex<Object>>) {
        let node = Arc::clone(&item);
        let dep = Rc::new(DepTree::init(item));
        node.lock().unwrap().set_dependency(&dep);
        self.dependency_graph.push(dep)
    }
    
}
