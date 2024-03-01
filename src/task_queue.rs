extern crate queues;

use std::rc::Rc;
use crate::array::{DepTree, Object};
use std::sync::{Arc, Mutex};


pub struct TaskQueue {
    dependency_graph: Vec<
        Vec<
            Rc<DepTree>
        >
    >,
}

impl TaskQueue {
    pub(crate) fn init() -> Self {
        Self {
            dependency_graph: vec![Vec::new()],
        }
    }

    pub fn push_object(&mut self, item: Arc<Mutex<Object>>) -> Arc<Mutex<Object>> {
        let node = Arc::clone(&item);
        let dep = Rc::new(DepTree::init(node));

        if self.dependency_graph.len() <= dep.get_height() {
            for _ in 0..self.dependency_graph.len() * 2 {
                self.dependency_graph.push(Vec::new());
            }
        }

        let h = &self.dependency_graph[dep.get_height()];

        for other in h {
            if let Some(merged) = other.merge(&dep) {
                return merged;
            }
        }

        {
            item.lock().unwrap().set_dependency(&dep);
        }
        //Amortizes to O(1)
        item.lock().unwrap().set_dependency(&dep);
        self.dependency_graph[dep.get_height()].push(dep);
        return item;
    }
    pub fn print_items(&self) {
        for lvl in &self.dependency_graph {
            for item in lvl {
                print!("-{}", item.get_name());
            }
            println!();
        }
    }

}
