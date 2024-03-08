extern crate queues;

use std::cmp::{max};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::array::{DepTree, Object};
use crate::classes::ComputationGraph::{NoPattern, Op, M};
use crate::classes::Operation::*;
use crate::classes::{ComputationGraph, ItemLoc, Operation, ThreadCommands};
use crate::classes::ThreadCommands::{Calculation, NullType};

// TODO: REWRITE TRIGGERED
// TODO: 
// 1. Convert to Fibonacci Tree / Priority Queue interface, where
//    items are sorted using 


pub struct TaskQueue {
    dependency_graph: VecDeque<Arc<Mutex<VecDeque<Arc<Mutex<DepTree>>>>>>,
    height_boost: usize,
    live_calculations: Arc<Mutex<u64>>
}

impl TaskQueue {
    pub(crate) fn init() -> Self {
        Self {
            dependency_graph: VecDeque::new(),
            height_boost: 0,
            live_calculations: Arc::new(Mutex::new(0))
        }
    }

    pub fn get_next(&mut self) -> ThreadCommands {
        let mut bottom_line = match self.dependency_graph.front() {
            Some(line) => Arc::clone(line),
            None => return NullType,
        };

        let calc_item = match bottom_line.lock().unwrap().pop_front() {
            Some(next) => next,
            None => return NullType,
        };

        if bottom_line.lock().unwrap().is_empty() {
            self.dependency_graph.pop_front();
        }

        let node = calc_item.lock().unwrap().get_node(); // Assuming get_node is a method of DepTree

        let live_calculations = Arc::clone(&self.live_calculations);
        *live_calculations.lock().unwrap() -= 1;
        
        Calculation(Arc::clone(&node))
    }
    pub fn push_object(&mut self, item: Arc<Mutex<Object>>) -> Arc<Mutex<Object>> {
        let dep = Arc::new(Mutex::new(DepTree::init(Arc::clone(&item))));
        // dep = self.optimize(dep);

        let height = dep.lock().unwrap().get_height() - self.height_boost;
        let set_height = max(height, 2);
        if self.dependency_graph.len() <= set_height {
            for _ in 0..set_height * 2 {
                self.dependency_graph.push_back(Arc::new(Mutex::new(VecDeque::new())));
            }
        }
        for other in self.dependency_graph[height].lock().unwrap().iter() {
            if let Some(merged) = other.lock().unwrap().merge(&dep) {
                return merged;
            }
        }

        {
            item.lock().unwrap().set_dependency(&dep);
        }
        let lvl = Arc::clone(&self.dependency_graph[height]);
        lvl.lock().unwrap().push_front(dep);

        let live_calculations = Arc::clone(&self.live_calculations);
        *live_calculations.lock().unwrap() += 1;

        return item;
    }


    pub fn print_items(&self) {
        for lvl in &self.dependency_graph {
            for item in lvl.lock().unwrap().iter() {
                print!("-{}", item.lock().unwrap().get_name());
            }
            println!();
        }
    }
    pub fn still_live(&self) -> bool {
        return *self.live_calculations.lock().unwrap() > 0
    }
    
}
