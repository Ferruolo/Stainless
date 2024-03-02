extern crate queues;

use crate::array::{DepTree, Object};
use crate::classes::Operations;
use crate::classes::Operations::MatMul;
use crate::task_queue::ComputationGraph::*;

use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};
use libc::gai_strerror;

pub struct TaskQueue {
    dependency_graph: Vec<Vec<Rc<DepTree>>>,
    optim_patterns: HashMap<Operations, Vec<ComputationGraph>>
}

pub enum ComputationGraph {
    Op(Box<ComputationGraph>, Box<ComputationGraph>),
    M(usize),
    None,
}

impl TaskQueue {
    pub(crate) fn init() -> Self {
        Self {
            dependency_graph: vec![Vec::new()],
            optim_patterns: HashMap::from([(
                MatMul,
                vec![
                    Op(
                        Box::new(Op(Box::new(M(0)), Box::new(M(1)))),
                        Box::new(Op(Box::new(M(2)), Box::new(M(3)))),
                    ),
                    Op(
                        Box::new(Op(
                            Box::new(Op(Box::new(M(0)), Box::new(M(1)))),
                            Box::new(M(2)),
                        )),
                        Box::new(M(3)),
                    ),
                    Op(
                        Box::new(Op(
                            Box::new(M(0)),
                            Box::new(Op(Box::new(M(1)), Box::new(M(2)))),
                        )),
                        Box::new(M(3)),
                    ),
                    Op(
                        Box::new(M(1)),
                        Box::new(Op(
                            Box::new(Op(Box::new(M(1)), Box::new(M(2)))),
                            Box::new(M(3))
                        ))
                    ),
                    Op(
                        Box::new(M(1)),
                        Box::new(Op(
                            Box::new(M(2)),
                            Box::new(Op(Box::new(M(3)), Box::new(M(4)))),
                        ))
                    )
                ],
            )]),
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

    fn search_fxn(&self, graph: &ComputationGraph, children: &Vec<Rc<DepTree>>) -> (u8, Vec<u8>) {
        match graph {
            Op(a, b) => {
                let l = self.search_fxn(&*a, children);
                let r = self.search_fxn(&*b, children);
                let new_shape = vec![l.1[0], r.1[0]];
                (l.0 + r.0 + new_shape[0] * l.1[1] * new_shape[1], new_shape) // Needs to be extended to work with ALL possible optimizations
            }
            M(m) => {
                (children[m].get_shape()[0] * children[m].get_shape()[1],
                 children[m].get_shape().clone()
                )
            }
            None => {(0, vec![])}
        }
    }

    pub fn optimize(&self, node: Rc<DepTree>) { //TODO: Rewrite MEEEEEE I'm SLOWWWW (make me functional)
        let forge_op = node.get_forge_op();
        if node.get_height() < 2 || node.get_num_children() < 2 {
            return;
        }
        let children = node.get_children();
        // Check if we even can optimize this calculation
        // If we don't have matching actions, we can't optimize them
        {
            let ops: Vec<Operations> = children.iter().map(|c| c.get_forge_op()).collect();
            if !ops.iter().all(|&op| op == ops[0]) {
                return;
            }
        }

        let mut sub_children = Vec::new();
        for child in children {
            sub_children.push(child);
        }

        let patterns = self.optim_patterns.get(&forge_op).unwrap();
        let mut best_op ;
        let mut best_runtime = 0;


        for i in 0..patterns.len() {
            let (runtime, _) = self.search_fxn(&patterns[i], &sub_children);
            if runtime < best_runtime {
                best_op = &patterns[i];
                best_runtime = runtime
            }
        }



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
