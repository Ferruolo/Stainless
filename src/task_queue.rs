extern crate queues;

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::array::{DepTree, Object};
use crate::classes::{ItemLoc, Operation};
use crate::classes::Operation::*;
use crate::task_queue::ComputationGraph::*;

pub struct TaskQueue {
    dependency_graph: Vec<Vec<Rc<DepTree>>>,
    optim_patterns: HashMap<Operation, Vec<ComputationGraph>>,
}

pub enum ComputationGraph {
    Op(Box<ComputationGraph>, Box<ComputationGraph>),
    M(usize),
    NoPattern,
}

impl TaskQueue {
    pub(crate) fn init() -> Self {
        Self {
            dependency_graph: vec![Vec::new()],
            // I feel like theres a better way to look at all optim patterns
            // without putting them explicit, but can't think of it
            // TODO: Optimize optim pattern finding
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
                            Box::new(M(3)),
                        )),
                    ),
                    Op(
                        Box::new(M(1)),
                        Box::new(Op(
                            Box::new(M(2)),
                            Box::new(Op(Box::new(M(3)), Box::new(M(4)))),
                        )),
                    ),
                ],
            )]),
        }
    }

    pub fn push_object(&mut self, item: Arc<Mutex<Object>>) -> Arc<Mutex<Object>> {
        let node = Arc::clone(&item);
        let dep = Rc::new(DepTree::init(node));

        //Amortizes to O(1)
        if self.dependency_graph.len() <= dep.get_height() {
            for _ in 0..self.dependency_graph.len() * 2 {
                self.dependency_graph.push(Vec::new());
            }
        }

        for other in &self.dependency_graph[dep.get_height()] {
            if let Some(merged) = other.merge(&dep) {
                return merged;
            }
        }

        {
            item.lock().unwrap().set_dependency(&dep);
        }

        // item.lock().unwrap().set_dependency(&dep);
        self.dependency_graph[dep.get_height()].push(dep);
        return item;
    }

    // TODO: Don't Forget About the Parents (Is creating a new item worth it here?)!!!!!
    fn pattern_runtime(&self, graph: &ComputationGraph, children: &Vec<Rc<DepTree>>) -> (u8, Vec<u8>) {
        match graph {
            Op(a, b) => {
                let l = self.pattern_runtime(&*a, children);
                let r = self.pattern_runtime(&*b, children);
                let new_shape = vec![l.1[0], r.1[0]];
                (l.0 + r.0 + new_shape[0] * l.1[1] * new_shape[1], new_shape)
                // Needs to be extended to work with ALL possible optimizations
            }
            M(m) => (
                children[*m].get_shape()[0] * children[*m].get_shape()[1],
                children[*m].get_shape().clone(),
            ),
            NoPattern => (0, vec![]),
        }
    }


    fn pattern_equal(&self, left: &ComputationGraph, right: &ComputationGraph) -> bool{
        match (left, right) {
            (Op(l1, l2), Op(r1, r2)) => {
                self.pattern_equal(l1, r1) && self.pattern_equal(l2, r2)
            }
            (M(l), M(r)) => {l == r}
            (_, _) => {false }
        }
    }


    //Could we pass a vector ref down here and save some energy? I feel like that might work well
    pub fn get_pattern_binop(
        &self,
        node: &Rc<DepTree>,
        remaining: usize,
        item_num: usize,
    ) -> (ComputationGraph, Vec<Rc<DepTree>>) {
        // Base Case
        if remaining == 0 {
            return (M(item_num), vec![Rc::clone(node)]);
        }

        let children = node.get_children();
        match (children[0].get_forge_op(), children[1].get_forge_op()) {
            (MatMul, MatMul) => {
                let l = self.get_pattern_binop(
                    &children[0], remaining - 2, item_num
                );
                let mut r = self.get_pattern_binop(
                    &children[1], remaining - 2, item_num + 1
                );
                // Combine Children
                let new_op = Op(Box::new(l.0), Box::new(r.0));
                let mut descendents = l.1.clone();
                descendents.append(&mut r.1);
                return (new_op, descendents);
            }

            // I feel like these could be combined
            (MatMul, _) => {
                // New Op is on Left hand side!
                let l = self.get_pattern_binop(
                    &children[0], remaining - 1, item_num
                );
                let new_op = Op(Box::new(l.0), Box::new(M(item_num + 1)));
                let mut descendents = l.1.clone();
                descendents.push(Rc::clone(&children[1]));
                return (new_op, descendents);
            }

            (_, MatMul) => {
                // New Op is on Right hand side!
                let r = self.get_pattern_binop(
                    &children[0], remaining - 1, item_num + 1
                );
                let new_op = Op(Box::new(r.0), Box::new(M(item_num)));
                let mut descendents = r.1.clone();
                descendents.push(Rc::clone(&children[1]));
                return (new_op, descendents);
            }

            (_, _) => {
                return (M(0), vec![]);
            }
        }
    }

    fn pattern_finder(&self, patterns: &Vec<ComputationGraph>, children: &Vec<Rc<DepTree>>, index: usize) ->(u8, Vec<u8>, usize) {
        {
            if (index < patterns.len()) {
                let this = self.pattern_runtime(&patterns[index], children);
                let other = self.pattern_finder(patterns, children, index + 1);
                if (this.0 > other.0) {
                    return (this.0, this.1, index);
                } else {
                    return other;
                }
            } else {
                // Assume that this will never be returned as final, this may kick me in the ass
                return (u8::MAX, vec![], 0);
            }
        }
    }


    fn get_best_pattern(&self, children: &Vec<Rc<DepTree>>, forge_op: Operation) -> (u8, Vec<u8>, usize)  {

        if let Some(patterns) = self.optim_patterns.get(&forge_op) {
            return self.pattern_finder(patterns, children, 0);
        } else {
            return (0, vec![], 0);
        }
    }

    fn delete_old_tree(&self, graph: &ComputationGraph, tree: Rc<DepTree>, parent_name: Option<u64>) {
        // if let Some(parent) = parent_name {
        //     tree.remove_parent(parent);
        // }
        let name = tree.get_name();

        match graph {
            Op(l, r) => {
                self.delete_old_tree(&*l, Rc::clone(&tree.get_children()[0]), Some(name))
            }
            M(_) => {return;}
            NoPattern => {return;}
        }
        if (tree.get_num_parents() == 0) {
            tree.erase();
        }
    }

    fn build_new_tree(&self, graph:  &ComputationGraph, children: Vec<Rc<DepTree>>, forge_op: Operation, name: u64, loc: ItemLoc) -> Rc<DepTree> {
        match graph {
            Op(l, r) => {
                let left_tree = self.build_new_tree(l, children.clone(), forge_op, name + 1, loc);
                let right_tree = self.build_new_tree(r, children.clone(), forge_op, name + 1, loc);
                let new_shape = vec![left_tree.get_shape()[0], right_tree.get_shape()[1]];
                return  Rc::new(DepTree::init(
                    Arc::new(Mutex::new(Object::init(
                        name,
                        &new_shape,
                        false,
                        forge_op,
                        Some(left_tree.get_node()),
                        Some(right_tree.get_node())
                    )))))
                }
            
            M(idx) => {
                return Rc::clone(&children[0]);
            }
            NoPattern => {
                panic!("Built tree with no pattern")
            }
        }
    }


    pub fn optimize(&self, node: Rc<DepTree>) {
        if node.get_height() < 2 || node.get_num_children() < 2 {
            return;
        }

        let forge_op = node.get_forge_op();

        match forge_op {
            Operation::Add => {
                // Addition Order doesnt matter
                // However MatMul plus Addition can be simplified to GEMM

                return; //TODO: Optimize to use GEMM
            }
            MatMul => {
                let (pattern, children) =
                    self.get_pattern_binop(&node, 4, 0);
                let best_pattern = self.get_best_pattern(&children, forge_op);
                if self.pattern_equal(&pattern, self.optim_patterns.get(&MatMul)[best_pattern.2]) {
                    return;
                }
                
                
            }
            Operation::Init => {
                return;
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
