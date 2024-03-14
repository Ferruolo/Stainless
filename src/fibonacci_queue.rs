use std::collections::HashMap;
use std::rc::Rc;

use crate::dep_tree::DepTree;

pub trait FibInterface {
    fn get_key(&self) -> u64;
    fn compare_items(&self, other: &Self) -> bool;
}

struct FibNode<T: Clone> {
    data: T,
    parent: Option<Rc<FibNode<T>>>,
    child: Option<Rc<FibNode<T>>>,
    left_sibling: Option<Rc<FibNode<T>>>,
    right_sibling: Option<Rc<FibNode<T>>>,
    degree: usize,
    mark: bool,
}

impl<T: Clone> FibNode<T> {
    fn new(data: T) -> Rc<Self> {
        Rc::new(FibNode {
            data,
            parent: None,
            child: None,
            left_sibling: (None),
            right_sibling: (None),
            degree: (0),
            mark: false,
        })
    }
}

pub struct FibonacciHeap<T: FibInterface + Clone> {
    min: Option<Rc<FibNode<T>>>,
    data: HashMap<u64, Rc<FibNode<T>>>,
}

pub struct DepTreeFibonacciHeap {
    heap: FibonacciHeap<Rc<DepTree>>,
    nodes: HashMap<u64, Rc<FibNode<Rc<DepTree>>>>, // Map from name to DepTreeNode
}

impl<T: FibInterface + Clone> FibonacciHeap<T> {
    pub fn new() -> Self {
        FibonacciHeap {
            min: (None),
            data: HashMap::new(),
        }
    }

    pub fn insert(&mut self, data: T) -> Rc<FibNode<T>> {
        let node = FibNode::new(data);
        self.data.insert(data.get_key(), Rc::clone(&node));

        if self.min.is_none() {
            self.min.replace(Rc::clone(&node));
        } else {
            self.union_root_lists(node);
        }

        Rc::clone(&node)
    }

    pub fn get_min(&self) -> Option<T> {
        match &self.min {
            None => {None}
            Some(a) => {
                Some(a.data.clone())
            }
        }
    }

    pub fn extract_min(&mut self) -> Option<T> {
        let mut min_node = self.min.take()?;
        let mut child = min_node.child.take();

        while let Some(child_node) = child {
            let next_child = child_node
                .right_sibling
                .as_ref()
                .map(|n| Rc::clone(n));
            self.union_root_lists(Rc::clone(&child_node));
            child = next_child;
        }

        let min_data = min_node.data.clone();
        self.data.remove(&min_data.get_key());

        if let Some(new_min) = self.find_new_min() {
            self.min.replace(new_min);
        }

        Some(min_data)
    }

    fn find_new_min(&self) -> Option<Rc<FibNode<T>>> {
        let mut min_node: Option<Rc<FibNode<T>>> = None;
        for node in self.data.values() {
            if let Some(ref current_min) = min_node {
                if node.data.compare_items(&current_min.data) {
                    min_node = Some(Rc::clone(node));
                }
            } else {
                min_node = Some(Rc::clone(node));
            }
        }
        min_node
    }

    fn union_root_lists(&mut self, node: Rc<FibNode<T>>) {
        let mut current = self.min.take();
        self.min.replace(Rc::clone(&node));

        let mut node_ref = Some(Rc::clone(&node));
        let mut current_ref = current.as_ref();

        loop {
            let next_node = node_ref
                .as_ref()
                .and_then(|n| n.right_sibling.as_ref().cloned());
            let next_current = current_ref
                .as_ref()
                .and_then(|n| n.right_sibling.as_ref().cloned());

            if let Some(n) = next_node {
                node_ref = Some(n);
            } else {
                break;
            }

            if let Some(c) = next_current {
                current_ref = Some(&c);
            } else {
                break;
            }
        }

        if let (Some(node_ref), Some(current_ref)) = (node_ref, current_ref) {
            node_ref
                .right_sibling
                .replace(Rc::clone(&current_ref));
            current_ref
                .right_sibling
                .replace(Rc::clone(&node_ref));
        }
    }
}

impl DepTreeFibonacciHeap {
    pub(crate) fn new() -> Self {
        DepTreeFibonacciHeap {
            heap: FibonacciHeap::new(),
            nodes: HashMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, tree: Rc<DepTree>) {
        let node = self.heap.insert(Rc::clone(&tree));
        self.nodes.insert(tree.get_name(), Rc::clone(&node));
    }

    pub(crate) fn update_num_dependencies(
        &mut self,
        name: u64
    ) -> Option<Rc<DepTree>> {
        let node = self.nodes.get(&name)?;
        let mut tree = &mut node.data;

        tree.increment_num_dependencies();
        Some(Rc::clone(&tree))
    }

    pub(crate) fn get_min(&self) -> Option<Rc<DepTree>> {
        self.heap.get_min()
    }

    pub(crate) fn extract_min(&mut self) -> Option<Rc<DepTree>> {
        let min_tree = self.heap.extract_min()?;
        self.nodes.remove(&min_tree.get_name());
        Some(min_tree)
    }
}
