use std::collections::HashMap;
use std::hash::Hash;
// Have to roll our own to keep with the whole point of this project
use std::rc::Rc;
use crate::dep_tree::DepTree;

// Define the Node struct for the Fibonacci heap
struct Node<T: Ord + Hash> {
    data: T,
    degree: usize,
    parent: Option<Rc<Node<T>>>,
    children: Vec<Rc<Node<T>>>,
    mark: bool,
}

// Define the Fibonacci heap struct
pub struct FibonacciHeap<T: Ord + Hash> {
    min: Option<Rc<Node<T>>>,
    n: usize,
    lookup: HashMap<T, Node<T>>
}

impl<T: Ord + Hash> FibonacciHeap<T> {
    // Creates a new, empty Fibonacci heap
    pub fn new() -> Self {
        FibonacciHeap {
            min: None,
            n: 0,
            lookup: HashMap::new()
        }
    }

    // Inserts a new node with the given data into the heap
    pub fn insert(&mut self, data: T) {
        let new_node = Rc::new(Node {
            data,
            degree: 0,
            parent: None,
            children: vec![],
            mark: false,
        });

        // If the heap is empty, set the new node as the minimum
        if self.min.is_none() {
            self.min = Some(Rc::clone(&new_node));
        } else {
            // Otherwise, insert the new node into the root list
            self.insert_into_root_list(&new_node);

            // Update the minimum if the new node has a smaller value
            if new_node.data < self.min.as_ref().unwrap().data {
                self.min = Some(Rc::clone(&new_node));
            }
        }

        self.n += 1;
    }

    // Returns a reference to the node with the minimum key in the heap
    pub fn minimum(&self) -> Option<&T> {
        self.min.as_ref().map(|node| &node.data)
    }

    // Removes and returns the node with the minimum key from the heap
    pub fn extract_min(&mut self) -> Option<T> {
        self.min.take().map(|min_node| {
            let mut min_clone = Rc::clone(&min_node);
            let min_data = min_clone.data.clone();
            self.n -= 1;

            // Add all children of the minimum node to the root list
            let mut children = vec![];
            std::mem::swap(&mut children, &mut min_clone.children);
            for mut child in children {
                child.parent = None;
            }
            self.min = self.merge_roots(children);

            // Consolidate the heap after removing the minimum node
            self.consolidate();

            min_data
        })
    }

    // Joins two Fibonacci heaps into a new one, consisting of all the nodes from the two heaps
    pub fn union(&mut self, other: &mut Self) {
        if self.min.is_none() {
            self.min = other.min.take();
        } else if let Some(other_min) = other.min.take() {
            self.min = self.merge_roots(vec![self.min.take().unwrap(), other_min]);
        }
        self.n += other.n;
        other.n = 0;
        self.consolidate();
    }

    // Decreases the key of a node to the new value
    pub fn decrease_key(&mut self, node: &Rc<Node<T>>, new_data: T) {
        if new_data > node.data {
            panic!("New key is greater than the current key");
        }

        let mut node_clone = Rc::clone(node);
        node_clone.data = new_data;

        // If the node is not a root node and its new value is smaller than its parent's value,
        // cut the node and perform cascading cuts
        if let Some(parent) = node_clone.parent.as_ref() {
            if new_data < parent.data {
                self.cut(&node_clone);
                self.cascading_cut(&node_clone);
            }
        }

        // Update the minimum node if the new value is smaller than the current minimum
        if new_data < self.min.as_ref().unwrap().data {
            self.min = Some(Rc::clone(&node_clone));
        }
    }

    // Deletes a node from the heap
    pub fn delete(&mut self, node: &Rc<Node<T>>) {
        let min_data = self.min.as_ref().unwrap().data.clone();
        self.decrease_key(node, min_data);
        self.extract_min();
    }

    // Helper function to insert a node into the root list
    fn insert_into_root_list(&mut self, node: &Rc<Node<T>>) {
        let mut node_clone = Rc::clone(node);
        let mut min_clone = self.min.as_ref().unwrap().clone();

        let last_child = (*min_clone.children.last().unwrap()).clone();
        min_clone.children.push(Rc::clone(&node_clone));
        node_clone.parent = Some(Rc::clone(&min_clone));
        last_child.children.push(Rc::clone(&node_clone));
        node_clone.children.push(last_child);
    }

    // Helper function to merge root lists
    fn merge_roots(&self, mut roots: Vec<Rc<Node<T>>>) -> Option<Rc<Node<T>>> {
        if roots.is_empty() {
            return None;
        }

        let mut merged_root = roots.remove(0);
        let mut curr = Rc::clone(&merged_root);
        let mut next;

        for root in roots {
            next = Rc::clone(&root);
            let mut next_clone = Rc::clone(&next);
            let mut curr_clone = Rc::clone(&curr);
            next_clone.children.push(Rc::clone(&curr));
            curr_clone.parent = Some(Rc::clone(&next));
            curr = next;
        }

        let mut curr_clone = Rc::clone(&curr);
        curr_clone.parent = Some(Rc::clone(&merged_root));
        Some(merged_root)
    }

    // Helper function to consolidate the heap
    fn consolidate(&mut self) {
        let mut degree_table = vec![None; self.n + 1];
        let mut curr = self.min.take();

        while let Some(node) = curr {
            let mut node_clone = Rc::clone(&node);
            let d = node_clone.degree;

            while let Some(existing) = degree_table[d].take() {
                let mut existing_clone = Rc::clone(&existing);
                if existing_clone.data < node_clone.data {
                    std::mem::swap(&mut existing_clone.children, &mut node_clone.children);
                    for child in &mut node_clone.children {
                        child.parent = Some(Rc::clone(&existing));
                    }
                    node_clone.parent = Some(Rc::clone(&existing));
                    existing_clone.children.push(Rc::clone(&node));
                    existing_clone.degree += 1;
                    curr = Some(existing);
                    node_clone.mark = false;
                } else {
                    existing_clone.parent = Some(Rc::clone(&node));
                    node_clone.children.push(existing);
                    node_clone.degree += 1;
                    curr = Some(Rc::clone(&node));
                    existing_clone.mark = false;
                }
            }

            degree_table[d] = Some(curr.clone().unwrap());
            curr = self.min.take();
        }

        self.min = None;
        for node in degree_table.into_iter().filter_map(|n| n) {
            let node_clone = Rc::clone(&node);
            if self.min.is_none() || node_clone.data < self.min.as_ref().unwrap().data {
                self.min = Some(Rc::clone(&node));
            } else {
                let mut min_clone = Rc::clone(&self.min.as_ref().unwrap());
                let mut node_clone = Rc::clone(&node);
                let last_child = (*min_clone.children.last().unwrap()).clone();
                min_clone.children.push(Rc::clone(&node));
                node_clone.parent = Some(Rc::clone(&self.min.as_ref().unwrap()));
                last_child.children.push(Rc::clone(&node));
                node_clone.children.push(last_child);
            }
        }
    }

    // Helper function to cut a node from its parent
    fn cut(&mut self, node: &Rc<Node<T>>) {
        let mut node_clone = Rc::clone(node);
        if let Some(parent) = node_clone.parent.as_ref() {
            let mut parent_clone = Rc::clone(parent);
            parent_clone.children.retain(|child| !Rc::ptr_eq(child, &node_clone));
            parent_clone.degree -= 1;
            node_clone.parent = None;
            node_clone.mark = false;
            self.insert_into_root_list(&node_clone);
        }
    }

    // Helper function for cascading cuts
    fn cascading_cut(&mut self, node: &Rc<Node<T>>) {
        let mut curr = Some(Rc::clone(node));
        while let Some(parent) = curr.as_ref().unwrap().parent.as_ref() {
            let mut parent_clone = Rc::clone(parent);
            if parent_clone.mark {
                self.cut(&parent_clone);
                curr = Some(Rc::clone(&parent_clone.parent.as_ref().unwrap()));
            } else {
                parent_clone.mark = true;
                curr = None;
            }
        }
    }

    pub fn add_dependencies(&mut self, item: &Rc<DepTree>) {
        let node = match self.lookup.get(&item) {
            None => {return }
            Some(node) => {node}
        };
        item.increment_num_dependencies();
        
    }

}