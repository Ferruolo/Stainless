use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::array::Object;
use crate::classes::{ItemLoc, Operation};



/*
*  Dep Tree:
*  - Data Structure to track elements inside of task queue
*  - Helps efficiently schedule elements by watching dependencies
*  - Possibly can be integrated back into Object class, but might not need to
*/
pub struct DepTree {
    node: Arc<Mutex<Object>>, // Reference back to main respective object
    forge_op: Operation,      // Operation to forge DepTree
    children: Vec<Arc<Mutex<DepTree>>>, // List of DepTree Children (the dependencies of this node)
    height: usize,            // Height of item
    shape: Vec<u64>,          // Shape of respective node
    num_parents: u8,          // Number of items which depdend on this one
    loc: ItemLoc,              // Location of this nodes
}

impl DepTree {
    pub(crate) fn init(refers_to: Arc<Mutex<Object>>) -> Self {
        //Link children to this node
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
        //Calculate Height and shapes
        let max_height = children.iter().map(|c| c.lock().unwrap().height + 1).max().unwrap_or(0);
        let shape = refers_to.lock().unwrap().get_shape().clone();
        let loc = refers_to.lock().unwrap().get_loc();
        Self {
            node: Arc::clone(&refers_to),
            forge_op: refers_to.lock().unwrap().get_op(),
            children,
            height: max_height,
            shape: shape,
            num_parents: 0,
            loc,
        }
    }

    /*
    * If two item are calculated in the same way
    * we want to merge them so that we only have to do the calculation once.
    */
    pub fn merge(&self, other: &Arc<Mutex<DepTree>>) -> Option<Arc<Mutex<Object>>> {
        let len = other.lock().unwrap().children.len();
        let o_forge_op = other.lock().unwrap().forge_op;
        // Make sure items have same number of children and same forge op

        if len == 0 || len != self.children.len() || o_forge_op != self.forge_op {
            return None;
        }
        // If children and forge op match, get rid of the other object and return
        // the node for this tree
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

    /*
    * Returns number of children
    */
    pub fn get_num_children(&self) -> usize {
        return self.children.len();
    }

    /*
    * Returns list of children
    */
    pub fn get_children(&self) -> Vec<Arc<Mutex<DepTree>>> {
        return self.children.clone();
    }

    /*
    * Returns height of children
    */
    pub fn get_height(&self) -> usize {
        return self.height;
    }

    /*
    * Returns name of respective node
    */
    pub fn get_name(&self) -> u64 {
        return self.node.lock().unwrap().get_name();
    }

    /*
     * Returns forge op for this node
     */
    pub fn get_forge_op(&self) -> Operation {
        return self.forge_op;
    }

    /*
    *   Returns the shape of the respective object
    */
    pub fn get_shape(&self) -> &Vec<u64> {
        return &self.shape;
    }

    /*
    *   Returns the respective node for the items
    */
    pub fn get_node(&self) -> Arc<Mutex<Object>> {
        return Arc::clone(&self.node);
    }

    /*
    *   increase number of parents by 1
    */
    pub fn increment_num_parents(&mut self) {
        self.num_parents += 1
    }

    /*
    *   returns number of parents
    */
    pub fn get_num_parents(&self) -> u8 {
        return self.num_parents;
    }

    /*
    * Erase node from existence (make sure there are no remaining references)
    */
    pub fn erase(&self) {
        todo!();
    }

    /*
    * Returns location of respective node
    */
    pub fn get_loc(&self) -> ItemLoc {
        return self.loc;
    }
}