use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use crate::classes::ItemLoc;
use crate::fibonacci_queue::HeapInterface;
use crate::object::Object;


pub(crate) struct DepTree {
    node: Arc<Mutex<Object>>,
    children: Vec<Rc<RefCell<DepTree>>>,
    location: ItemLoc,
    height: usize,
    num_dependencies: usize,
    name: u64,
}

impl DepTree {
    pub fn init(obj: Arc<Mutex<Object>>, name_lookup: &mut HashMap<u64, Rc<RefCell<DepTree>>>) -> Rc<RefCell<Self>> {
        let name = obj.lock().unwrap().get_name();
        let children: Vec<Rc<RefCell<DepTree>>> = {
            let unwrapped = &obj.lock().unwrap();
            let unwrapped_children = [unwrapped.get_right(), unwrapped.get_left()];
            let child_names: Vec<&Rc<RefCell<DepTree>>> = unwrapped_children
                .iter()
                .filter_map(|item| {
                    if let Some(child) = item {
                        let name = child.lock().unwrap().get_name();
                        name_lookup.get(&name)
                    } else {
                        None
                    }
                })
                .collect();
            child_names.iter().map(|item| Rc::clone(item)).collect()
        };
        for child in &children {
            child.borrow_mut().increment_num_dependencies();
        }
        let height = children.iter().map(|i| i.borrow().get_height()).max().unwrap_or(0) + 1;
        let loc = obj.lock().unwrap().get_loc();
        return Rc::new(RefCell::new(Self {
            node: obj,
            children,
            location: loc,
            height,
            num_dependencies: 0,
            name,
        }));
    }

    pub fn get_height(&self) -> usize {
        return self.height;
    }

    pub(crate) fn increment_num_dependencies(&mut self) {
        self.num_dependencies += 1;
    }

    pub fn get_name(&self) -> u64 {
        return self.name;
    }

    pub fn get_children(&self) -> &Vec<Rc<RefCell<DepTree>>> {
        return &self.children;
    }

    pub fn get_num_dependencies(&self) -> usize {
        return self.num_dependencies;
    }

    pub fn kill_children(&mut self) {
        self.children.clear();
    }

    pub fn get_node(&self) -> Arc<Mutex<Object>> {
        return Arc::clone(&self.node);
    }
}


impl Hash for DepTree {
    fn hash<H: Hasher>(&self, state: &mut H) {
        return self.name.hash(state)
    }
}


impl HeapInterface for Rc<RefCell<DepTree>> {
    fn get_key(&self) -> u64 {
        todo!()
    }

    fn compare_items(&self, other: &Self) -> bool {
        if self.borrow().height != other.borrow().height {
            return self.borrow().height < other.borrow().height;
        } else if self.borrow().num_dependencies != other.borrow().num_dependencies {
            return self.borrow().num_dependencies > other.borrow().num_dependencies
        } else if self.borrow().name != other.borrow().name {
            return self.borrow().name < other.borrow().name;
        } else {
            return false
        }
    }

    fn decrease(&mut self) {
        self.borrow_mut().increment_num_dependencies();
    }
}