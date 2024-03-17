use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::mem::swap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use libc::pthread_mutexattr_t;
use crate::classes::ItemLoc;
use crate::fibonacci_queue::HeapInterface;
use crate::object::Object;


pub(crate) struct DepTree {
    node: Arc<Mutex<Object>>,
    children: Vec<Arc<Mutex<DepTree>>>,
    location: ItemLoc,
    height: usize,
    num_dependencies: usize,
    name: u64,
    parent: Option<Arc<Mutex<DepTree>>>
}

impl DepTree {
    pub fn init(obj: Arc<Mutex<Object>>, name_lookup: &mut HashMap<u64, Arc<Mutex<DepTree>>>) -> Arc<Mutex<Self>> {
        let name = obj.lock().unwrap().get_name();
        let children: Vec<Arc<Mutex<DepTree>>> = {
            let unwrapped = &obj.lock().unwrap();
            let unwrapped_children = [unwrapped.get_right(), unwrapped.get_left()];
            let child_names: Vec<&Arc<Mutex<DepTree>>> = unwrapped_children
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
            child_names.iter().map(|item| Arc::clone(item)).collect()
        };
        for child in &children {
            child.lock().unwrap().increment_num_dependencies();
        }
        let children_clone = children.clone();
        let height = children.iter().map(|i| i.lock().unwrap().get_height()).max().unwrap_or(0) + 1;
        let loc = obj.lock().unwrap().get_loc();
        let new_item = Arc::new(Mutex::new(Self {
            node: obj,
            children,
            location: loc,
            height,
            num_dependencies: 0,
            name,
            parent: None
        }));

        for child in children_clone {
            child.lock().unwrap().add_parent(&new_item);
        }
        return new_item;
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

    pub fn get_children(&self) -> &Vec<Arc<Mutex<DepTree>>> {
        return &self.children;
    }

    pub fn get_num_dependencies(&self) -> usize {
        return self.num_dependencies;
    }

    pub fn detatch(&mut self) {
        self.children.clear();
        let mut parent = None;
        swap(&mut parent, &mut self.parent);

        if let Some(parent) = parent {
            parent.lock().unwrap().detatch_child(self.name);
        }
    }
    pub fn detatch_child(&mut self, name: u64) {
        println!("{} detached from {}", name, self.name);
        for i in 0..self.children.len() {
            if self.children[i].lock().unwrap().get_name() == name {
                self.children.remove(i);
                break
            }
        }
    }
    pub fn add_parent(&mut self, parent: &Arc<Mutex<DepTree>>) {
        self.parent = Some(parent.clone());
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


impl HeapInterface for Arc<Mutex<DepTree>> {
    fn get_key(&self) -> u64 {
        self.lock().unwrap().node.lock().unwrap().get_name()
    }

    fn compare_items(&self, other: &Self) -> bool {
        let get_height = |x: &Arc<Mutex<DepTree>>|{x.lock().unwrap().get_height()};
        let get_num_dependencies = |x: &Arc<Mutex<DepTree>>|{x.lock().unwrap().num_dependencies};
        let get_name = |x: &Arc<Mutex<DepTree>>| {x.lock().unwrap().get_name()};

        let l_height = get_height(self);
        let r_height = get_height(other);
        let l_num_deps = get_num_dependencies(self);
        let r_num_deps = get_num_dependencies(other);
        let l_name = get_name(self);
        let r_name = get_name(self);


        return if l_height != r_height {
            l_height > r_height
        } else if l_num_deps != r_num_deps {
            l_num_deps < r_num_deps
        } else if l_name != r_name {
            l_name > r_name
        } else {
            false
        }
    }

    fn no_dependencies_remaining(&self) -> bool {
        self.lock().unwrap().children.is_empty()
    }

    fn decrease(&mut self) {
        self.lock().unwrap().increment_num_dependencies();
    }
}