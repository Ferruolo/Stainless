use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::classes::ItemLoc;
use crate::object::Object;

pub(crate) struct DepTree {
    node: Arc<Mutex<Object>>,
    children: Vec<Rc<DepTree>>,
    location: ItemLoc,
    height: usize,
    num_dependencies: usize,
    name: u64,
}

impl DepTree {
    pub fn init(obj: Arc<Mutex<Object>>, name_lookup: &mut HashMap<u64, Rc<DepTree>>) -> Rc<Self> {
        let name = obj.lock().unwrap().get_name();
        let children: Vec<Rc<DepTree>> = {
            let unwrapped = &obj.lock().unwrap();
            let unwrapped_children = [unwrapped.get_right(), unwrapped.get_left()];
            let child_names: Vec<&Rc<DepTree>> = unwrapped_children
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
        for mut child in children {
            child.increment_num_dependencies();
        }
        let height = children.iter().map(|i| i.get_height()).max().unwrap_or(0) + 1;
        let loc = obj.lock().unwrap().get_loc();
        return Rc::new(Self {
            node: obj,
            children,
            location: loc,
            height,
            num_dependencies: 0,
            name,
        });
    }

    pub fn get_height(&self) -> usize {
        return self.height;
    }

    fn increment_num_dependencies(&mut self) {
        self.num_dependencies += 1;
    }

    pub fn get_name(&self) -> u64 {
        return self.name;
    }

    pub fn get_children(&self) -> &Vec<Rc<DepTree>> {
        return &self.children;
    }

    pub fn get_num_dependencies(&self) -> usize {
        return self.num_dependencies;
    }
}
