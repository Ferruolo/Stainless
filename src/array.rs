use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
use crate::classes::{ItemLoc, Operations};

pub(crate) struct Object<'a, 'b, 'c> {
    loc: ItemLoc,
    name: u8,
    shape: Vec<u8>,
    dependency_tree: Option<&'c Box<DepTree<'c, 'c, 'c>>>,
    left:   Option<&'a Arc<Mutex<Object<'a, 'a, 'a>>>>,
    right: Option<&'b Arc<Mutex<Object<'b, 'b, 'b>>>>,
    track_deps: bool,
    forge_op: Operations
}

impl<'a, 'b, 'c> Object<'a, 'b, 'c> {
    pub(crate) fn init(name: u8,
                       shape: &Vec<u8>,
                       track_deps: bool,
                       forge_op: Operations,
                       left: Option<&'a Arc<Mutex<Object<'a, 'a, 'a>>>>,
                       right: Option<&'b Arc<Mutex<Object<'b, 'b, 'b>>>>,
    ) -> Self {
        Self {
            loc: ItemLoc::CPU,
            name,
            shape: shape.clone(),
            dependency_tree: None,
            left,
            right,
            track_deps,
            forge_op,
        }
    }

    pub fn add_dependency(&mut self, dep: &'c Box<DepTree<'c, 'c, 'c>>) {
        self.dependency_tree = Some(dep);
    }

    pub fn get_shape(&self) -> &Vec<u8> {
        &self.shape
    }

    fn get_op(&self) -> Operations {
        self.forge_op
    }

    fn get_left(&self) ->  Option<&'a Arc<Mutex<Object<'a, 'a, 'a>>>> {
        return self.left;
    }
    fn get_right(&self) ->  Option<&'b Arc<Mutex<Object<'b, 'b, 'b>>>> {
        return self.right;
    }

    pub fn get_dep(&self) -> Option<&'c Box<DepTree<'c, 'c, 'c>>> {
        return self.dependency_tree
    }
}

// impl DerefMut for Object {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         return self::
//     }
// }


pub struct DepTree<'a, 'b, 'c> {
    node: Arc<Mutex<Object<'a, 'a, 'a>>>,
    forge_op: Operations,
    children: Vec<&'c Box<DepTree<'b, 'b, 'b>>>
}

impl<'a, 'b> DepTree<'a, 'a, 'a> {
    pub(crate) fn init (refers_to: Arc<Mutex<Object<'a, 'a, 'a>>>) -> Self {
        let mut children = Vec::new();
        for c in [refers_to.lock().unwrap().get_left(),
            refers_to.lock().unwrap().get_right()] {
            if let Some(child) = c{
                if let Some(dep) = child.lock().unwrap().get_dep() {
                    children.push(dep)
                }
            }
        }

        Self {
            node: refers_to,
            forge_op: refers_to.get_op(),
            children
        }
    }
}


//
// pub struct DepTree<'a, 'b, 'c> {
//     node: &'a Arc<Object<'a, 'b, 'c>>,
//     forge_op: Operations,
//     children: Vec<&'c Box<&'b DepTree<'a, 'b, 'c>>>,
//     height: usize,
//     parents: Vec<Box<&'b DepTree<'a, 'b, 'c>>>,
// }
//
// impl DepTree<'_, '_, '_>{
//     pub(crate) fn Init(node: & Arc<Object>) -> Self {
//
//         Self {
//             node,
//             forge_op: node.get_op(),
//             children: vec![],
//             height: 0,
//             parents: Vec::new(),
//         }
//     }
//
//     pub fn get_height(&self) -> usize {
//         return self.height;
//     }
//
//     pub fn get_dependency_lvl(&self) -> usize {
//         return self.height;
//     }
//
//     pub fn get_children(&self) -> &Vec<&Box<&DepTree<'_, '_, '_>>> {
//         return &self.children;
//     }
//
//
//     // Same Forge Op -> Same num children, all <= 3 so this is basically O(1)
//     pub fn is_equal(&self, other: &Box<DepTree>) -> bool {
//         if self.forge_op != other.forge_op {return false;}
//         for c in self.children {
//             for o in other.children {
//                 if !ptr::eq(c, o) {
//                     return false;
//                 }
//             }
//         }
//         return true;
//     }
//
// }



//merge right into left



