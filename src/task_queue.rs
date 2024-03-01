extern crate queues;

use crate::array::DepTree;
// use crate::array::{DepTree, Object};



pub struct TaskQueue <'a>{
    dependency_graph: Vec<Box<DepTree<'a, 'a, 'a>>>,

}




impl <'a> TaskQueue <'a> {
    pub(crate) fn init() -> Self{
        Self {
            dependency_graph: Vec::new()
        }
    }
    
    pub fn push_object(&mut self, item: Box<DepTree<'a, 'a, 'a>>) {
        self.dependency_graph.push(item)
    }



    // fn merge_into_dependency_graph(&mut self, item: &Box<DepTree>) -> &Box<DepTree> {
    //
    //     for i in self.dependency_graph[0] {
    //         if (item.is_equal(i)) {
    //             i.merge(item);
    //             return i;
    //         }
    //     }
    //     item
    // }
}


