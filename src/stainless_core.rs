use std::sync::{Arc, Mutex};
use crate::array::{DepTree, Object};
use crate::classes::Operations::{*};
use crate::task_queue::TaskQueue;

// use crate::task_queue::TaskQueue;

pub(crate) struct Executor<'a> {
    queue: TaskQueue<'a>,
    name_iter: u8
}

impl Executor<'_> {
    pub fn init() -> Self {
        Self {
            queue: TaskQueue::init(),
            name_iter: 0,
        }
    }

    pub fn build_matrix(&mut self, shape: &Vec<u8>) -> Arc<Mutex<Object>> {
        let mut new_obj = Arc::new(Mutex::new(Object::init(
            self.name_iter,
            shape,
            true,
            Init,
            None,
            None
        )));
        let dep = Box::new(DepTree::init(Arc::clone(&new_obj)));

        new_obj.lock().unwrap().add_dependency(& dep);
        self.queue.push_object(dep);


        self.name_iter += 1;
        return new_obj;
    }

    pub(crate) fn add<'a, 'b, 'c>
        (
            &mut self,
            left: &'a Arc<Mutex<Object<'a, 'a, 'a>>>,
            right: &'b Arc<Mutex<Object<'b, 'b, 'b>>>
        ) -> Arc<Mutex<Object<'a, 'b, 'c>>> {

        let mut new_obj = Arc::new(
            Mutex::new(
            Object::init(

                self.name_iter,
                left.lock().unwrap().get_shape(),
                true,
                Add,
                Some(left),
                Some(right)
                )));
        let dep = Box::new(DepTree::init(Arc::clone(&new_obj)));
        new_obj.lock().unwrap().add_dependency(& dep);


        self.name_iter += 1;
        return new_obj;
    }

    pub fn print_graph() {
        todo!();
    }



}