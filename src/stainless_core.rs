use crate::array::{Object};
use crate::classes::Operation::*;
use crate::task_queue::TaskQueue;
use std::sync::{Arc, Mutex};
use crate::classes::Operation;

// use crate::task_queue::TaskQueue;

pub(crate) struct Executor {
    queue: TaskQueue,
    name_iter: u64,
}

impl Executor {
    pub fn init() -> Self {
        Self {
            queue: TaskQueue::init(),
            name_iter: 0,
        }
    }
    fn mat_initializer(
        &mut self,
        shape: &Vec<u8>,
        track_deps: bool,
        forge_op: Operation,
        left: Option<&Arc<Mutex<Object>>>,
        right: Option<&Arc<Mutex<Object>>>
    ) -> Arc<Mutex<Object>>{
        let parse_children =
            |a: Option<&Arc<Mutex<Object>>>| {
            if let Some(c) = a {
                return Some(Arc::clone(c))
            } else {
                return None;
            }
        };


        let new_obj = Arc::new(
            Mutex::new(
                Object::init(
                    self.name_iter,
                    shape,
                    track_deps,
                    forge_op,
                    parse_children(left),
                    parse_children(right),
                )
            )
        );
        let opt_object = self.queue.push_object(new_obj);

        self.name_iter += 1;
        return opt_object;
    }



    pub fn build_matrix(&mut self, shape: &Vec<u8>) -> Arc<Mutex<Object>> {
        return self.mat_initializer(
            shape, true, Init, None, None
        );
    }

    pub(crate) fn add(
        &mut self,
        left: &Arc<Mutex<Object>>,
        right: &Arc<Mutex<Object>>,
    ) -> Arc<Mutex<Object>> {
        let new_shape = left.lock().unwrap().get_shape().clone();

        return self.mat_initializer(
            &new_shape,
            true,
            Add,
            Some(left),
            Some(right)
        );
    }

    pub(crate) fn mat_mul(
        &mut self,
        left: &Arc<Mutex<Object>>,
        right: &Arc<Mutex<Object>>,
    ) -> Arc<Mutex<Object>> {
        if left.lock().unwrap().get_shape()[1] != right.lock().unwrap().get_shape()[0] {
            panic!("Received Bad Shapes")
        }


        let new_shape = vec![
            left.lock().unwrap().get_shape()[0],
            right.lock().unwrap().get_shape()[1]
        ];
        return self.mat_initializer(
            &new_shape,
            true,
            MatMul,
            Some(left),
            Some(right)
        );
    }




    pub fn print_graph(&self) {
        self.queue.print_items();
    }


}
