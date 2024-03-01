use crate::array::{DepTree, Object};
use crate::classes::Operations::*;
use crate::task_queue::TaskQueue;
use std::sync::{Arc, Mutex};
use libc::iovec;
use crate::classes::Operations;

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
        forge_op: Operations,
        left: Option<&Arc<Mutex<Object>>>,
        right: Option<&Arc<Mutex<Object>>>
    ) -> Arc<Mutex<Object>>{
        let parse_children =
            |a: Option<&Arc<Mutex<Object>>>| {
            if let Some(c) = a {
                return Some(Arc::clone(&c))
            } else {
                return None;
            }
        };


        let mut new_obj = Arc::new(
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
        self.queue.push_object(Arc::clone(&new_obj));

        self.name_iter += 1;
        return new_obj;
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
        return self.mat_initializer(
            left.lock().unwrap().get_shape(),
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




    pub fn print_graph() {
        todo!();
    }
}
