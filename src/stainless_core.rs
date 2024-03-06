use crate::array::{Object};
use crate::classes::Operation::*;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use crate::classes::Operation;
use crate::tast_queue::TaskQueue;





pub(crate) struct Executor {
    queue: Arc<Mutex<TaskQueue>>, // Main task queue, accepts push and pop requests,
                                  // returns items in optimized order
    name_iter: u64, // Current name item
    workers: Vec<JoinHandle<()>>, // Set of all worker threads, allows for easy management of threads
    terminator: Arc<Mutex<bool>>, // Tells threads when terminate
    requires_completion_to_kill: bool // Prevents code from terminating before end of threads
}

impl Executor {
    pub fn init(num_workers: u8) -> Self {
        //Create Queue, queue must be locked in a mutex, despite the fact that execution only pulls from one thread
        let queue = Arc::new(Mutex::new(TaskQueue::init()));
        // Create terminator, kills threads when set to true
        let terminator = Arc::new(Mutex::new(false));
        let mut workers = Vec::new();
        //TODO: Move to hub and spoke model, as this is easier to generalize to distributed systems
        for _ in 0..num_workers {
            //Initiate reference to queue so that worker can pull the next
            let queue_ref = Arc::clone(&queue);
            let terminator = Arc::clone(&terminator);
            workers.push(thread::spawn(
                move || {
                    loop {
                        //terminate if everythings done
                        if *terminator.lock().unwrap() {
                            break
                        }
                        // Get item, if one exists
                        let item = {
                            if let Some(item) = queue_ref.lock().unwrap().get_next() {
                                item
                            } else {
                                continue;
                            }
                        };
                        // Wrap up if
                        let forge_op = item.lock().unwrap().get_op();
                        match forge_op {
                            Add => {
                                let target = item.lock().unwrap();
                                let left = target.get_left();
                                let right = target.get_right();
                                match (left, right) {
                                    (Some(l), Some(r)) => {
                                        let l_name = l.lock().unwrap().get_name();
                                        let r_name = r.lock().unwrap().get_name();
                                        println!("{} + {} = {}", l_name, r_name, target.get_name());
                                    }
                                    (_, _) => {
                                        println!("Add item has no ")
                                    }
                                }
                            }
                            MatMul => {
                                println!("MatMul");
                            }
                            Init => {
                                continue;
                            }
                        }


                    }
                }
            ))
        }

        Self {
            queue,
            name_iter: 0,
            workers,
            terminator,
            requires_completion_to_kill: true
        }
    }

    pub fn kill(&mut self) {
        loop {
            if !self.queue.lock().unwrap().still_live() {
                break
            }
        }
        *self.terminator.lock().unwrap() = true;
        // TODO: Join Threads
    }
    fn mat_initializer(
        &mut self,
        shape: &Vec<u64>,
        track_deps: bool,
        forge_op: Operation,
        left: Option<&Arc<Mutex<Object>>>,
        right: Option<&Arc<Mutex<Object>>>
    ) -> Arc<Mutex<Object>>{
        let parse_children =
            |a: Option<&Arc<Mutex<Object>>>| {
                return if let Some(c) = a {
                    Some(Arc::clone(c))
                } else {
                    None
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
        let opt_object = self.queue.lock().unwrap().push_object(new_obj);

        self.name_iter += 1;
        return opt_object;
    }



    pub fn create_uniform_random_matrix(&mut self, shape: &Vec<u64>) -> Arc<Mutex<Object>> {
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
        self.queue.lock().unwrap().print_items();
    }


}
