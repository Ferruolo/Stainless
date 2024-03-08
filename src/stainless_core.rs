use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;

use crate::array::Object;
use crate::classes::Operation::*;
use crate::classes::ThreadCommands::{CacheMove, Calculation, NullType, FREE, KILL};
use crate::classes::{Operation, ThreadCommands};
use crate::tast_queue::TaskQueue;

pub(crate) struct Executor {
    queue: Arc<Mutex<TaskQueue>>, // Main task queue, accepts push and pop requests,
    // returns items in optimized order
    name_iter: u64, // Current name item
    manager: JoinHandle<()>,
    manager_address: Sender<ThreadCommands>,
    requires_completion_to_kill: bool, // Prevents code from terminating before end of threads
}

impl Executor {
    pub fn init(num_workers: u8) -> Self {
        //Create Queue, queue must be locked in a mutex, despite the fact that execution only pulls from one thread
        let queue = Arc::new(Mutex::new(TaskQueue::init()));
        // Create terminator, kills threads when set to true

        let (manager_address, manager) = 
            Self::initialize_concurrency(num_workers, &queue);

        Self {
            queue,
            name_iter: 0,
            manager,
            manager_address,
            requires_completion_to_kill: true,
        }
    }

    fn initialize_concurrency(
        num_workers: u8,
        queue: &Arc<Mutex<TaskQueue>>,
    ) -> (Sender<ThreadCommands>, JoinHandle<()>) {
        let task_queue = Arc::clone(&queue);
        let (tx, rx) = mpsc::channel();
        let manager_address = tx.clone();
        let manager = thread::spawn(move || {
            let mut workers = Vec::new();
            let mut message_box = Vec::new();
            for i in 0..num_workers {
                Self::initialize_worker(tx.clone(), &mut workers, &mut message_box, i);
            }
            loop {
                match rx.recv().unwrap() {
                    FREE(i, return_address) => {
                        let next_item = task_queue.lock().unwrap().get_next();
                        return_address.send(next_item).unwrap();
                    }
                    CacheMove(_) => continue,
                    Calculation(_) => continue,
                    KILL => {
                        for messenger in message_box {
                            messenger.send(KILL).unwrap()
                        }
                        while !workers.is_empty() {
                            let t = workers.pop().unwrap();
                            t.join().unwrap();
                        }
                        break;
                    }
                    NullType => {}
                }
            }
        });
        (manager_address, manager)
    }

    fn initialize_worker(
        tx: Sender<ThreadCommands>,
        mut workers: &mut Vec<JoinHandle<()>>,
        mut message_box: &mut Vec<Sender<ThreadCommands>>,
        i: u8,
    ) {
        let ret_address = tx.clone();
        let (w_tx, w_rx) = mpsc::channel();
        message_box.push(w_tx.clone());
        workers.push(thread::spawn(move || {
            ret_address.send(FREE(i as usize, w_tx.clone())).unwrap();
            loop {
                let response: ThreadCommands = match w_rx.recv().unwrap() {
                    FREE(_, _) => NullType,
                    CacheMove(..) => FREE(i as usize, w_tx.clone()),
                    Calculation(..) => FREE(i as usize, w_tx.clone()),
                    KILL => {
                        break;
                    }
                    NullType => continue,
                };
                ret_address.send(response).unwrap();
            }
        }));
    }

    pub fn kill(&mut self) {
        loop {
            if !self.queue.lock().unwrap().still_live() {
                break;
            }
        }
        self.manager_address.send(KILL).unwrap()
    }
    fn mat_initializer(
        &mut self,
        shape: &Vec<u64>,
        track_deps: bool,
        forge_op: Operation,
        left: Option<&Arc<Mutex<Object>>>,
        right: Option<&Arc<Mutex<Object>>>,
    ) -> Arc<Mutex<Object>> {
        let parse_children = |a: Option<&Arc<Mutex<Object>>>| {
            return if let Some(c) = a {
                Some(Arc::clone(c))
            } else {
                None
            };
        };

        let new_obj = Arc::new(Mutex::new(Object::init(
            self.name_iter,
            shape,
            track_deps,
            forge_op,
            parse_children(left),
            parse_children(right),
        )));
        let opt_object = self.queue.lock().unwrap().push_object(new_obj);

        self.name_iter += 1;
        return opt_object;
    }

    pub fn create_uniform_random_matrix(&mut self, shape: &Vec<u64>) -> Arc<Mutex<Object>> {
        return self.mat_initializer(shape, true, Init, None, None);
    }

    pub(crate) fn add(
        &mut self,
        left: &Arc<Mutex<Object>>,
        right: &Arc<Mutex<Object>>,
    ) -> Arc<Mutex<Object>> {
        let new_shape = left.lock().unwrap().get_shape().clone();

        return self.mat_initializer(&new_shape, true, Add, Some(left), Some(right));
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
            right.lock().unwrap().get_shape()[1],
        ];
        return self.mat_initializer(&new_shape, true, MatMul, Some(left), Some(right));
    }
    pub fn print_graph(&self) {
        self.queue.lock().unwrap().print_items();
    }
}
