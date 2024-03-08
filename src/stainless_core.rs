use std::ops::ControlFlow::Continue;
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
    manager: JoinHandle<()>, // Manager thread, controls all other threads
    manager_address: Sender<ThreadCommands>, // Address of manager thread
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
            manager_address
        }
    }

    /*
    * Initializes Manager Queue, moved out of INIT for readability
    * Defines all workers, then loops while waiting to receive instructions (See Classes/Instruction
    * for possible instructions). Upon receiving instructions, it matches the result, and takes the
    * corresponding action, which usually means taking the next item from the queue and sending it to the next
    * available thread
    */
    fn initialize_concurrency(
        num_workers: u8,
        queue: &Arc<Mutex<TaskQueue>>,
    ) -> (Sender<ThreadCommands>, JoinHandle<()>) {
        // Clone Task Queue to move it to threads
        let task_queue = Arc::clone(&queue);
        let (tx, rx) = mpsc::channel();
        let manager_address = tx.clone();


        /*
        * Spawn Manager thread here:
        * This thread reads in all tasks from the task queue, and decides where to send them.
        * Currently, this choice is arbitrary. In the future, destination of tasks, along with
        * management of memory allocation, will happen in this thread. This thread also works to
        * prevent race conditions, or any other accidents that can come from memory sharing model
        */
        let manager = thread::spawn(move || {
            // Initialize the worker threads, no computation is performed by the main thread
            let mut workers = Vec::new();
            let mut message_box = Vec::new();
            for i in 0..num_workers {
                Self::initialize_worker(tx.clone(), &mut workers, &mut message_box, i);
            }

            // Loop on manager
            loop {
                // Tasks only happen when a message is received. Workers notify manager whenever a
                // task is complete
                match rx.recv().unwrap() {

                    FREE(i, return_address) => {
                        // Send next availible task to the free worker
                        let next_item = task_queue.lock().unwrap().get_next();
                        return_address.send(next_item).unwrap();
                    }
                    KILL => {
                        // Terminates all threads after computation is complete
                        // TODO: Need to wait for everything to be done before we kill
                        for messenger in message_box {
                            messenger.send(KILL).unwrap()
                        }
                        while !workers.is_empty() {
                            let t = workers.pop().unwrap();
                            t.join().unwrap();
                        }
                        break;
                    }
                    // No other relevant messages for manager right now
                    _ => {}
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
        // Initialize manager address, message channel
        let ret_address = tx.clone();
        let (w_tx, w_rx) = mpsc::channel();
        //Add message channel to manager's list of message channels
        message_box.push(w_tx.clone());

        // Initialize worker thread and add it to manager's list of threads
        workers.push(thread::spawn(move || {
            // When Initialized, request first task from manager
            ret_address.send(FREE(i as usize, w_tx.clone())).unwrap();
            /*
             Worker loops on receiver. When it receives a task it uses matching to determine
             course of action, then preforms task, and then signals that it is ready for next task.
             There is of now no need to perform any other actions, or to return values.
            */
            loop {
                let response: ThreadCommands = match w_rx.recv().unwrap() {
                    CacheMove(..) => FREE(i as usize, w_tx.clone()),
                    Calculation(wrapped_target) => {
                        let target = wrapped_target.lock().unwrap();
                        // Match target calculation, lock target at beginning in case
                        // other items need it
                        match target.get_op() {
                            Add => {
                                let (left, right) = match (target.get_left(), target.get_right()) {
                                    (Some(l), Some(r)) => {
                                        (Arc::clone(l), Arc::clone(r))
                                    }
                                    _ => {panic!("Invalid Add Operation made it to execution")}
                                };
                            }
                            MatMul => {
                                let (left, right) = match (target.get_left(), target.get_right()) {
                                    (Some(l), Some(r)) => {
                                        (Arc::clone(l), Arc::clone(r))
                                    }
                                    _ => {panic!("Invalid MatMul Operation made it to execution")}
                                };

                            }
                            Init => {}
                        }


                        FREE(i as usize, w_tx.clone())
                    },
                    KILL => {
                        break;
                    }
                    _ => {continue}
                };
                ret_address.send(response).unwrap();
            }
        }));
    }

    pub fn kill(&mut self) {
        // Kills when everything has been computed
        loop {
            // Move this loop to inside manager thread somehow
            if !self.queue.lock().unwrap().still_live() {
                break;
            }
        }
        self.manager_address.send(KILL).unwrap()
    }


    /*
    * This is a utility function, which allows for all other functions to easily initialize
    * matrices without having to write repeat code 10x.
    */
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


    /*
    * Creates uniform random matrix of given size. Defaults to initializing on CUDA
     */
    pub fn create_uniform_random_matrix(&mut self, shape: &Vec<u64>) -> Arc<Mutex<Object>> {
        return self.mat_initializer(shape, true, Init, None, None);
    }


    /*
    * Schedules the add of two matrices and returns target object
    */
    pub(crate) fn add(
        &mut self,
        left: &Arc<Mutex<Object>>,
        right: &Arc<Mutex<Object>>,
    ) -> Arc<Mutex<Object>> {
        let new_shape = left.lock().unwrap().get_shape().clone();

        return self.mat_initializer(&new_shape, true, Add, Some(left), Some(right));
    }

    /*
    * Schedules the multiplication of two matrices and returns target object
    */
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
