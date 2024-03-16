use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use crate::binding_interface::{create_uniform_random_mat_interface, matrix_mul_interface};

use crate::classes::ThreadCommands::*;
use crate::classes::{MatrixInitType, Operation, ThreadCommands};
use crate::classes::MatrixInitType::UniformRandomMatrix;
use crate::classes::Operation::{Init, MatrixMult};
use crate::object::Object;
use crate::task_scheduler::Scheduler;

pub(crate) struct MultiThread {
    // Manager Thread, controls all other threads. All other threads
    // Live within and are managed by this thread. Holds all tasks and schedules
    // them for executions
    manager: Rc<JoinHandle<()>>,
    // Inbox for manager, only way to send items to the
    // manager to be executed, and to schedule kill
    manager_inbox: Sender<ThreadCommands>,
    // Increments, makes sure every item has unique name
    name_iter: u64,
}


/*

*/
pub(crate) trait Executor {
    fn init(num_workers: u8) -> Self;
    // fn build_constant_matrix();
    // fn build_diagonal_matrix();

    ///
    ///
    /// # Arguments
    ///
    /// * `shape`:
    /// * `left`:
    /// * `right`:
    /// * `forge_op`:
    ///
    /// returns: Arc<Mutex<Object>, Global>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    fn matrix_builder(&self,
                      shape: &Vec<u64>,
                      left: Option<&Arc<Mutex<Object>>>,
                      right: Option<&Arc<Mutex<Object>>>,
                      forge_op: Operation
    ) -> Arc<Mutex<Object>>;
    ///
    ///
    /// # Arguments
    ///
    /// * `shape`:
    ///
    /// returns: ()
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    fn build_uniform_random_matrix(&self, shape: &Vec<u64>) -> Arc<Mutex<Object>>;


    fn mat_mul(&self, left: &Arc<Mutex<Object>>, right: &Arc<Mutex<Object>>) -> Arc<Mutex<Object>>;


    fn kill(&self);
}







impl Executor for MultiThread {
    fn init(num_workers: u8) -> Self {
        let (manager, message_box) = spin_up(num_workers);
        Self {
            manager: Rc::new(manager),
            manager_inbox: message_box,
            name_iter: 0,
        }
    }
    fn matrix_builder(&self,
                      shape: &Vec<u64>,
                      left: Option<&Arc<Mutex<Object>>>,
                      right: Option<&Arc<Mutex<Object>>>,
                      forge_op: Operation
    ) -> Arc<Mutex<Object>> {
        let handle_deps = |x: Option<&Arc<Mutex<Object>>>| {
            match x {
                None => {None}
                Some(item) => {Some(Arc::clone(item))}
            }
        };


        let new_obj = Arc::new(Mutex::new(Object::init(
            self.name_iter,
            &shape,
            true,
            forge_op,
            handle_deps(left),
            handle_deps(right)
        )));
        self.manager_inbox.send(Calculation(Arc::clone(&new_obj))).unwrap();
        return new_obj;
    }
    fn build_uniform_random_matrix(&self, shape: &Vec<u64>) -> Arc<Mutex<Object>> {
        self.matrix_builder(
            shape, None, None, Init(UniformRandomMatrix)
        )
    }

    fn mat_mul(&self, left: &Arc<Mutex<Object>>, right: &Arc<Mutex<Object>>) -> Arc<Mutex<Object>> {
        let get_shape = |x: &Arc<Mutex<Object>>| {
            x.lock().unwrap().get_shape().clone()
        };

        let left_shape = get_shape(left);
        let right_shape = get_shape(right);

        if left_shape[1] != right_shape[0] {
            let left_name = left.lock().unwrap().get_name();
            let right_name = right.lock().unwrap().get_name();
            panic!("Error computing product of {} and {} - Shapes did not match", left_name, right_name);
        }


        let new_shape = vec![left_shape[0], right_shape[1]];
        self.matrix_builder(&new_shape, Some(left), Some(right), MatrixMult)
    }

    fn kill(&self) {
        self.manager_inbox.send(KILL).unwrap();
    }

    // fn build_constant_matrix() {}
    //
    // fn build_diagonal_matrix() {}


}

// Spins up and defines the manager
#[inline]
fn spin_up(num_workers: u8) -> (JoinHandle<()>, Sender<ThreadCommands>) {
    // Define the communicate channel
    let (sender, receiver): (Sender<ThreadCommands>, Receiver<ThreadCommands>) = mpsc::channel();
    let ret_sender = sender.clone();
    return (
        thread::spawn(move || {
            // initialize scheduler and workers
            let mut scheduler = Scheduler::init();
            let workers = initialize_workers(num_workers, &sender);
            let mut worker_queue: VecDeque<Sender<ThreadCommands>> = VecDeque::new();
            loop {
                if scheduler.can_kill() {break;}
                // Read next instruction
                match receiver.try_recv().unwrap_or(NullType) {
                    // FREE -> A thread has been freed. Either schedule item
                    // or add the worker to the queue
                    FREE(address) => match scheduler.get_next() {
                        None => worker_queue.push_back(address),
                        Some(cmd) => address.send(cmd).unwrap(),
                    },
                    // Kill all threads
                    KILL => {
                        scheduler.schedule_shutdown();
                    }
                    // If there's nothing to do, lets do some of our work on
                    // task
                    NullType => {
                        if let Some(address) = worker_queue.pop_front() {
                            if let Some(task) = scheduler.get_next() {
                                address.send(task).unwrap()
                            } else {
                                worker_queue.push_front(address);
                            }
                        }
                    }
                    // Schedule a new object for computation
                    ComputeObject(obj) => {
                        match worker_queue.pop_front() {
                            None => {
                                scheduler.schedule(obj)
                            }
                            Some(sender) => {
                                sender.send(ComputeObject(obj)).unwrap();
                            }
                        }
                    }
                    _ => continue,
                }
            }

            kill_workers(workers);
        }),
        ret_sender,
    );
}


/*
    Spin up and define worker processes. Will hopefully be inlined to avoid function overhead
*/
#[inline]
fn initialize_workers(
    num_workers: u8,
    manager_address: &Sender<ThreadCommands>,
) -> Vec<JoinHandle<()>> {
    let mut workers = Vec::new();
    for _i in 0..num_workers {
        // Define chanel for worker
        let (tx, rx): (Sender<ThreadCommands>, Receiver<ThreadCommands>) = mpsc::channel();
        let manager_address_local = manager_address.clone();
        workers.push(thread::spawn(move || loop {
            // Read in new message from manager
            match rx.recv().unwrap() {
                // TODO: Cache Move
                CacheMove(_) => {}
                Calculation(obj) => {
                    perform_calculation(obj);

                }
                KILL => break,
                _ => {
                    continue;
                }
            }
            manager_address_local.send(FREE(tx.clone())).unwrap()
        }))
    }
    return workers;
}

/*
*   Cleans up workers
*/
fn kill_workers(workers: Vec<JoinHandle<()>>) {
    for worker in workers {
        worker.join().unwrap();
    }
}
/*
    Performs calculations
*/
#[inline]
fn perform_calculation(target: Arc<Mutex<Object>>) {
    let forge_op = target.lock().unwrap().get_op();
    match forge_op {
        Operation::Add => {
            todo!();
        }
        Operation::MatrixMult => unsafe {
            let left = Arc::clone(&target.lock().unwrap().get_left().unwrap());
            let right = Arc::clone(&target.lock().unwrap().get_left().unwrap());
            let new_mat = matrix_mul_interface(left, right);
            target.lock().unwrap().set_matrix(new_mat);
        }
        Init(matType) => {
            match matType {
                UniformRandomMatrix => unsafe {
                    let new_shape = {
                        target.lock().unwrap().get_shape().clone()
                    };
                    let new_matrix = create_uniform_random_mat_interface(&new_shape);
                    target.lock().unwrap().set_matrix(new_matrix);
                }
                MatrixInitType::ConstantMatrix(_) => {}
                MatrixInitType::DiagonalMatrix(_) => {}
            }
        }
    }
}
