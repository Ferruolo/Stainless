use mpsc::channel;
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::thread::JoinHandle;

use crate::binding_interface::{
    create_uniform_random_mat_interface, matrix_mul_interface, print_matrix_interface,
};
use crate::classes::MatrixInitType::UniformRandomMatrix;
use crate::classes::Operation::{Init, MatrixMult};
use crate::classes::ThreadCommands::{CacheMove, ComputeObject, NullType, FREE, KILL};
use crate::classes::{MatrixInitType, Operation, ThreadCommands};
use crate::object::{Object, ObjectInterface};
use crate::task_scheduler::Scheduler;

// Spins up and defines the manager
#[inline]
pub fn spin_up(
    num_workers: usize,
) -> (
    JoinHandle<()>,
    Sender<ThreadCommands>,
    Receiver<Option<bool>>,
) {
    // Define the communicate channel
    let (sender, receiver): (Sender<ThreadCommands>, Receiver<ThreadCommands>) = channel();
    let ret_sender = sender.clone();
    let (home_send, home_receive) = channel();
    return (
        thread::spawn(move || {
            // initialize scheduler and workers
            let mut scheduler = Scheduler::init();
            let workers = initialize_workers(num_workers, sender.clone());
            let mut worker_queue: VecDeque<Sender<ThreadCommands>> = VecDeque::new();
            home_send.send(Some(true)).unwrap();
            loop {
                if scheduler.can_kill() && worker_queue.len() == num_workers {
                    break;
                }
                if !worker_queue.is_empty() {
                    if let Some(cmd) = scheduler.get_next() {
                        if let Some(worker) = worker_queue.pop_front() {
                            worker.send(cmd).unwrap();
                        }
                    }
                }

                // Read next instruction
                match receiver.try_recv().unwrap_or(NullType) {
                    // FREE -> A thread has been freed. Either schedule item
                    // or add the worker to the queue
                    FREE(address, recently_computed) => {
                        scheduler.release_item(recently_computed);
                        match scheduler.get_next() {
                            None => worker_queue.push_back(address),
                            Some(cmd) => address.send(cmd).unwrap(),
                        }
                    }
                    // Kill all threads
                    KILL => {
                        println!("Kill;");
                        if scheduler.can_kill() {
                            break;
                        }
                        scheduler.schedule_shutdown();
                    }
                    // If there's nothing to do, lets do some of our work on
                    // task
                    NullType => {}
                    // Schedule a new object for computation
                    ThreadCommands::Calculation(obj) => {
                        let obj_name = obj.get_name();
                        let forge_op = obj.get_op();
                        let forge_op_text = match forge_op {
                            Operation::Add => "add",
                            MatrixMult => "MatMul",
                            Init(_) => "Init",
                        };

                        println!(
                            "Object {} Recieved - Operation: {}",
                            obj_name, forge_op_text
                        );

                        scheduler.schedule(obj);
                    }
                    _ => continue,
                }
            }
            println!("Processes_killed");
            kill_workers(workers, worker_queue);
        }),
        ret_sender,
        home_receive,
    );
}

/*
    Spin up and define worker processes. Will hopefully be inlined to avoid function overhead
*/
#[inline]
fn initialize_workers(
    num_workers: usize,
    manager_address: Sender<ThreadCommands>,
) -> Vec<JoinHandle<()>> {
    let mut workers = Vec::new();
    for i in 0..num_workers {
        // Define chanel for worker
        let (tx, rx): (Sender<ThreadCommands>, Receiver<ThreadCommands>) = channel();
        let manager_address_local = manager_address.clone();
        workers.push(thread::spawn(move || {
            manager_address_local.send(FREE(tx.clone(), 0)).unwrap();
            'worker_thread: loop {
                // Read in new message from manager
                let mut obj_name: u64 = 0;
                match rx.recv().unwrap() {
                    // TODO: Cache Move
                    CacheMove(_) => {}
                    ComputeObject(obj) => {
                        obj_name = obj.get_name();
                        perform_calculation(obj);
                    }
                    KILL => break 'worker_thread,
                    ThreadCommands::PrintMatrix(obj) => unsafe {
                        print_matrix_interface(obj);
                    },
                    _ => {
                        continue 'worker_thread;
                    }
                }

                manager_address_local
                    .send(FREE(tx.clone(), obj_name))
                    .unwrap();
                println!("Thread Free'd - Requesting more work");
            }
            println!("thread {} Killed", i);
        }))
    }
    return workers;
}

/*
*   Cleans up workers
*/
fn kill_workers(workers: Vec<JoinHandle<()>>, worker_queue: VecDeque<Sender<ThreadCommands>>) {
    println!("Workers in queue at death: {}", worker_queue.len());

    for worker in worker_queue {
        worker.send(KILL).unwrap();
    }
    for worker in workers {
        worker.join().unwrap();
    }
}
/*
    Performs calculations
*/
#[inline]
fn perform_calculation(mut target: Object) {
    let forge_op = target.get_op();
    match forge_op {
        Operation::Add => {
            todo!();
        }
        MatrixMult => unsafe {
            let left = target.get_left().unwrap().clone();
            let right = target.get_right().unwrap().clone();
            let new_mat = matrix_mul_interface(left, right);
            target.set_matrix(new_mat);
        },
        Init(mat_type) => match mat_type {
            UniformRandomMatrix => unsafe {
                // println!("Calculation call reached");
                let new_shape = { target.get_shape().clone() };
                let new_matrix = create_uniform_random_mat_interface(&new_shape);
                target.set_matrix(new_matrix);
                let name = target.get_name();
                println!("Successfully initiated {}", name);
            },
            MatrixInitType::ConstantMatrix(_) => {}
            MatrixInitType::DiagonalMatrix(_) => {}
        },
        // PrintMatrix => unsafe {
        //     let mat = match target.lock().unwrap().get_left() {
        //         None => return,
        //         Some(m) => m,
        //     };
        //     print_matrix_interface(mat);
        // },
    }
}
