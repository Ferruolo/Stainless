use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::thread::JoinHandle;

use crate::classes::ThreadCommands;
use crate::classes::ThreadCommands::*;
use crate::task_scheduler::Scheduler;

pub(crate) struct Executor {
    manager: Rc<JoinHandle<()>>,
    manager_inbox: Sender<ThreadCommands>,
    name_iter: u64,
}

impl Executor {
    pub(crate) fn init(num_workers: u8) -> Self {
        let (manager, message_box) = spin_up(num_workers);
        Self {
            manager: Rc::new(manager),
            manager_inbox: message_box,
            name_iter: 0,
        }
    }
}
#[inline]
fn spin_up(num_workers: u8) -> (JoinHandle<()>, Sender<ThreadCommands>) {
    let (sender, receiver): (Sender<ThreadCommands>, Receiver<ThreadCommands>) = mpsc::channel();
    return (
        thread::spawn(move || {
            let mut scheduler = Scheduler::init();
            let workers = initialize_workers(num_workers, &sender);
            let mut worker_queue: VecDeque<Sender<ThreadCommands>> = VecDeque::new();
            loop {
                // Fetch Items from mailbox
                match receiver.try_recv().unwrap_or(NullType) {
                    FREE(address) => match scheduler.get_next() {
                        None => worker_queue.push_back(address),
                        Some(cmd) => address.send(cmd).unwrap(),
                    },
                    KILL => {
                        scheduler.schedule_shutdown();
                    }
                    NullType => {
                        if let Some(address) = worker_queue.pop_front() {
                            if let Some(task) = scheduler.get_next() {
                                address.send(task).unwrap()
                            } else {
                                worker_queue.push_front(address);
                            }
                        }
                        if scheduler.can_kill() {
                            break;
                        }
                    }
                    ComputeObject(obj) => {
                        scheduler.schedule(obj);
                    }
                    _ => continue,
                }
            }
            kill_workers(workers);
        }),
        sender,
    );
}
#[inline]
fn initialize_workers(
    num_workers: u8,
    manager_address: &Sender<ThreadCommands>,
) -> Vec<JoinHandle<()>> {
    let mut workers = Vec::new();
    for i in 0..num_workers {
        let (tx, rx): (Sender<ThreadCommands>, Receiver<ThreadCommands>) = mpsc::channel();
        workers.push(thread::spawn(move || loop {
            match rx.recv().unwrap() {
                CacheMove(_) => {}
                Calculation(obj) => {
                    todo!();
                }
                KILL => break,
                _ => {
                    continue;
                }
            }
        }))
    }
    return workers;
}

fn kill_workers(workers: Vec<JoinHandle<()>>) {
    for worker in workers {
        worker.join().unwrap();
    }
}
