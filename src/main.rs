
pub mod classes;
mod stainless_core;
mod object;
mod task_scheduler;
mod bindings;
mod binding_interface;
mod dep_tree;
mod fibonacci_queue;
mod concurent_processes;

use crate::stainless_core::{Executor, MultiThread};

fn main() {
    let mut exec = MultiThread::init(1);
    // exec.spin_up_threads(2);
    let shape = vec![32, 32];
    let a = exec.build_uniform_random_matrix(&shape);
    let b = exec.build_uniform_random_matrix(&shape);
    let prod = exec.mat_mul(&a, &b);
    exec.print_matrix(&a);
    exec.print_matrix(&b);
    exec.print_matrix(&prod);
    exec.kill();
    println!("Success!");
}

