
pub mod classes;
mod stainless_core;
mod object;
mod task_scheduler;
mod bindings;
mod binding_interface;
mod dep_tree;
mod optimizations;
mod fibonacci_queue;

use crate::stainless_core::{Executor, MultiThread};

fn main() {
    let mut exec = MultiThread::init(2);
    // exec.spin_up_threads(2);
    let shape = vec![2, 2];
    let a = exec.build_uniform_random_matrix(&shape);
    let b = exec.build_uniform_random_matrix(&shape);
    let c = exec.build_uniform_random_matrix(&shape);
    //

    let prod = exec.mat_mul(&a, &b);
    exec.kill();
    println!("Success!");
}

