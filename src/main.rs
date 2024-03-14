
mod classes;
mod stainless_core;
mod object;
mod task_scheduler;
mod bindings;
mod binding_interface;
mod dep_tree;
mod optimizations;
mod fibonacci_queue;

use crate::stainless_core::MultiThread;


fn main() {
    let mut exec = MultiThread::init(2);
    // exec.spin_up_threads(2);
    let shape = vec![2, 2];
    let a = exec.create_uniform_random_matrix(&shape);
    let b = exec.create_uniform_random_matrix(&shape);
    let c = exec.create_uniform_random_matrix(&shape);
    //
    let sum = exec.add(&a, &b);
    let sum2 = exec.add(&a, &b);
    let prod = exec.mat_mul(&a, &b);
    exec.print_graph();
    exec.kill();
    println!("Success!");
}

