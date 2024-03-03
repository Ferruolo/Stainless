
mod classes;
mod stainless_core;
mod array;
mod tast_queue;

mod bindings;
mod binding_interface;

use crate::stainless_core::Executor;


fn main() {
    let mut exec = Executor::init(2);
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

