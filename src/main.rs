
mod classes;
mod task_queue;
mod stainless_core;
mod array;

use crate::stainless_core::Executor;


fn main() {
    let mut exec = Executor::init();
    let shape = vec![2, 2];
    let a = exec.build_matrix(&shape);
    let b = exec.build_matrix(&shape);
    let c = exec.build_matrix(&shape);
    //
    // let sum = exec.add(a, b);
    // exec.print_graph();
    println!("Success!");
}

