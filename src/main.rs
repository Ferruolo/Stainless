use crate::bindings::{hello, MatrixAdd, printMatrix};
use crate::matrices::create_ones_matrix_rust;
mod bindings;
mod matrices;
fn main() {
    let shape1 = vec![2, 2];
    let shape2 = vec![2, 2];
    let m1 = create_ones_matrix_rust(2, shape1);
    let m2 = create_ones_matrix_rust(2, shape2);
    let m3 = unsafe { MatrixAdd(m1, m2) };
    unsafe {printMatrix(m3)};
}

