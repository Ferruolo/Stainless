// use crate::matrices;
// use crate::matrices::{Matrix, MatrixAdd};
// use crate::stainless_core::Operation::Add;
//
// enum Operation {
//     Add,
//     Subtract,
//     MatMul,
//     ElementWiseMul,
//     Inverse,
//     Transpose
// }
//
// struct Tensor<'a, 'b> {
//     dimensions: Vec<u8>,
//     size: u8,
//     mat: Matrix,
//     forge_operation: Operation,
//     left: &'a Tensor<'a, 'a>,
//     right: &'b Tensor<'b, 'b>
// }
//
// trait Executor {
//     fn init();
//     fn add<'a, 'b>(left: &'a Tensor, right: &'b Tensor) -> Tensor<'a, 'b>;
//     // fn subtract(left: &Tensor, right: &Tensor, target: &Tensor);
//     // fn mat_mul(left: &Tensor, right: &Tensor, target: &Tensor);
//     // fn elementwise_mul(left: &Tensor, right: &Tensor, target: &Tensor);
//     // fn inverse(left: &Tensor, right: &Tensor, target: &Tensor);
//     // fn transpose(left: &Tensor, right: &Tensor, target: &Tensor);
//
//     // TODO: add more elements to from simply typed lambda calculus?
// }
//
//
// // Define Language
//
// struct SingleCore {
//     queue: Vec<fn () -> ()>
// }
//
// impl Executor for SingleCore { // For Single Core,
//     fn init() {
//         todo!()
//     }
//
//     fn add<'a, 'b>(left: &'a Tensor, right: &'b Tensor) -> Tensor<'a, 'b> {
//         let a = &left.mat;
//         let b = &right.mat;
//         match (a, b) {
//             (Matrix::CpuMatrix(l), Matrix::CpuMatrix(r)) =>{
//                 Tensor {
//                     dimensions: left.dimensions.clone(),
//                     size: left.size,
//                     mat: unsafe {Matrix::CpuMatrix(MatrixAdd(*l, *r))},
//                     forge_operation: Operation::Add,
//                     left,
//                     right,
//                 }
//             }
//             ((_, _)) => panic!("TypeError")
//         }
//     }
//
//     // fn subtract(left: &Tensor, right: &Tensor, target: &Tensor) {
//     //     todo!()
//     // }
//     //
//     // fn mat_mul(left: &Tensor, right: &Tensor, target: &Tensor) {
//     //     todo!()
//     // }
//     //
//     // fn elementwise_mul(left: &Tensor, right: &Tensor, target: &Tensor) {
//     //     todo!()
//     // }
//     //
//     // fn inverse(left: &Tensor, right: &Tensor, target: &Tensor) {
//     //     todo!()
//     // }
//     //
//     // fn transpose(left: &Tensor, right: &Tensor, target: &Tensor) {
//     //     todo!()
//     // }
// }
//
// fn build_tensor(shape: &Vec<u64>, elements: &Vec<u64>) {
//     let elments =
// }
