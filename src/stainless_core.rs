use std::mem::swap;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::classes::MatrixInitType::UniformRandomMatrix;
use crate::classes::Operation::{Init, MatrixMult, PrintMatrix};
use crate::classes::ThreadCommands::*;
use crate::classes::{Operation, ThreadCommands};
use crate::concurent_processes::spin_up;
use crate::object::Object;

pub(crate) struct MultiThread {
    // Manager Thread, controls all other threads. All other threads
    // Live within and are managed by this thread. Holds all tasks and schedules
    // them for executions
    manager: Option<JoinHandle<()>>,
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

    fn matrix_builder(
        &mut self,
        shape: &Vec<u64>,
        left: Option<&Arc<Mutex<Object>>>,
        right: Option<&Arc<Mutex<Object>>>,
        forge_op: Operation,
    ) -> Arc<Mutex<Object>>;

    fn build_uniform_random_matrix(&mut self, shape: &Vec<u64>) -> Arc<Mutex<Object>>;

    fn mat_mul(
        &mut self,
        left: &Arc<Mutex<Object>>,
        right: &Arc<Mutex<Object>>,
    ) -> Arc<Mutex<Object>>;
    fn print_matrix(&mut self, mat: &Arc<Mutex<Object>>) -> Arc<Mutex<Object>>;

    fn kill(&mut self);
}

impl Executor for MultiThread {
    fn init(num_workers: u8) -> Self {
        let (manager, message_box, phone_home) = spin_up(num_workers);
        loop {
            match phone_home.recv().unwrap() {
                None => {}
                Some(_) => break,
            }
        }
        Self {
            manager: Some(manager),
            manager_inbox: message_box,
            name_iter: 0,
        }
    }
    fn matrix_builder(
        &mut self,
        shape: &Vec<u64>,
        left: Option<&Arc<Mutex<Object>>>,
        right: Option<&Arc<Mutex<Object>>>,
        forge_op: Operation,
    ) -> Arc<Mutex<Object>> {
        let handle_deps = |x: Option<&Arc<Mutex<Object>>>| match x {
            None => None,
            Some(item) => Some(Arc::clone(item)),
        };

        let new_obj = Arc::new(Mutex::new(Object::init(
            self.name_iter,
            &shape,
            true,
            forge_op,
            handle_deps(left),
            handle_deps(right),
        )));
        self.manager_inbox
            .send(Calculation(Arc::clone(&new_obj)))
            .unwrap();
        self.name_iter += 1;
        println!("Matrix {} initiated", self.name_iter);

        return new_obj;
    }
    fn build_uniform_random_matrix(&mut self, shape: &Vec<u64>) -> Arc<Mutex<Object>> {
        self.matrix_builder(shape, None, None, Init(UniformRandomMatrix))
    }

    fn mat_mul(
        &mut self,
        left: &Arc<Mutex<Object>>,
        right: &Arc<Mutex<Object>>,
    ) -> Arc<Mutex<Object>> {
        let get_shape = |x: &Arc<Mutex<Object>>| x.lock().unwrap().get_shape().clone();
        let left_shape = get_shape(left);
        let right_shape = get_shape(right);

        if left_shape[1] != right_shape[0] {
            let left_name = left.lock().unwrap().get_name();
            let right_name = right.lock().unwrap().get_name();
            panic!(
                "Error computing product of {} and {} - Shapes did not match",
                left_name, right_name
            );
        }

        let new_shape = vec![left_shape[0], right_shape[1]];
        self.matrix_builder(&new_shape, Some(left), Some(right), MatrixMult)
    }

    fn print_matrix(&mut self, mat: &Arc<Mutex<Object>>) -> Arc<Mutex<Object>> {
        let shape = mat.lock().unwrap().get_shape().clone();
        self.matrix_builder(&shape, Some(mat), None, PrintMatrix)
    }

    fn kill(&mut self) {
        self.manager_inbox.send(KILL).unwrap();
        let mut manager = None;
        swap(&mut self.manager, &mut manager);
        manager.unwrap().join().unwrap();
    }

    // fn build_constant_matrix() {}
    //
    // fn build_diagonal_matrix() {}
}
