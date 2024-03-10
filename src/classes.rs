use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use crate::object::Object;
use crate::bindings::Matrix;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Operation {
    Add,
    MatMul,
    Init
}

#[derive(Clone, Copy)]
pub(crate) enum ItemLoc {
    GPU, CPU, DISK
}
pub enum ComputationGraph {
    Op(Box<ComputationGraph>, Box<ComputationGraph>),
    M(usize),
    NoPattern,
}

pub enum Executable {
    GpuMatrix(*mut Matrix),
    CpuMatrix,
    DiskItem,
    None
}


#[derive(Clone)]
pub(crate) enum LocationMove {
    Gpu2Gpu,
    Gpu2Cpu,
    Cpu2Gpu,
    Cpu2Cpu,
    Cpu2Disk,
    Disk2Cpu,
    Disk2Disk
}



pub(crate) enum ThreadCommands {
    FREE(Sender<ThreadCommands>),
    CacheMove(LocationMove),
    Calculation(Arc<Mutex<Object>>),
    ComputeObject(Arc<Mutex<Object>>),
    KILL,
    NullType
}
