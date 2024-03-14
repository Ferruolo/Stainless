use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use crate::object::Object;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) enum MatrixInitType {
    UniformRandomMatrix,
    ConstantMatrix(u32),
    DiagonalMatrix(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Operation {
    Add,
    MatrixMult,
    Init(MatrixInitType),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ItemLoc {
    GPU,
    CPU,
    DISK,
}
pub enum ComputationGraph {
    Op(Box<ComputationGraph>, Box<ComputationGraph>),
    M(usize),
    NoPattern,
}


#[derive(Clone)]
pub(crate) enum LocationMove {
    Gpu2Gpu,
    Gpu2Cpu,
    Cpu2Gpu,
    Cpu2Cpu,
    Cpu2Disk,
    Disk2Cpu,
    Disk2Disk,
    Machine2Machine,
}

pub(crate) enum ThreadCommands {
    FREE(Sender<ThreadCommands>),
    CacheMove(LocationMove),
    Calculation(Arc<Mutex<Object>>),
    ComputeObject(Arc<Mutex<Object>>),
    KILL,
    NullType,
}

