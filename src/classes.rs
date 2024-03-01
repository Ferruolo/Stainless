
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operations {
    Add,
    MatMul,
    Init
}

pub(crate) enum ItemLoc {
    GPU, CPU, DISK
}

// #[derive(Clone)]
// pub(crate) enum CacheMove<'a> {
//     Gpu2Gpu(&'a Object),
//     Gpu2Cpu(&'a Object),
//     Cpu2Gpu(&'a Object),
//     Cpu2Cpu(&'a Object),
//     Cpu2Disk(&'a Object),
//     Disk2Cpu(&'a Object),
//     Disk2Disk(&'a Object)
// }

// pub(crate) enum Instruct<'a> {
//     CacheMove(CacheMove<'a>),
//     Calculation(&'a Object),
//     None
// }

