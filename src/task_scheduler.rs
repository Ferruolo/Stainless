/*
* The scheduler is an internal object that is only of use for the main thread.
* DO NOT use this anywhere else besides the main thread.
*/

use std::cmp::max;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use crate::classes::ThreadCommands::{CacheMove, ComputeObject};
use crate::classes::{LocationMove, ThreadCommands};
use crate::dep_tree::DepTree;
use crate::object::Object;

pub(crate) struct Scheduler {
    name_lookup: HashMap<u64, Rc<DepTree>>,
    computation_queue: VecDeque<Vec<Rc<DepTree>>>,
    location_move_queue: VecDeque<LocationMove>,
    terminator: bool,
    num_live: u8,
    height_adjust: usize,
}

impl Scheduler {
    pub fn init() -> Self {
        return Self {
            name_lookup: HashMap::with_capacity(20),
            computation_queue: VecDeque::from([Vec::new()]),
            location_move_queue: Default::default(),
            terminator: false,
            num_live: 0,
            height_adjust: 0,
        };
    }

    pub fn schedule(&mut self, obj: Arc<Mutex<Object>>) {
        let new_dep_tree = DepTree::init(obj, &mut self.name_lookup);
        self.name_lookup
            .insert(new_dep_tree.get_name(), Rc::clone(&new_dep_tree));
        self.adjust_height(new_dep_tree.get_height());

        self.reorder_queue(new_dep_tree.get_children());
        self.computation_queue[new_dep_tree.get_height()].push(new_dep_tree);

        self.num_live += 1;
    }

    pub fn get_next(&mut self) -> Option<ThreadCommands> {
        if let Some(lov_move) = self.location_move_queue.pop_back() {
            return Some(CacheMove(lov_move));
        } else {
            return self.fetch_next_item();
        }
    }

    fn fetch_next_item(&mut self) -> Option<Rc<DepTree>> {
        let item = match self.computation_queue.back().unwrap().pop()  {
            None => {return None}
            Some(item) => {
                item
            }
        };
        if self.schedule_movements(&item) {
            self.computation_queue.back().unwrap().push(item);
            return None;
        } else {
            if self.computation_queue.is_empty() {
                self.computation_queue.pop_back();
            }
            return Some(item);
        }
    }

    fn schedule_movements(&self, item: &Rc<DepTree>) -> bool {
        return false;
    }

    fn adjust_height(&mut self, height: usize) {
        // Amortizes to O(1) add
        let cur_height = max(height - self.height_adjust, self.computation_queue.len());
        if cur_height > self.computation_queue.len() {
            for _ in 0..cur_height {
                self.computation_queue.push_front(Vec::new())
            }
        }
    }

    pub fn terminate() {
        todo!();
    }

    pub fn schedule_shutdown(&mut self) {
        self.terminator = true;
    }

    pub fn can_kill(&self) -> bool {
        return self.terminator && self.num_live == 0;
    }

    fn reorder_queue(&mut self, items: &Vec<Rc<DepTree>>) {
        for item in items {
            let height = item.get_height();
        }
    }

    fn sort_inplace(
        &mut self,
        to_sort: &Vec<Rc<DepTree>>,
        low: usize,
        high: usize,
    ) -> (usize, usize) {
        if low == high {
            let num_deps = to_sort[low].get_num_dependencies();

            return (num_deps, num_deps);
        } else {
            let partition = (low - high) / 2;
            let (l_low, l_high) = self.sort_inplace(to_sort, low, partition);
            let (r_low, r_high) = self.sort_inplace(to_sort, partition, high);
            if l_high < r_low {
                return (l_low, r_high);
            }
            // TODO: Finish implementing this at some point using MergeSort

            return (0, 0);
        }
    }
}
