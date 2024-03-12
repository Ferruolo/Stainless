/*
* The scheduler is an internal object that is only of use for the main thread.
* DO NOT use this anywhere else besides the main thread.
*/

/*
* Ideologically, there will be a set of correctness and optimality proofs provided for all decisions
* made here. However, I have 6 days before I have to show this to people,
* and that can be done later :(
*/

use crate::classes::ThreadCommands::{CacheMove, Calculation};
use crate::classes::{LocationMove, ThreadCommands};
use crate::dep_tree::DepTree;
use crate::object::Object;
use crate::FibonacciQueue::FibonacciHeap;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub(crate) struct Scheduler {
    name_lookup: HashMap<u64, Rc<DepTree>>,
    // Used for O(1) lookup of children,
    //preventing circular dependency which would
    // require locking DepTrees behind a mutex
    computation_queue: FibonacciHeap<Rc<DepTree>>,
    // Queue of items to be computed. Designed so items with lowest height and lowest number
    // of direct dependencies are computed first.
    // TODO: Write correctness and optimality proofs for this
    location_move_queue: VecDeque<LocationMove>,
    // Queue of item location moves.
    // Currently not implemented,
    // will eventually support all moves listed in classes/LocationMove
    terminator: bool,
    // If a system kill is scheduled or not
    num_live: u8,
    // Number of live elements,
    // prevents pre-mature kills
}

impl Scheduler {
    pub fn init() -> Self {
        // Initialize Elements
        return Self {
            name_lookup: HashMap::with_capacity(20),
            computation_queue: FibonacciHeap::new(),
            location_move_queue: Default::default(),
            terminator: false,
            num_live: 0,
        };
    }

    /*
     * Schedules the object in the computation queue
     */
    pub fn schedule(&mut self, obj: Arc<Mutex<Object>>) {
        // Create new depedency tree for node
        let new_dep_tree = DepTree::init(obj, &mut self.name_lookup);
        //Add new tree into lookup table so that children can be added later
        self.name_lookup
            .insert(new_dep_tree.get_name(), Rc::clone(&new_dep_tree));
        // increment number of dependencies all children to re-establish invariant
        for item in new_dep_tree.get_children() {
            self.computation_queue.add_dependencies(item);
        }
        // Queue up for computation
        self.computation_queue.insert(new_dep_tree);

        // Increment number of live elements
        self.num_live += 1;
    }

    /*
        Retrieves next item to be executed, ensures that items are properly moved before
        execution to prevent any errors
    */

    pub fn get_next(&mut self) -> Option<ThreadCommands> {
        // Make sure all cache movements are scheduled. If everything is in the
        // right place for the min item, then nothing will be scheduled here
        self.schedule_movements(self.computation_queue.minimum().unwrap().clone());

        // Make sure all cache movements are executed before the next item is taken
        // off the queue. If all cache movements are completed, schedule the computation
        return if let Some(next) = self.location_move_queue.pop_front() {
            Some(CacheMove(next))
        } else if let Some(mut next) = self.computation_queue.extract_min() {
            next.kill_children();
            self.num_live -= 1;
            // Remove all children from the tree
            // ensures we don't have hanging memory
            Some(Calculation(next.get_node()))
 
        } else {
            None
        };
    }

    // Takes a dependency tree and schedules the memory movements necessary to
    // compute the respective items
    fn schedule_movements(&self, item: Rc<DepTree>) -> bool {
        return false;
    }


    /*
        Schedules shutdown by setting terminator to true.
        Process will shut down there are no more live elements
    */
    pub fn schedule_shutdown(&mut self) {
        self.terminator = true;
    }

    /*
        Returns true if process can be killed 
    */
    pub fn can_kill(&self) -> bool {
        return self.terminator && self.num_live == 0;
    }
}
