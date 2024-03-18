/*
* The scheduler is an internal object that is only of use for the main thread.
* DO NOT use this anywhere else besides the main thread.
*/

/*
* Ideologically, there will be a set of correctness and optimality proofs provided for all decisions
* made here. However, I have 2 days before I have to show this to people,
* and that can be done later :(
*/

use crate::classes::ThreadCommands::{ ComputeObject};
use crate::classes::{LocationMove, ThreadCommands};
use crate::dep_tree::DepTree;
use crate::object::{Object, ObjectInterface};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

pub(crate) struct Scheduler {
    name_lookup: HashMap<u64, Arc<Mutex<DepTree>>>,
    // Used for O(1) lookup of children,
    //preventing circular dependency which would
    // require locking DepTrees behind a mutex
    computation_queue: VecDeque<Object>,
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
            computation_queue: VecDeque::new(),
            location_move_queue: Default::default(),
            terminator: false,
            num_live: 0,
        };
    }

    /*
     * Schedules the object in the computation queue
     */
    pub fn schedule(&mut self, obj: Object) {

        let name = obj.get_name();

        // Queue up for computation
        self.computation_queue.push_back(obj);
        println!("Scheduled {}", name);
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
        if let Some(obj) = self.computation_queue.pop_front() {
            let name = obj.get_name();
            self.num_live -= 1;
            println!("Sending {} for computation", name);
            Some(ComputeObject(obj))
        } else {
            None
        }
    }

    // Takes a dependency tree and schedules the memory movements necessary to
    // compute the respective items
    fn schedule_movements(&self, _item: Arc<Mutex<DepTree>>) -> bool {
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
        // println!("Killable: {}", self.terminator);
        return self.terminator && self.num_live == 0;
    }
}
