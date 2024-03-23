# Stainless - Efficient Machine Learning Computation in Rust
## What even is Stainless?
Stainless aims to provide a highly-abstracted but easily customizable framework 
for Machine Learning in Rust. Ideally, Stainless will allow you to avoid having to think about 
memory management at all. However, if you choose to, you can easily add new functionality, or change
how memory is managed. Stainless is designed for performance computing,
and aims to be the tool I wish I had when I started learning Machine Learning. 


## Components:

### Stainless Core:
This is where the core of stainless lives, and the interface. To interact, 
you can call it like so

```rust
    let mut exec = MultiThread::init(1);
    let shape = vec![32, 32];
    let a = exec.build_uniform_random_matrix(&shape);
    let b = exec.build_uniform_random_matrix(&shape);
    let prod = exec.mat_mul(&a, &b);
    exec.print_matrix(&a);
    exec.print_matrix(&b);
    exec.print_matrix(&prod);
    exec.kill();
```
when interface funcitons are called, they send Messages of type ThreadCommands 
(usually Calculation) to the manager thread


### Manager Thread:
This is the CEO of Stainless and lives in concurent_proccesses/spin_up .
It manages all uncompleted items, sorts them in the
optimal execution order, and manages where they're stored. 
It begins by scheduling the next availible command, then reads the next message in 
it's inbox, wheather that be "FREE" (thread has been freed),
"Calculation" (schedule this item for calculation)

#### task_scheduler.rs
This is where the tasks are stored before they are sent for computation,
it manages the order in which they're computed, and makes sure that A) All 
dependencies are calculated, B) all necessary items are on the 
right processor on the right machine.

#### DepTree
This is where info on Objects is stored, including their dependencies, the number of
dependencies still being computed, and how important they are to 
items that will be computed after them. 

### Objects
Stainless Matricies, and other items, are reffered to as Objects. Objects are
calculated from a set of other objects, or are loaded in, or are calculated manually.
They only interact with other objects.


## Classes.rs
This is where we define all Enums that we use. Seriously, all of them are defined in
this file and (theoretically) well documented.

## Priority Heap
My own implementation of a Priority Heap, because the provided one didn't have 
work the way I needed it to.
