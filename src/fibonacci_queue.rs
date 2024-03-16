/*
* Custom Implementation of Fibonacci Heap
* Note: This can (and should) be redone to be totally functional 
*/

use std::cell::RefCell;
use std::cmp::{max};
use std::collections::{HashMap, VecDeque};
use std::mem::swap;
use std::rc::Rc;

pub(crate) trait HeapInterface {
    fn get_key(&self) -> u64;
    fn compare_items(&self, other: &Self) -> bool;

    fn decrease(&mut self);
}

struct HeapNode<T: HeapInterface + Clone> {
    datum: T,
    children: Vec<Rc<RefCell<HeapNode<T>>>>,
    degree: usize,
    mark: bool
}

impl<T: HeapInterface + Clone> HeapNode<T> {
    pub fn push_children(&mut self, node: Rc<RefCell<HeapNode<T>>>) {
        self.children.push(Rc::from(node));
    }

    pub fn decrease(&mut self){
        self.datum.decrease();
    }
}


impl<T: HeapInterface + Clone> HeapNode<T> {
    pub fn init(datum: &T) -> Rc<RefCell<HeapNode<T>>> {
        return Rc::new(RefCell::new(Self {
            datum: datum.clone(),
            children: Vec::new(),
            degree: 0,
            mark: false,
        }))
    }

    pub fn compare(&self, other: &Self) -> bool {
        let other_datum = other.datum.clone();
        self.datum.compare_items(&other_datum)
    }

    pub fn get_children(&self) -> &Vec<Rc<RefCell<HeapNode<T>>>> {
        return &self.children;
    }
}



pub struct FibHeap<T: HeapInterface + Clone> {
    root_list: VecDeque<Rc<RefCell<HeapNode<T>>>>,
    max_deg: usize,
    lookup: HashMap<u64, Rc<RefCell<HeapNode<T>>>>
}

impl<T: HeapInterface + Clone> FibHeap <T> {
    pub fn init() -> Self {
        return Self {
            root_list: VecDeque::new(),
            max_deg: 0,
            lookup: HashMap::with_capacity(20)
        }
    }

    pub fn insert(&mut self, new_item: &T) {
        let node = HeapNode::init(new_item);
        let node_ref = node.clone();
        if  self.root_list.is_empty() || node.borrow().compare(&self.root_list.front().unwrap().borrow()) {
            self.root_list.push_front(node);
        } else {
            self.root_list.pop_back();
        }

        self.lookup.insert(new_item.get_key(), node_ref);
    }

    pub fn extract_min(&mut self) -> Option<T> {
        let datum = {
            let min_item = match self.root_list.pop_back() {
                None => { return None; }
                Some(node) => { node }
            };
            let datum = min_item.borrow().datum.clone();
            let mut children = vec![];
            swap(&mut min_item.borrow_mut().children, &mut children);

            while !children.is_empty() {
                let item = children.pop().unwrap();
                self.root_list.push_back(item);
            }
            self.consolidate();
            self.lookup.remove(&min_item.borrow().datum.get_key());
            datum
        };

        return Some(datum);
    }

    fn consolidate(&mut self) {
        let max_degree: usize = self.max_deg;

        let mut a: Vec<Option<Rc<RefCell<HeapNode<T>>>>> = Vec::new();
        for _i in 0..max_degree {
            a.push(None);
        }
        while let Some(mut node) = self.root_list.pop_front() {
            let mut d = node.borrow().degree;
            while let Some(mut other) = a[d].clone() {
                if other.borrow().compare(&node.borrow()) {
                    swap(&mut node, &mut other);
                }
                self.heap_link(&mut other, &mut node);
                a[d] = None;
                d = d + 1;
            }
            a[d] = Some(node.clone());
        }
        'reinsert_items: while let Some(node) = a.pop() {
            let node = match node {
                None => {continue 'reinsert_items }
                Some(i) => {i}
            };

            if !self.root_list.is_empty() || node.borrow().compare(&self.root_list.front().unwrap().borrow()) {
                self.root_list.push_front(node);
            } else {
                self.root_list.push_back(node);
            }
        }
    }


    fn heap_link(&mut self, y: &mut Rc<RefCell<HeapNode<T>>>, x: &Rc<RefCell<HeapNode<T>>>) {
        let mut popped_ref = {
            let back = self.root_list.pop_back();
            match back {
                None => {return;}
                Some(back) => {back}
            }
        };
        swap(y, &mut popped_ref);
        x.borrow_mut().degree += 1;
        self.max_deg = max(x.borrow().degree, self.max_deg);
        popped_ref.borrow_mut().mark = false;
        x.borrow_mut().push_children(popped_ref);
    }

    pub fn decrease_key(&self, k: u64) {
        let node: Rc<RefCell<HeapNode<T>>> = match self.lookup.get(&k) {
            None => {return;}
            Some(i) => {i.clone() }
        };
        node.borrow_mut().decrease();

        for other in &node.borrow_mut().children {
            if other.borrow().compare(&node.borrow()) {
                swap(&mut node.borrow_mut().datum, &mut other.borrow_mut().datum)
            }
        }
    }

    pub fn get_min(&self) -> Option<T>
    {
        match self.root_list.front() {
            None => {None}
            Some(node) => {Some(node.borrow().datum.clone())}
        }
    }
}

