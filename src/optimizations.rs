/*
*  Suite of functions which use pattern matching to optimize computation order.
*  Needs to be integrated in when I determine how
*  DISREGARD
*/

// fn get_pattern_binop(
//     tree: &Arc<Mutex<DepTree>>,
//     height: usize,
//     index: usize,
// ) -> (ComputationGraph, Vec<Arc<Mutex<DepTree>>>) {
//     if height == 0 {
//         return (M(index), vec![Arc::clone(tree)]);
//     } else {
//         let (l_graph, mut l_vec) =
//             get_pattern_binop(&tree.lock().unwrap().get_children()[0], height - 1, index);
//         let (r_graph, mut r_vec) = get_pattern_binop(
//             &tree.lock().unwrap().get_children()[1],
//             height - 1,
//             index + 2,
//         );
//         l_vec.append(&mut r_vec);
// 
//         return (Op(Box::new(l_graph), Box::new(r_graph)), l_vec);
//     }
// }
// 
// /*
// *
// * Optimization Library for task_queue used for optimizing task order
// *
// */
// 
// fn find_best_pattern(
//     children: &Vec<Arc<Mutex<DepTree>>>,
//     forge_op: Operation,
//     i: usize,
//     j: usize,
// ) -> (ComputationGraph, u64, Vec<u64>) {
//     match forge_op {
//         // No Need to memoize because we don't go very deep
//         MatMul => {
//             if i == j {
//                 let shape = children[i].lock().unwrap().get_shape().clone();
//                 (M(i), shape[0] * shape[1], shape)
//             } else {
//                 let i_shape = children[i].lock().unwrap().get_shape()[0];
//                 let j_shape = children[i].lock().unwrap().get_shape()[1];
//                 let (l_pattern, mut l_runtime, l_shape) =
//                     find_best_pattern(children, forge_op, i, j - 1);
//                 let (r_pattern, mut r_runtime, r_shape) =
//                     find_best_pattern(children, forge_op, i + 1, j);
//                 l_runtime += l_shape[0] * l_shape[1] * j_shape;
//                 r_runtime += i_shape * r_shape[0] * r_shape[1];
//                 if l_runtime <= r_runtime {
//                     let new_shape = vec![l_shape[0], j_shape];
//                     (
//                         Op(Box::new(l_pattern), Box::new(M(j))),
//                         l_runtime,
//                         new_shape,
//                     )
//                 } else {
//                     let new_shape = vec![l_shape[0], j_shape];
//                     (
//                         Op(Box::new(l_pattern), Box::new(M(j))),
//                         r_runtime,
//                         new_shape,
//                     )
//                 }
//             }
//         }
//         _ => (NoPattern, 0, vec![]),
//     }
// }
// 
// fn match_ops(pattern: ComputationGraph, tree: &Arc<Mutex<DepTree>>, forge_op: Operation) -> bool {
//     match pattern {
//         Op(l, r) => {
//             let tree_op = tree.lock().unwrap().get_forge_op();
//             let children = tree.lock().unwrap().get_children();
//             return tree_op == forge_op
//                 && match_ops(*l, &children[0], forge_op)
//                 && match_ops(*r, &children[1], forge_op);
//         }
//         M(_) => true,
//         ComputationGraph::NoPattern => false,
//     }
// }
// 
// fn pattern_equal(left: &ComputationGraph, right: &ComputationGraph) -> bool {
//     match (left, right) {
//         (Op(l1, l2), Op(r1, r2)) => pattern_equal(l1, r1) && pattern_equal(l2, r2),
//         (M(l), M(r)) => l == r,
//         (_, _) => false,
//     }
// }
// 
// fn calc_runtime(
//     pattern: &ComputationGraph,
//     children: &Vec<Arc<Mutex<DepTree>>>,
//     forge_op: Operation,
// ) -> (u64, u64) {
//     match forge_op {
//         Add => (0, 0),
//         MatMul => match pattern {
//             M(i) => {
//                 let shape = children[i.clone()].lock().unwrap().get_shape().clone();
//                 (shape[0] * shape[1], shape[1])
//             }
//             Op(l, r) => {
//                 let (l_rt, l_rightdim) = calc_runtime(&*l, children, forge_op);
//                 let (r_rt, r_right_dim) = calc_runtime(&*r, children, forge_op);
//                 return ((l_rt * r_rt) / l_rightdim, r_right_dim);
//             }
//             NoPattern => (0, 0),
//         },
//         Init => (0, 0),
//     }
// }
// 
// fn build_new_tree(
//     graph: &ComputationGraph,
//     children: Vec<Arc<Mutex<DepTree>>>,
//     forge_op: Operation,
//     name: u64,
//     loc: ItemLoc,
// ) -> Arc<Mutex<DepTree>> {
//     match graph {
//         Op(l, r) => {
//             let left_tree = build_new_tree(l, children.clone(), forge_op, name + 1, loc);
//             let right_tree = build_new_tree(r, children.clone(), forge_op, name + 1, loc);
// 
//             let l_shape = left_tree.lock().unwrap().get_shape().clone();
//             let r_shape = left_tree.lock().unwrap().get_shape().clone();
//             let new_shape = vec![l_shape[0], r_shape[0]];
// 
//             let l_node = left_tree.lock().unwrap().get_node();
//             let r_node = right_tree.lock().unwrap().get_node();
// 
//             return Arc::new(Mutex::new(DepTree::init(Arc::new(Mutex::new(
//                 Object::init(
//                     name,
//                     &new_shape,
//                     false,
//                     forge_op,
//                     Some(l_node),
//                     Some(r_node),
//                 ),
//             )))));
//         }
// 
//         M(idx) => {
//             return Arc::clone(&children[0]);
//         }
//         NoPattern => {
//             panic!("Built tree with no pattern")
//         }
//     }
// }



// fn optimize(dep_tree: Arc<Mutex<DepTree>>) -> Arc<Mutex<DepTree>> {
//     {
//         let d = dep_tree.lock().unwrap().get_height();
//         if d < 2 {
//             return dep_tree;
//         }
//     }
//     let forge_op = dep_tree.lock().unwrap().get_forge_op();
//     match forge_op {
//         Add => {
//             //TODO: If left forge_op = matmul return a GEMM
//             return dep_tree;
//         }
//         MatMul => {
//             let (pattern, children) = get_pattern_binop(&dep_tree, 2, 0);
//             let (best_pattern, runtime, shape) = find_best_pattern(&children, forge_op, 0, 3);
//             let (cur_runtime, _) = calc_runtime(&pattern, &children, forge_op);
//             if cur_runtime <= runtime {
//                 return dep_tree;
//             }
//             let name = dep_tree.lock().unwrap().get_name();
//             let loc = dep_tree.lock().unwrap().get_loc();
//             dep_tree.lock().unwrap().erase();
//             build_new_tree(&pattern, children, forge_op, name, loc)
//         }
//         Init => {
//             return dep_tree;
//         }
//     }
// }

