use std::{
    f32::consts::PI,
    ops::{Add, Mul},
};

use n_renderer::{
    object::{Face, Node},
    pos::Pos3D,
    render::Color,
    transform::Transform,
};

use crate::orbital::{complex::AbsArg, lookup};

/// Function to run the marching cubes algorithm for a single cube and add the resulting nodes edges and faces to the object, transform the vectors that were input
pub fn run_marching_cubes<
    T: AbsArg<Output = f32> + Copy + Mul<f32, Output = T> + Add<Output = T>,
>(
    func: impl Fn(Pos3D) -> T,
    limits: (usize, usize, usize),
    cutoff: impl Fn(T) -> bool,
) -> (Vec<Node<Pos3D>>, Vec<Face>) {
    let mut nodes = Vec::with_capacity(limits.0 * limits.1 * limits.2 * 12);
    let mut faces = Vec::with_capacity(limits.0 * limits.1 * limits.2 * 3);

    for i in -(limits.0 as isize)..=limits.0 as isize {
        for j in -(limits.1 as isize)..=limits.1 as isize {
            for k in -(limits.2 as isize)..=limits.2 as isize {
                // Rewrite the function such that it can be evaluated at the points of interest
                let offset = Pos3D {
                    x: i as f32,
                    y: j as f32,
                    z: k as f32,
                };
                let local_func = |pos: Pos3D| func(pos + offset);

                // Get the new nodes from the marching cubes algoritm
                let (mut new_nodes, mut new_faces) = marching_cubes(&local_func, &cutoff);

                // Get the current node index, so the edges and faces can be updated and properly appended
                let node_index = nodes.len();

                // Update the indices to match the new node indices
                new_faces.iter_mut().for_each(|face| {
                    face.node_a_index += node_index;
                    face.node_b_index += node_index;
                    face.node_c_index += node_index;
                });

                // Translate the nodes to their proper position
                new_nodes.iter_mut().for_each(|node: &mut Node<Pos3D>| {
                    *node = node.translate(offset);
                });

                // Append the nodes, edges and faces with the shifted indices to the total
                nodes.append(&mut new_nodes);
                faces.append(&mut new_faces);
            }
        }
    }
    (nodes, faces)
}

fn marching_cubes<T: AbsArg<Output = f32> + Copy + Mul<f32, Output = T> + Add<Output = T>>(
    func: &impl Fn(Pos3D) -> T,
    cutoff: &impl Fn(T) -> bool,
) -> (Vec<Node<Pos3D>>, Vec<Face>) {
    let mut nodes: Vec<Node<Pos3D>> = Vec::new();
    let mut faces: Vec<Face> = Vec::new();

    /*
    Calculate the position of the vertices and edges
    0, 000, (i      , j     , k     ), 0b0000_0001
    1, 001, (i + 1  , j     , k     ), 0b0000_0010
    2, 010, (i      , j + 1 , k     ), 0b0000_0100
    3, 011, (i + 1  , j + 1 , k     ), 0b0000_1000
    4, 100, (i      , j     , k + 1 ), 0b0001_0000
    5, 101, (i + 1  , j     , k + 1 ), 0b0010_0000
    6, 110, (i      , j + 1 , k + 1 ), 0b0100_0000
    7, 111, (i + 1  , j + 1 , k + 1 ), 0b1000_0000
    */

    // Test what corners are in and outside the shape and save this as a byte
    let mut byte: u8 = 0x0;
    for index in 0..8 {
        let pos = Pos3D {
            x: ((index >> 0) & 0x1) as f32,
            y: ((index >> 1) & 0x1) as f32,
            z: ((index >> 2) & 0x1) as f32,
        };
        byte ^= (cutoff(func(pos)) as u8) << index;
    }

    // Early return if all corners are inside or outside of the function
    if byte == 0x00 || byte == 0xff {
        return (nodes, faces);
    }

    // Look up the edges of the faces that are present in this situation
    let face_edge_indices = lookup::triangle_table(byte as usize);

    // Iterate over the faces for the current cube
    for face_edge_index in face_edge_indices.chunks(3) {
        // Get the positions of the vertices of the faces
        let face_vertices_values = face_edge_index
            .iter()
            .map(|edge| edge_to_boundary_vertex(func, *edge, cutoff))
            .collect::<Vec<(Pos3D, T)>>();

        // Generate a new face
        let node_index_offset = nodes.len();
        faces.push(Face {
            node_a_index: node_index_offset,
            node_b_index: node_index_offset + 1,
            node_c_index: node_index_offset + 2,
            r: 15,
        });

        // Generate the new nodes
        face_vertices_values
            .iter()
            .for_each(|&(vertex, value)| {
                nodes.push(Node {
                    pos: vertex,
                    r: 0,
                    color: {
                        Color::HSV((value.arg() / PI * 180.0 + 180.0) as u16, (0.8 * 256.0) as u8, (0.7 * 256.0) as u8)
                    },
                })
            });
    }

    // Move the position of the vertex to the cutoff point
    fn edge_to_boundary_vertex<
        T: Copy + AbsArg<Output = f32> + Mul<f32, Output = T> + Add<Output = T>,
    >(
        func: &impl Fn(Pos3D) -> T,
        edge_index: usize,
        cutoff: impl Fn(T) -> bool,
    ) -> (Pos3D, T) {
        // Get the indices of the vertices on the cube connected to the current edge
        let [vertex_0_index, vertex_1_index] = lookup::EDGE_VERTEX_INDICES[edge_index];

        // Calculate the position of the two vertices at the ends of the edge
        let vertex_0_pos = lookup::VERTEX_RELATIVE_POSITION[vertex_0_index];
        let vertex_1_pos = lookup::VERTEX_RELATIVE_POSITION[vertex_1_index];

        // Calculate the relative location of the new vertex along the edge
        let parameter = interpolate(vertex_0_pos, vertex_1_pos, &func, &cutoff, 3);

        // Calculate the actual position and value of the new vertex
        let interpolated_pos = vertex_0_pos + (vertex_1_pos - vertex_0_pos) * parameter;
        let interpolated_value = func(interpolated_pos);

        (interpolated_pos, interpolated_value)
    }

    /// Iteratively approximate the cutoff along the edge
    fn interpolate<T>(
        start: Pos3D,
        end: Pos3D,
        func: &impl Fn(Pos3D) -> T,
        cutoff: &impl Fn(T) -> bool,
        depth: usize,
    ) -> f32 {
        if depth == 0 {
            return 0.5;
        }

        // Test if the edge endpoints fall inside or outside
        let start_in = cutoff(func(start));
        let end_in = cutoff(func(end));

        // Determine the midpoint between the edge points and test if it falls inside or outside
        let midpoint = start + (end - start) * 0.5;
        let midpoint_in = cutoff(func(midpoint));

        // If the start is inside and the midpoint is outside or the other way around:
        // -> The new midpoint must be between the start and midpoint
        if start_in ^ midpoint_in {
            0.5 * interpolate(start, midpoint, func, cutoff, depth - 1)
        } else if end_in ^ midpoint_in {
            0.5 + 0.5 * interpolate(midpoint, end, func, cutoff, depth - 1)
        } else {
            panic!("I don't know what is happening here");
        }
    }

    (nodes, faces)
}
