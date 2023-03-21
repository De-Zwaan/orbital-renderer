use std::{collections::HashMap, f64::consts::PI};

mod complex;
mod lookup;

use complex::Complex;

use crate::{
    pos::Pos4D,
    shapes::{
        Color::{self, Green, Purple},
        Edge, Face, Node, Object,
    }, 
    orbital::complex::{
        AbsArg, Conjugate, Split, Exp
    },
};

fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + z * z + y * y).sqrt();
    let t = (y / r).acos();
    let p = y.signum() * (z / (x * x + z * z).sqrt()).acos();

    (r, t, p)
}

fn _spherical_to_cartesian(r: f64, t: f64, p: f64) -> (f64, f64, f64) {
    let x = r * p.sin() * t.cos();
    let y = r * p.sin() * t.sin();
    let z = r * p.cos();

    (x, y, z)
}

fn radial_wave_function(n: i32, l: i32, r: f64, a: f64) -> Complex {
    Complex(1.0, 0.0)
        * match (n, l) {
            (1, 0) => 1.0 / (1.0 * a).powi(3).sqrt() * 2.0 * (-r / a).exp(),
            (2, 0) => 1.0 / (2.0 * a).powi(3).sqrt() * 2.0 * (1.0 - r / (2.0 * a)) * (-r / (2.0 * a)).exp(),
            (2, 1) => 1.0 / (2.0 * a).powi(3).sqrt() * (r / (3.0_f64.sqrt() * a)) * (-r / (2.0 * a)).exp(),
            (3, 0) => 1.0 / (3.0 * a).powi(3).sqrt() * (2.0 - 4.0 * r / (3.0 * a) + (4.0 * r * r) / (27.0 * a * a)) * (-r / (3.0 * a)).exp(),
            (3, 1) => 1.0 / (3.0 * a).powi(3).sqrt() * (4.0 * 2.0_f64.sqrt() * r) / (9.0 * a) * (1.0 - r / (6.0 * a)) * (-r / (3.0 * a)).exp(),
            (3, 2) => 1.0 / (3.0 * a).powi(3).sqrt() * (2.0 * 2.0_f64.sqrt() * r * r) / (27.0 * 5.0_f64.sqrt() * a * a) * (-r / (3.0 * a)).exp(),
            _ => 0.0,
        }
}

fn angular_wave_function(l: i32, m: i32, t: f64, p: f64, _a: f64) -> Complex {
    match (l, m) {
        (0, 0) => 1.0 / (4.0 * PI).sqrt() * Complex(1.0, 0.0),
        (1, -1) => -(3.0 / (8.0 * PI)).sqrt() * t.sin() * Complex(0.0, -p).exp(),
        (1, 0) => (3.0 / (4.0 * PI)).sqrt() * t.cos() * Complex(1.0, 0.0),
        (1, 1) => (3.0 / (8.0 * PI)).sqrt() * t.sin() * Complex(0.0, p).exp() * Complex(1.0, 0.0),
        (2, -2) => -(5.0 / (32.0 * PI)).sqrt() * t.sin() * t.sin() * Complex(0.0, -2.0 * p).exp(),
        (2, -1) => -(5.0 / (8.0 * PI)).sqrt() * t.cos() * t.sin() * Complex(0.0, -p).exp(),
        (2, 0) => (5.0 / (16.0 * PI)).sqrt() * (3.0 * t.cos().powi(2) - 1.0) * Complex(1.0, 0.0),
        (2, 1) => (5.0 / (8.0 * PI)).sqrt() * t.cos() * t.sin() * Complex(0.0, p).exp(),
        (2, 2) => (5.0 / (32.0 * PI)).sqrt() * t.sin() * t.sin() * Complex(0.0, 2.0 * p).exp(),
        (3, -3) => -(35.0 / (64.0 * PI)).sqrt() * t.sin().powi(3) * Complex(0.0, -3.0 * p).exp(),
        (3, -2) => -(105.0 / (32.0 * PI)).sqrt() * t.cos() * t.sin().powi(2) * Complex(0.0, -2.0 * p).exp(),
        (3, -1) => -(21.0 / (64.0 * PI)).sqrt() * (5.0 * t.cos().powi(2) - 1.0) * t.sin() * Complex(0.0, -p).exp(),
        (3, 0) => (7.0 / (16.0 * PI)).sqrt() * (5.0 * t.cos().powi(3) - 3.0 * t.cos()) * Complex(1.0, 0.0),
        (3, 1) => (21.0 / (64.0 * PI)).sqrt() * (5.0 * t.cos().powi(2) - 1.0) * t.sin() * Complex(0.0, p).exp(),
        (3, 2) => (105.0 / (32.0 * PI)).sqrt() * t.cos() * t.sin().powi(2) * Complex(0.0, 2.0 * p).exp(),
        (3, 3) => (35.0 / (64.0 * PI)).sqrt() * t.sin().powi(3) * Complex(0.0, 3.0 * p).exp(),
        _ => Complex(0.0, 0.0),
    }
}

fn adapt(start_value: f64, end_value: f64, cutoff: f64) -> f64 {
    (cutoff - start_value.abs()) / (end_value.abs() - start_value.abs())
}

pub fn create_orbital_v2(res: usize, psi_min: f64, psi_max: f64, a: f64, max: f64) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut faces: Vec<Face> = Vec::new();

    let (n, l, m) = (3, 1, 0);

    // Generate psi for a number of points inside a cube
    let mut psi_generated: HashMap<(usize, usize, usize), f64> = HashMap::new();

    for i in 0..res {
        println!(
            "Calculating psi for {} of {} points...",
            i * res * res,
            res * res * res
        );
        for j in 0..res {
            for k in 0..res {
                let pos = Pos4D {
                    x: ((i as f64 / res as f64) - 0.5) * max,
                    y: ((j as f64 / res as f64) - 0.5) * max,
                    z: ((k as f64 / res as f64) - 0.5) * max,
                    w: 0.0,
                };

                let (r, t, p): (f64, f64, f64) = cartesian_to_spherical(pos.x, pos.y, pos.z);

                let psi = radial_wave_function(n, l, r, a)
                    * angular_wave_function(l, m, t, p, a);

                psi_generated.insert((i, j, k), psi.Re());
            }
        }
    }

    // Marching cubes-like algoritm
    for i in 0..(res - 1) {
        println!(
            "Generating {} of {} points...",
            i * res * res,
            res * res * res
        );
        for j in 0..(res - 1) {
            for k in 0..(res - 1) {
                let pos = Pos4D {
                    x: ((i as f64 / res as f64) - 0.5) * max,
                    y: ((j as f64 / res as f64) - 0.5) * max,
                    z: ((k as f64 / res as f64) - 0.5) * max,
                    w: 0.0,
                };

                /*
                Get psi for the points at:
                0 (i      , j     , k     ), 0b0000_0001
                1 (i      , j     , k + 1 ), 0b0000_0010
                2 (i      , j + 1 , k     ), 0b0000_0100
                3 (i      , j + 1 , k + 1 ), 0b0000_1000
                4 (i + 1  , j     , k     ), 0b0001_0000
                5 (i + 1  , j     , k + 1 ), 0b0010_0000
                6 (i + 1  , j + 1 , k     ), 0b0100_0000
                7 (i + 1  , j + 1 , k + 1 ), 0b1000_0000
                */

                // Store the values of psi of neighbouring nodes in an array
                let local_psi_generated = [
                    *psi_generated.get(&(i, j, k)).unwrap_or(&0.0),
                    *psi_generated.get(&(i + 1, j, k)).unwrap_or(&0.0),
                    *psi_generated.get(&(i, j + 1, k)).unwrap_or(&0.0),
                    *psi_generated.get(&(i + 1, j + 1, k)).unwrap_or(&0.0),
                    *psi_generated.get(&(i, j, k + 1)).unwrap_or(&0.0),
                    *psi_generated.get(&(i + 1, j, k + 1)).unwrap_or(&0.0),
                    *psi_generated.get(&(i, j + 1, k + 1)).unwrap_or(&0.0),
                    *psi_generated.get(&(i + 1, j + 1, k + 1)).unwrap_or(&0.0),
                ];

                let mut positive_byte: u8 = 0x0;
                let mut negative_byte: u8 = 0x0;

                // Encode the valid and invalid nodes of the cube into a byte
                for (i, local_psi) in local_psi_generated.iter().enumerate() {
                    positive_byte ^= ((*local_psi >= psi_min) as u8) << i;
                    negative_byte ^= ((-*local_psi >= psi_min) as u8) << i;
                }

                fn run_marching_cubes(byte: u8, local_values: [f64; 8], node_index: usize, cutoff: f64, pos: Pos4D, color: Color, size: f64) -> (Vec<Node>, Vec<Edge>, Vec<Face>) {
                    // Don't draw empty or filled cubes
                    if byte == 0x00 && byte == 0xff {return (Vec::new(), Vec::new(), Vec::new())};

                    let (new_nodes, mut new_edges, mut new_faces) =
                        marching_cubes(local_values, cutoff, byte, pos, color, size);

                    // Update the indices to match the new node indices
                    new_edges.iter_mut().for_each(|edge| {
                        edge.start_node_index += node_index;
                        edge.end_node_index += node_index;
                    });
                    new_faces.iter_mut().for_each(|face| {
                        face.node_a_index += node_index;
                        face.node_b_index += node_index;
                        face.node_c_index += node_index;
                    });

                    (new_nodes, new_edges, new_faces)
                }

                let (mut new_nodes_neg, mut new_edges_neg, mut new_faces_neg) = run_marching_cubes(negative_byte, local_psi_generated, nodes.len(), psi_min, pos, Green, max / res as f64);
                let (mut new_nodes_pos, mut new_edges_pos, mut new_faces_pos) = run_marching_cubes(positive_byte, local_psi_generated, nodes.len(), psi_min, pos, Purple, max / res as f64);
                
                // Append the new nodes, edges and faces to the total object
                nodes.append(&mut new_nodes_neg);
                nodes.append(&mut new_nodes_pos);
                edges.append(&mut new_edges_neg);
                edges.append(&mut new_edges_pos);
                faces.append(&mut new_faces_neg);
                faces.append(&mut new_faces_pos);
            }
        }
    }

    Object {
        nodes,
        edges,
        faces,
    }
}

fn marching_cubes(
    value: [f64; 8],
    cutoff: f64,
    byte: u8,
    pos: Pos4D,
    color: Color,
    size: f64,
) -> (Vec<Node>, Vec<Edge>, Vec<Face>) {
    let mut nodes: Vec<Node> = Vec::new();
    let edges: Vec<Edge> = Vec::new();
    let mut faces: Vec<Face> = Vec::new();

    /*
    Calculate the position of the vertices and edges
    0 (i      , j     , k     ), 0b0000_0001
    1 (i + 1  , j     , k     ), 0b0000_0010
    2 (i      , j + 1 , k     ), 0b0000_0100
    3 (i + 1  , j + 1 , k     ), 0b0000_1000
    4 (i      , j     , k + 1 ), 0b0001_0000
    5 (i + 1  , j     , k + 1 ), 0b0010_0000
    6 (i      , j + 1 , k + 1 ), 0b0100_0000
    7 (i + 1  , j + 1 , k + 1 ), 0b1000_0000
    */

    // Get the face edges for the current cube from the lookup table
    let face_edge_indices = lookup::triangle_table(byte as usize);

    // Iterate over the faces for the current cube
    for face_edge_index in face_edge_indices.chunks(3).into_iter() {
        if face_edge_index[0] == -1 {break};

        // Get the positions of the vertices of the faces 
        let face_vertices = face_edge_index.into_iter().map(| edge | edge_to_boundary_vertex(*edge, value, cutoff, pos, size)).collect::<Vec<Pos4D>>();
        
        // Generate a new face
        let node_index_offset = nodes.len();
        faces.push(Face { node_a_index: node_index_offset, node_b_index: node_index_offset + 1, node_c_index: node_index_offset + 2, r: 0.5, color });

        // Generate the new nodes
        face_vertices.iter().for_each(|vertex| nodes.push(Node { pos: *vertex, r: 0.0, color }));
    }

    // Move the position of the vertex to the cutoff point
    fn edge_to_boundary_vertex(edge_index: i8, value: [f64; 8], cutoff: f64, pos: Pos4D, size: f64) -> Pos4D {
        let [vertex_0_index, vertex_1_index] = lookup::EDGE_VERTEX_INDICES[edge_index as usize];
        let t0 = 1.0 - adapt(value[vertex_0_index as usize], value[vertex_1_index as usize], cutoff);
        let t1 = 1.0 - t0;
        let vertex_0_pos = lookup::VERTEX_RELATIVE_POSITION[vertex_0_index as usize] * size;
        let vertex_1_pos = lookup::VERTEX_RELATIVE_POSITION[vertex_1_index as usize] * size;
        
        pos + vertex_0_pos * t0 + vertex_1_pos * t1
    }

    (nodes, edges, faces)
}
