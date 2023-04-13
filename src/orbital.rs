use std::{collections::HashMap, f64::consts::PI};

mod complex;
mod lookup;

use complex::Complex;
use n_renderer::{render::{Node, Edge, Face, Object, Color}, pos::Pos4D};

use crate::{ 
    orbital::complex::{
        Split, Exp, AbsArg, Conjugate
    },
};

pub trait Factorial {
    type Output;

    fn factorial(self) -> Self::Output;
}

impl Factorial for i32 {
    type Output = i32;

    fn factorial(self) -> Self::Output {
        if self <= 1 {
            1
        } else {
            self * (self - 1).factorial()
        }
    }
}

fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x * x + z * z + y * y).sqrt();
    let r_nz = if r == 0.0 {0.0001} else {r};
    let t = (y / r_nz).acos();
    let xz_dist = (x * x + z * z).sqrt();
    let xz_dist_nz = if xz_dist == 0.0 {0.0001} else {xz_dist};
    let p = y.signum() * (z / xz_dist_nz).acos();

    (r, t, p)
}

fn _spherical_to_cartesian(r: f64, t: f64, p: f64) -> (f64, f64, f64) {
    let x = r * p.sin() * t.cos();
    let y = r * p.sin() * t.sin();
    let z = r * p.cos();

    (x, y, z)
}

/// Calculates the radial part of the wave function, R(r) for a given n, l, m and bohr radius, a
fn radial_wave_function(n: i32, l: i32, r: f64, a: f64) -> f64 {
    // Complex(1.0, 0.0)
    //     * match (n, l) {
    //         (1, 0) => 1.0 / (1.0 * a).powi(3).sqrt() * 2.0 * (-r / a).exp(),
    //         (2, 0) => 1.0 / (2.0 * a).powi(3).sqrt() * 2.0 * (1.0 - r / (2.0 * a)) * (-r / (2.0 * a)).exp(),
    //         (2, 1) => 1.0 / (2.0 * a).powi(3).sqrt() * (r / (3.0_f64.sqrt() * a)) * (-r / (2.0 * a)).exp(),
    //         (3, 0) => 1.0 / (3.0 * a).powi(3).sqrt() * (2.0 - 4.0 * r / (3.0 * a) + (4.0 * r * r) / (27.0 * a * a)) * (-r / (3.0 * a)).exp(),
    //         (3, 1) => 1.0 / (3.0 * a).powi(3).sqrt() * (4.0 * 2.0_f64.sqrt() * r) / (9.0 * a) * (1.0 - r / (6.0 * a)) * (-r / (3.0 * a)).exp(),
    //         (3, 2) => 1.0 / (3.0 * a).powi(3).sqrt() * (2.0 * 2.0_f64.sqrt() * r * r) / (27.0 * 5.0_f64.sqrt() * a * a) * (-r / (3.0 * a)).exp(),
    //         _ => 0.0,
    //     }
    
    // General radial wave function
    (8.0 / (n as f64 * a).powi(3) * (n - l - 1).factorial() as f64 / (2 * n * (n + l).factorial()) as f64).sqrt() * (-r / (n as f64 * a)).exp() * (2.0 * r).powi(l) / (n as f64 * a).powi(l) * laguerre_polynomials(2 * l + 1, n - l - 1, (2.0 * r) / (n as f64 * a))
}

/// Calculates the generalised/associated laguerre polynomials (L^k_n (x))for a given k, n and x
/// Uses the Rodrigues representation: https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
pub fn laguerre_polynomials(k: i32, n: i32, x: f64) -> f64 {
    let mut result = 0.0;
    for m in 0..=n {
        result += (-1_i32).pow(m as u32) as f64 * (n + k).factorial() as f64 / ((n - m).factorial() * (k + m).factorial() * m.factorial()) as f64 * x.powi(m)
    }
    result
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

fn psi((n, l, m): (i32, i32, i32), (r, t, p): (f64, f64, f64), a: f64) -> Complex {
    radial_wave_function(n, l, r, a) * angular_wave_function(l, m, t, p, a)
}

pub fn create_orbital(res: usize, psi_min: f64, max: f64, a: f64, (n, l, m): (i32, i32, i32)) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut faces: Vec<Face> = Vec::new();

    let s = (2, 0, 0);
    let pz = (2, 1, 0);
    let px = (2, 1, 1);
    let py = (2, 1, -1);
    
    // Generate psi for a number of points inside a cube
    let mut psi_generated: HashMap<(usize, usize, usize), Complex> = HashMap::new();

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

                let sc: (f64, f64, f64) = cartesian_to_spherical(pos.x, pos.y, pos.z);
                // let sc_A: (f64, f64, f64) = cartesian_to_spherical(pos.x - 1.0, pos.y, pos.z);
                // let sc_B: (f64, f64, f64) = cartesian_to_spherical(pos.x + 1.0, pos.y, pos.z);

                // let psi_d = psi((n, l, m), sc, a);
                // let psi_A = psi(px, sc_A, a);
                // let psi_B = psi(px, sc_B, a);

                let psi_s = psi(s, sc, a);
                let psi_x = psi(px, sc, a);
                let psi_y = psi(py, sc, a);
                let psi_z = psi(pz, sc, a);

                let sp3_1 = Complex(0.5, 0.0) * (psi_s + psi_x + psi_y + psi_z);

                psi_generated.insert((i, j, k), sp3_1.Re() * Complex(1.0, 0.0));
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

                // Store the values of psi of neighbouring nodes in a smaller array
                let local_psi_generated = [
                    *psi_generated.get(&(i, j, k)).unwrap_or(&Complex(0.0, 0.0)),
                    *psi_generated.get(&(i + 1, j, k)).unwrap_or(&Complex(0.0, 0.0)),
                    *psi_generated.get(&(i, j + 1, k)).unwrap_or(&Complex(0.0, 0.0)),
                    *psi_generated.get(&(i + 1, j + 1, k)).unwrap_or(&Complex(0.0, 0.0)),
                    *psi_generated.get(&(i, j, k + 1)).unwrap_or(&Complex(0.0, 0.0)),
                    *psi_generated.get(&(i + 1, j, k + 1)).unwrap_or(&Complex(0.0, 0.0)),
                    *psi_generated.get(&(i, j + 1, k + 1)).unwrap_or(&Complex(0.0, 0.0)),
                    *psi_generated.get(&(i + 1, j + 1, k + 1)).unwrap_or(&Complex(0.0, 0.0)),
                ];

                let mut byte: u8 = 0x0;
                // Encode the valid and invalid nodes of the cube into a byte
                for (i, local_psi) in local_psi_generated.iter().enumerate() {
                    byte ^= ((local_psi.abs() >= psi_min) as u8) << i;
                }

                /// Function to run the marching cubes algorithm for a single cube and add the resulting nodes edges and faces to the object, transform the vectors that were input
                fn run_marching_cubes(byte: u8, local_values: [Complex; 8], cutoff: f64, pos: Pos4D, color: Color, size: f64, mut nodes: Vec<Node>, mut edges: Vec<Edge>, mut faces: Vec<Face>) -> (Vec<Node>, Vec<Edge>, Vec<Face>) {
                    // Don't draw empty or filled cubes
                    if byte == 0x00 && byte == 0xff {return (Vec::new(), Vec::new(), Vec::new())};

                    // Get the new nodes from the marching cubes algoritm
                    let (mut new_nodes, mut new_edges, mut new_faces) =
                        marching_cubes(local_values, cutoff, byte, pos, color, size);

                    // Get the current node index, so the edges and faces can be updated and properly appended
                    let node_index = nodes.len();

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

                    // Append the nodes, edges and faces with the shifted indices to the total
                    nodes.append(&mut new_nodes);
                    edges.append(&mut new_edges);
                    faces.append(&mut new_faces);

                    (nodes, edges, faces)
                }

                (nodes, edges, faces) = run_marching_cubes(byte, local_psi_generated, psi_min, pos, Color::White, max / res as f64, nodes, edges, faces);
            }
        }
    }
    
    println!("Object contains {} points before optimisation...", nodes.len());
    (nodes, edges, faces) = remove_duplicates(nodes, edges, faces, 0.01);
    println!("Object contains {} points after optimisation...", nodes.len());

    Object {
        nodes,
        edges,
        faces,
    }
}

fn marching_cubes(
    value: [Complex; 8],
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
        let face_vertices_values = face_edge_index.into_iter().map(| edge | edge_to_boundary_vertex(*edge as usize, value, cutoff, pos, size)).collect::<Vec<(Pos4D, Complex)>>();
        
        // Generate a new face
        let node_index_offset = nodes.len();
        faces.push(Face { node_a_index: node_index_offset, node_b_index: node_index_offset + 1, node_c_index: node_index_offset + 2, r: 1.5 });

        // Generate the new nodes
        face_vertices_values.iter().for_each(|&(vertex, value)| nodes.push(Node { pos: vertex, r: 0.0, color: {
            hue_to_rgb(value.arg() + PI, 0.5, 0.7)
        }}));
    }

    fn hue_to_rgb(h: f64, s: f64, v: f64) -> Color {
        let region = 3.0 * h / PI;
        let c = v * s;
        let x = c * (1.0 - (region % 2.0 - 1.0).abs());

        let (r, g, b) = {
            if (0.0..1.0).contains(&region) {
                (c, x, 0.0)
            } else if (1.0..2.0).contains(&region) {
                (x, c, 0.0)
            } else if (2.0..3.0).contains(&region) {
                (0.0, c, x)
            } else if (3.0..4.0).contains(&region) {
                (0.0, x, c)
            } else if (4.0..5.0).contains(&region) {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            }
        };

        Color::RGB(((r + (v - c)) * 255.0).clamp(0.0, 255.0) as u8, ((g + (v - c)) * 255.0).clamp(0.0, 255.0) as u8, ((b + (v - c)) * 255.0).clamp(0.0, 255.0) as u8)
    }

    // Move the position of the vertex to the cutoff point
    fn edge_to_boundary_vertex(edge_index: usize, value: [Complex; 8], cutoff: f64, pos: Pos4D, size: f64) -> (Pos4D, Complex) {
        let [vertex_0_index, vertex_1_index] = lookup::EDGE_VERTEX_INDICES[edge_index];
        let t0 = 1.0 - adapt(value[vertex_0_index].abs(), value[vertex_1_index].abs(), cutoff);
        let t1 = 1.0 - t0;
        let vertex_0_pos = {
            let [x, y, z, _] = lookup::VERTEX_RELATIVE_POSITION[vertex_0_index];
            Pos4D { x, y, z, w: value[vertex_0_index].Im() } * size
        };
        let vertex_1_pos = {
            let [x, y, z, _] = lookup::VERTEX_RELATIVE_POSITION[vertex_1_index];
            Pos4D { x, y, z, w: value[vertex_1_index].Im() } * size
        };
        
        (pos + vertex_0_pos * t0 + vertex_1_pos * t1, value[vertex_0_index] * t0 + value[vertex_1_index] * t1)
    }

    fn adapt(start_value: f64, end_value: f64, cutoff: f64) -> f64 {
        ((cutoff - start_value.abs()) / (end_value.abs() - start_value.abs())).clamp(0.0, 1.0)
    }

    (nodes, edges, faces)
}

fn remove_duplicates(nodes: Vec<Node>, edges: Vec<Edge>, faces: Vec<Face>, _threshold: f64) -> (Vec<Node>, Vec<Edge>, Vec<Face>) {
    let mut unique_nodes: Vec<Node> = Vec::new();
    let mut unique_edges: Vec<Edge> = Vec::new();
    let mut unique_faces: Vec<Face> = Vec::new();

    // Create a hashmap to store all duplicate vertices
    let mut duplicated_nodes: HashMap<usize, usize> = HashMap::new();

    for (old_index, &node) in nodes.iter().enumerate() {
        let new_index = if !unique_nodes.contains(&node) {
            unique_nodes.push(node);
            unique_nodes.len() - 1
        } else {
            unique_nodes.iter().position(|&unique_node| unique_node == node).unwrap()
        };
        // println!("Moved node from {} to {}", old_index, new_index);
        duplicated_nodes.insert(old_index, new_index);
    }

    for edge in edges {
        let mut unique_edge = edge.clone();
        if duplicated_nodes.contains_key(&edge.start_node_index) {
            unique_edge.start_node_index = *duplicated_nodes.get(&edge.start_node_index).unwrap();
        }
        if duplicated_nodes.contains_key(&edge.end_node_index) {
            unique_edge.end_node_index = *duplicated_nodes.get(&edge.end_node_index).unwrap();
        }

        if unique_edge.start_node_index >= unique_nodes.len() || unique_edge.end_node_index >= unique_nodes.len() {
            println!("{:?}", unique_edge)
        } else {
            unique_edges.push(unique_edge);
        }
    }

    for face in faces {
        let mut unique_face = face.clone();
        if duplicated_nodes.contains_key(&face.node_a_index) {
            unique_face.node_a_index = *duplicated_nodes.get(&face.node_a_index).unwrap();
        }
        if duplicated_nodes.contains_key(&face.node_b_index) {
            unique_face.node_b_index = *duplicated_nodes.get(&face.node_b_index).unwrap();
        }
        if duplicated_nodes.contains_key(&face.node_c_index) {
            unique_face.node_c_index = *duplicated_nodes.get(&face.node_c_index).unwrap();
        }

        if unique_face.node_a_index >= unique_nodes.len() || unique_face.node_b_index >= unique_nodes.len() || unique_face.node_c_index >= unique_nodes.len() {
            println!("{:?}", unique_face)
        } else {
            unique_faces.push(unique_face)
        }
    }

    (unique_nodes, unique_edges, unique_faces)
}