use std::{
    collections::HashMap, 
    f32::consts::PI
};

mod complex;
mod lookup;

use complex::Complex;
use n_renderer::{render::{Node, Edge, Face, Object, Color}, pos::Pos4D, remove_duplicates};

use crate::orbital::complex::{Split, Exp, AbsArg};

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

fn cartesian_to_spherical(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let r = (x * x + z * z + y * y).sqrt();
    let r_nz = if r == 0.0 {0.0001} else {r};
    let t = (y / r_nz).acos();
    let xz_dist = (x * x + z * z).sqrt();
    let xz_dist_nz = if xz_dist == 0.0 {0.0001} else {xz_dist};
    let p = y.signum() * (z / xz_dist_nz).acos();

    (r, t, p)
}

fn _spherical_to_cartesian(r: f32, t: f32, p: f32) -> (f32, f32, f32) {
    let x = r * p.sin() * t.cos();
    let y = r * p.sin() * t.sin();
    let z = r * p.cos();

    (x, y, z)
}

/// Calculates the radial part of the wave function, R(r) for a given n, l, m and bohr radius, a
fn radial_wave_function(n: i32, l: i32, r: f32, a: f32) -> f32 {
    // Complex(1.0, 0.0)
    //     * match (n, l) {
    //         (1, 0) => 1.0 / (1.0 * a).powi(3).sqrt() * 2.0 * (-r / a).exp(),
    //         (2, 0) => 1.0 / (2.0 * a).powi(3).sqrt() * 2.0 * (1.0 - r / (2.0 * a)) * (-r / (2.0 * a)).exp(),
    //         (2, 1) => 1.0 / (2.0 * a).powi(3).sqrt() * (r / (3.0_f32.sqrt() * a)) * (-r / (2.0 * a)).exp(),
    //         (3, 0) => 1.0 / (3.0 * a).powi(3).sqrt() * (2.0 - 4.0 * r / (3.0 * a) + (4.0 * r * r) / (27.0 * a * a)) * (-r / (3.0 * a)).exp(),
    //         (3, 1) => 1.0 / (3.0 * a).powi(3).sqrt() * (4.0 * 2.0_f32.sqrt() * r) / (9.0 * a) * (1.0 - r / (6.0 * a)) * (-r / (3.0 * a)).exp(),
    //         (3, 2) => 1.0 / (3.0 * a).powi(3).sqrt() * (2.0 * 2.0_f32.sqrt() * r * r) / (27.0 * 5.0_f32.sqrt() * a * a) * (-r / (3.0 * a)).exp(),
    //         _ => 0.0,
    //     }
    
    // General radial wave function
    (8.0 / (n as f32 * a).powi(3) * (n - l - 1).factorial() as f32 / (2 * n * (n + l).factorial()) as f32).sqrt() * (-r / (n as f32 * a)).exp() * (2.0 * r).powi(l) / (n as f32 * a).powi(l) * laguerre_polynomials(2 * l + 1, n - l - 1, (2.0 * r) / (n as f32 * a))
}

/// Calculates the generalised/associated laguerre polynomials (L^k_n (x))for a given k, n and x
/// Uses the Rodrigues representation: https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
pub fn laguerre_polynomials(k: i32, n: i32, x: f32) -> f32 {
    let mut result = 0.0;
    for m in 0..=n {
        result += (-1_i32).pow(m as u32) as f32 * (n + k).factorial() as f32 / ((n - m).factorial() * (k + m).factorial() * m.factorial()) as f32 * x.powi(m)
    }
    result
}

fn angular_wave_function(l: i32, m: i32, t: f32, p: f32, _a: f32) -> Complex {
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

fn psi((n, l, m): (i32, i32, i32), (r, t, p): (f32, f32, f32), a: f32) -> Complex {
    radial_wave_function(n, l, r, a) * angular_wave_function(l, m, t, p, a)
}

pub fn create_orbital(res: usize, psi_min: f32, max: f32, a: f32, (n, l, m): (i32, i32, i32)) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut faces: Vec<Face> = Vec::new();

    // let s = (2, 0, 0);
    // let pz = (2, 1, 0);
    // let px = (2, 1, 1);
    // let py = (2, 1, -1);
    
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
                    x: ((i as f32 / res as f32) - 0.5) * max,
                    y: ((j as f32 / res as f32) - 0.5) * max,
                    z: ((k as f32 / res as f32) - 0.5) * max,
                    w: 0.0,
                };

                let sc: (f32, f32, f32) = cartesian_to_spherical(pos.x, pos.y, pos.z);
                // let sc_A: (f32, f32, f32) = cartesian_to_spherical(pos.x - 1.0, pos.y, pos.z);
                // let sc_B: (f32, f32, f32) = cartesian_to_spherical(pos.x + 1.0, pos.y, pos.z);

                let psi_d = psi((n, l, m), sc, a);
                // let psi_A = psi(px, sc_A, a);
                // let psi_B = psi(px, sc_B, a);

                // let psi_s = psi(s, sc, a);
                // let psi_x = psi(px, sc, a);
                // let psi_y = psi(py, sc, a);
                // let psi_z = psi(pz, sc, a);c

                // let sp3_1 = Complex(0.5, 0.0) * (psi_s + psi_x + psi_y + psi_z);

                psi_generated.insert((i, j, k), psi_d.Re() * Complex(1.0, 0.0));
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
                    x: ((i as f32 / res as f32) - 0.5) * max,
                    y: ((j as f32 / res as f32) - 0.5) * max,
                    z: ((k as f32 / res as f32) - 0.5) * max,
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

                const DEFAULT: Complex = Complex(0.0, 0.0);

                // Store the values of psi of neighbouring nodes in a smaller array
                let local_psi_generated = [
                    *psi_generated.get(&(i + 0, j + 0, k + 0)).unwrap_or(&DEFAULT),
                    *psi_generated.get(&(i + 1, j + 0, k + 0)).unwrap_or(&DEFAULT),
                    *psi_generated.get(&(i + 0, j + 1, k + 0)).unwrap_or(&DEFAULT),
                    *psi_generated.get(&(i + 1, j + 1, k + 0)).unwrap_or(&DEFAULT),
                    *psi_generated.get(&(i + 0, j + 0, k + 1)).unwrap_or(&DEFAULT),
                    *psi_generated.get(&(i + 1, j + 0, k + 1)).unwrap_or(&DEFAULT),
                    *psi_generated.get(&(i + 0, j + 1, k + 1)).unwrap_or(&DEFAULT),
                    *psi_generated.get(&(i + 1, j + 1, k + 1)).unwrap_or(&DEFAULT),
                ];

                let mut byte: u8 = 0x0;
                // Encode the valid and invalid nodes of the cube into a byte
                for (i, local_psi) in local_psi_generated.iter().enumerate() {
                    byte ^= ((local_psi.abs() >= psi_min) as u8) << i;
                }

                /// Function to run the marching cubes algorithm for a single cube and add the resulting nodes edges and faces to the object, transform the vectors that were input
                fn run_marching_cubes(byte: u8, local_values: [Complex; 8], cutoff: f32, pos: Pos4D, size: f32, mut nodes: Vec<Node>, mut edges: Vec<Edge>, mut faces: Vec<Face>) -> (Vec<Node>, Vec<Edge>, Vec<Face>) {
                    // Don't draw empty or filled cubes
                    if byte == 0x00 && byte == 0xff {return (Vec::new(), Vec::new(), Vec::new())};

                    // Get the new nodes from the marching cubes algoritm
                    let (mut new_nodes, mut new_edges, mut new_faces) =
                        marching_cubes(local_values, cutoff, byte, pos, size);

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

                (nodes, edges, faces) = run_marching_cubes(byte, local_psi_generated, psi_min, pos, max / res as f32, nodes, edges, faces);
            }
        }
    }
    
    println!("Object contains {} points before optimisation...", nodes.len());
    let object = remove_duplicates(Object {
        nodes,
        edges,
        faces,
    });
    println!("Object contains {} points after optimisation...", object.nodes.len());
    object
}

fn marching_cubes(
    value: [Complex; 8],
    cutoff: f32,
    byte: u8,
    pos: Pos4D,
    size: f32,
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
    for face_edge_index in face_edge_indices.chunks(3) {
        if face_edge_index[0] == -1 {break};

        // Get the positions of the vertices of the faces 
        let face_vertices_values = face_edge_index.iter().map(| edge | 
            edge_to_boundary_vertex(*edge as usize, value, cutoff, pos, size)
        ).collect::<Vec<(Pos4D, Complex)>>();
        
        // Generate a new face
        let node_index_offset = nodes.len();
        faces.push(Face { node_a_index: node_index_offset, node_b_index: node_index_offset + 1, node_c_index: node_index_offset + 2, r: 15 });

        // Generate the new nodes
        face_vertices_values.iter().for_each(|&(vertex, value)| nodes.push(Node { pos: vertex, r: 0, color: {
            Color::HSV((value.arg() / PI * 180.0 + 180.0) as u16, (0.8 * 256.0) as u8, (0.7 * 256.0) as u8)
        }}));
    }

    // Move the position of the vertex to the cutoff point
    fn edge_to_boundary_vertex(edge_index: usize, value: [Complex; 8], cutoff: f32, pos: Pos4D, size: f32) -> (Pos4D, Complex) {
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

    fn adapt(start_value: f32, end_value: f32, cutoff: f32) -> f32 {
        ((cutoff - start_value.abs()) / (end_value.abs() - start_value.abs())).clamp(0.0, 1.0)
    }

    (nodes, edges, faces)
}