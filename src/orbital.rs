// pub fn create_orbital(res: i32, psi_min: f64, psi_max: f64, scale: f64) -> Object {
//     let mut nodes: Vec<Node> = Vec::new();
//     let edges: Vec<Edge> = Vec::new();
//     let faces: Vec<Face> = Vec::new();

//     let phi = PI * (3.0 - (5.0_f64).sqrt());

//     for psi in (psi_min * res as f64) as i32..(psi_max * res as f64) as i32 {
//         for i in 0..res {
//             let y = 1.0 - (i as f64 / (res - 1) as f64) * 2.0;
//             let radius = (1.0 - y * y).sqrt();

//             let theta = phi * i as f64;

//             let x = theta.cos() * radius;
//             let z = theta.sin() * radius;

//             // let (r, t, p) = cartesian_to_spherical(x, y, z);

//             // psi = 1.0 / (1 * scale).powf(3.0 / 2.0) * 2.0 * E.powf(-r / scale);
//             let r = -((psi / res) as f64 * (1.0 * scale).powf(3.0 / 2.0) / 2.0).ln()
//                 * scale
//                 * (4.0 * PI).sqrt();

//             // Calculate the new position of the point
//             let pos = Pos4D { x, y, z, w: 0.0 } * r;

//             nodes.push(Node {
//                 pos: pos,
//                 r: (psi / res) as f64,
//                 color: Purple,
//             })
//         }
//     }

//     Object {
//         nodes,
//         edges,
//         faces,
//     }
// }

use std::f64::consts::PI;

mod complex;

use complex::Complex;

use crate::{
    pos::Pos4D,
    shapes::{
        Color::{self, Green, Purple},
        Edge, Face, Node, Object,
    },
};

use self::complex::{Exp, Split};

fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let x_nz = x; //if x as i32 == 0 {0.00000001} else {x};
    let z_nz = y; //if z as i32 == 0 {0.00000001} else {z};

    let r = (x * x + z * z + y * y).sqrt();
    let t = ((x * x + z * z).sqrt() / z_nz).atan();
    let p = (z / x_nz).atan();

    (r, t, p)
}

fn _spherical_to_cartesian(r: f64, t: f64, p: f64) -> (f64, f64, f64) {
    let x = r * p.sin() * t.cos();
    let y = r * p.sin() * t.sin();
    let z = r * p.cos();

    (x, y, z)
}

fn radial_wave_function(n: i32, l: i32, r: f64, a: f64) -> f64 {
    match (n, l) {
        (1, 0) => 1.0 / (1.0 * a).powi(3).sqrt() * 2.0 * (-r / a).exp(),
        (2, 0) => {
            1.0 / (2.0 * a).powi(3).sqrt() * 2.0 * (1.0 - r / (2.0 * a)) * (-r / (2.0 * a)).exp()
        }
        (2, 1) => {
            1.0 / (2.0 * a).powi(3).sqrt() * (r / (3.0_f64.sqrt() * a)) * (-r / (2.0 * a)).exp()
        }
        (3, 0) => {
            1.0 / (3.0 * a).powi(3).sqrt()
                * (2.0 - 4.0 * r / (3.0 * a) + (4.0 * r * r) / (27.0 * a * a))
                * (-r / (3.0 * a)).exp()
        }
        (3, 1) => {
            1.0 / (3.0 * a).powi(3).sqrt() * (4.0 * 2.0_f64.sqrt() * r) / (9.0 * a)
                * (1.0 - r / (6.0 * a))
                * (-r / (3.0 * a)).exp()
        }
        (3, 2) => {
            1.0 / (3.0 * a).powi(3).sqrt() * (2.0 * 2.0_f64.sqrt() * r * r)
                / (27.0 * 5.0_f64.sqrt() * a * a)
                * (-r / (3.0 * a)).exp()
        }
        _ => 0.0,
    }
}

fn angular_wave_function(l: i32, m: i32, t: f64, p: f64, _a: f64) -> f64 {
    match (l, m) {
        (0, 0) => 1.0 / (4.0 * PI).sqrt(),
        (1, -1) => -((3.0 / (8.0 * PI)).sqrt() * t.sin() * Complex(0.0, -p).exp()).Re(),
        (1, 0) => (3.0 / (4.0 * PI)).sqrt() * t.cos(),
        (1, 1) => ((3.0 / (8.0 * PI)).sqrt() * t.sin() * Complex(0.0, p).exp()).Re(),
        (2, -2) => {
            -((5.0 / (32.0 * PI)).sqrt() * t.sin() * t.sin() * Complex(0.0, -2.0 * p).exp()).Re()
        }
        (2, -1) => -((5.0 / (8.0 * PI)).sqrt() * t.cos() * t.sin() * Complex(0.0, -p).exp()).Re(),
        (2, 0) => (5.0 / (16.0 * PI)).sqrt() * (3.0 * t.cos().powi(2) - 1.0),
        (2, 1) => ((5.0 / (8.0 * PI)).sqrt() * t.cos() * t.sin() * Complex(0.0, p).exp()).Re(),
        (2, 2) => {
            ((5.0 / (32.0 * PI)).sqrt() * t.sin() * t.sin() * Complex(0.0, 2.0 * p).exp()).Re()
        }
        (3, -3) => {
            -((35.0 / (64.0 * PI)).sqrt() * t.sin().powi(3) * Complex(0.0, -3.0 * p).exp()).Re()
        }
        (3, -2) => -((105.0 / (32.0 * PI)).sqrt()
            * t.cos()
            * t.sin().powi(2)
            * Complex(0.0, -2.0 * p).exp())
        .Re(),
        (3, -1) => -((21.0 / (64.0 * PI)).sqrt()
            * (5.0 * t.cos().powi(2) - 1.0)
            * t.sin()
            * Complex(0.0, -p).exp())
        .Re(),
        (3, 0) => (7.0 / (16.0 * PI)).sqrt() * (5.0 * t.cos().powi(3) - 3.0 * t.cos()),
        (3, 1) => ((21.0 / (64.0 * PI)).sqrt()
            * (5.0 * t.cos().powi(2) - 1.0)
            * t.sin()
            * Complex(0.0, p).exp())
        .Re(),
        (3, 2) => {
            ((105.0 / (32.0 * PI)).sqrt() * t.cos() * t.sin().powi(2) * Complex(0.0, 2.0 * p).exp())
                .Re()
        }
        (3, 3) => {
            ((35.0 / (64.0 * PI)).sqrt() * t.sin().powi(3) * Complex(0.0, 3.0 * p).exp()).Re()
        }
        _ => 0.0,
    }
}

fn adapt(start_value: f64, end_value: f64, cutoff: f64) -> f64 {
    (cutoff - start_value.abs()) / (end_value.abs() - start_value.abs()) //.clamp(-1.0, 1.0)
                                                                         // 0.5
}

pub fn create_orbital_v2(_res: usize, psi_min: f64, psi_max: f64, a: f64, max: f64) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let faces: Vec<Face> = Vec::new();

    const RESOLUTION: usize = 50;

    let (n, l, m) = (1, 0, 0);

    // Generate psi for a number of points inside a cube
    let mut psi_generated: [[[f64; RESOLUTION]; RESOLUTION]; RESOLUTION] =
        [[[0.0; RESOLUTION]; RESOLUTION]; RESOLUTION];

    for i in 0..RESOLUTION {
        println!(
            "Calculating psi for {} of {} points...",
            i * RESOLUTION * RESOLUTION,
            RESOLUTION * RESOLUTION * RESOLUTION
        );
        for j in 0..RESOLUTION {
            for k in 0..RESOLUTION {
                let pos = Pos4D {
                    x: ((i as f64 / RESOLUTION as f64) - 0.5) * max,
                    y: ((j as f64 / RESOLUTION as f64) - 0.5) * max,
                    z: ((k as f64 / RESOLUTION as f64) - 0.5) * max,
                    w: 0.0,
                };

                let (r, t, p): (f64, f64, f64) = cartesian_to_spherical(pos.x, pos.y, pos.z);

                let psi: f64 =
                    radial_wave_function(n, l, r, a) * angular_wave_function(l, m, t, p, a);

                psi_generated[i][j][k] = psi;

                // let psi_squared: f64 = psi * psi;
            }
        }
    }

    // Marching cubes-like algoritm
    for i in 0..(RESOLUTION - 1) {
        println!(
            "Generating {} of {} points...",
            i * RESOLUTION * RESOLUTION,
            RESOLUTION * RESOLUTION * RESOLUTION
        );
        for j in 0..(RESOLUTION - 1) {
            for k in 0..(RESOLUTION - 1) {
                let pos = Pos4D {
                    x: ((i as f64 / RESOLUTION as f64) - 0.5) * max,
                    y: ((j as f64 / RESOLUTION as f64) - 0.5) * max,
                    z: ((k as f64 / RESOLUTION as f64) - 0.5) * max,
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
                let local_psi_generated: [f64; 8] = [
                    psi_generated[i][j][k],
                    psi_generated[i][j][k + 1],
                    psi_generated[i][j + 1][k],
                    psi_generated[i][j + 1][k + 1],
                    psi_generated[i + 1][j][k],
                    psi_generated[i + 1][j][k + 1],
                    psi_generated[i + 1][j + 1][k],
                    psi_generated[i + 1][j + 1][k + 1],
                ];

                let mut byte: u8 = 0x0;

                // Encode the valid and invalid nodes of the cube into a byte
                for (i, local_psi) in local_psi_generated.iter().enumerate() {
                    let is_in_range = local_psi.abs() <= psi_max && local_psi.abs() >= psi_min;
                    byte ^= (is_in_range as u8) << i;
                }

                // Don't draw empty or filled cubes
                if byte != 0x00 && byte != 0xff {
                    let (mut new_nodes, mut new_edges) = marching_cubes(
                        local_psi_generated,
                        psi_min,
                        byte,
                        pos,
                        max / RESOLUTION as f64,
                    );
                    nodes.append(&mut new_nodes);
                    edges.append(&mut new_edges);
                }
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
    size: f64,
) -> (Vec<Node>, Vec<Edge>) {
    let mut nodes: Vec<Node> = Vec::new();
    let edges: Vec<Edge> = Vec::new();
    // let faces: Vec<Face> = Vec::new();

    /*
    Calculate the position of the vertices and edges
    0 (i      , j     , k     ), 0b0000_0001
    1 (i      , j     , k + 1 ), 0b0000_0010
    2 (i      , j + 1 , k     ), 0b0000_0100
    3 (i      , j + 1 , k + 1 ), 0b0000_1000
    4 (i + 1  , j     , k     ), 0b0001_0000
    5 (i + 1  , j     , k + 1 ), 0b0010_0000
    6 (i + 1  , j + 1 , k     ), 0b0100_0000
    7 (i + 1  , j + 1 , k + 1 ), 0b1000_0000
    */

    let color: Color = if value[0] < 0.0 { Green } else { Purple };

    // Test what edges have only one of two points active
    if byte >> 0 & 1 == 1 && byte >> 1 & 1 == 0 {
        // 0 & !1
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: 0.0,
                    z: adapt(value[0], value[1], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 0 & 1 == 1 && byte >> 2 & 1 == 0 {
        // 0 & !2
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: adapt(value[0], value[2], cutoff) * size,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 0 & 1 == 1 && byte >> 4 & 1 == 0 {
        // 0 & !4
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: adapt(value[0], value[4], cutoff) * size,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    if byte >> 1 & 1 == 1 && byte >> 0 & 1 == 0 {
        // 1 & !0
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: 0.0,
                    z: -adapt(value[1], value[0], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 1 & 1 == 1 && byte >> 3 & 1 == 0 {
        // 1 & !3
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: adapt(value[1], value[3], cutoff) * size,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 1 & 1 == 1 && byte >> 5 & 1 == 0 {
        // 1 & !5
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: adapt(value[1], value[5], cutoff) * size,
                    y: 0.0,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    if byte >> 2 & 1 == 1 && byte >> 3 & 1 == 0 {
        // 2 & !3
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: size,
                    z: adapt(value[2], value[3], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 2 & 1 == 1 && byte >> 0 & 1 == 0 {
        // 2 & !0
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: -adapt(value[2], value[0], cutoff) * size,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 2 & 1 == 1 && byte >> 6 & 1 == 0 {
        // 2 & !6
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: adapt(value[2], value[6], cutoff) * size,
                    y: size,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    if byte >> 3 & 1 == 1 && byte >> 2 & 1 == 0 {
        // 3 & !2
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: size,
                    z: -adapt(value[3], value[2], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 3 & 1 == 1 && byte >> 1 & 1 == 0 {
        // 3 & !1
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: 0.0,
                    y: -adapt(value[3], value[1], cutoff) * size,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 3 & 1 == 1 && byte >> 7 & 1 == 0 {
        // 3 & !7
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: adapt(value[3], value[7], cutoff) * size,
                    y: size,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    if byte >> 4 & 1 == 1 && byte >> 5 & 1 == 0 {
        // 4 & !5
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: 0.0,
                    z: adapt(value[4], value[5], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 4 & 1 == 1 && byte >> 6 & 1 == 0 {
        // 4 & !6
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: adapt(value[4], value[6], cutoff) * size,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 4 & 1 == 1 && byte >> 0 & 1 == 0 {
        // 4 & !0
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: -adapt(value[4], value[0], cutoff) * size,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    if byte >> 5 & 1 == 1 && byte >> 4 & 1 == 0 {
        // 5 & !4
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: 0.0,
                    z: -adapt(value[5], value[4], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 5 & 1 == 1 && byte >> 7 & 1 == 0 {
        // 5 & !7
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: adapt(value[5], value[7], cutoff) * size,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 5 & 1 == 1 && byte >> 1 & 1 == 0 {
        // 5 & !1
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: -adapt(value[5], value[1], cutoff) * size,
                    y: 0.0,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    if byte >> 6 & 1 == 1 && byte >> 7 & 1 == 0 {
        // 6 & !7
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: size,
                    z: adapt(value[6], value[7], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 6 & 1 == 1 && byte >> 4 & 1 == 0 {
        // 6 & !4
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: -adapt(value[6], value[4], cutoff) * size,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 6 & 1 == 1 && byte >> 2 & 1 == 0 {
        // 6 & !2
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: -adapt(value[6], value[2], cutoff) * size,
                    y: size,
                    z: 0.0,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    if byte >> 7 & 1 == 1 && byte >> 6 & 1 == 0 {
        // 7 & !6
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: size,
                    z: -adapt(value[7], value[6], cutoff) * size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 7 & 1 == 1 && byte >> 5 & 1 == 0 {
        // 7 & !5
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: size,
                    y: -adapt(value[7], value[5], cutoff) * size,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }
    if byte >> 7 & 1 == 1 && byte >> 3 & 1 == 0 {
        // 7 & !3
        nodes.push(Node {
            pos: pos
                + Pos4D {
                    x: -adapt(value[7], value[3], cutoff) * size,
                    y: size,
                    z: size,
                    w: 0.0,
                },
            r: 1.0,
            color,
        })
    }

    (nodes, edges)
}
