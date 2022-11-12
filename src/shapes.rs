use std::f64::consts::PI;

use rand_distr::{Normal, Distribution};

use crate::{Object, pos::{Pos4D, Len}, Node, Edge};

pub fn create_3_cube() -> Object {
    let points = [
        Pos4D { x: -1.0, y: -1.0, z: -1.0, w:  0.0},
        Pos4D { x: -1.0, y: -1.0, z:  1.0, w:  0.0},
        Pos4D { x: -1.0, y:  1.0, z: -1.0, w:  0.0},
        Pos4D { x: -1.0, y:  1.0, z:  1.0, w:  0.0},
        Pos4D { x:  1.0, y: -1.0, z: -1.0, w:  0.0},
        Pos4D { x:  1.0, y: -1.0, z:  1.0, w:  0.0},
        Pos4D { x:  1.0, y:  1.0, z: -1.0, w:  0.0},
        Pos4D { x:  1.0, y:  1.0, z:  1.0, w:  0.0},
    ];

    Object {
        nodes: vec![
            Node { pos: points[00], r: 0.1 },
            Node { pos: points[01], r: 0.1 },
            Node { pos: points[02], r: 0.1 },
            Node { pos: points[03], r: 0.1 },
            Node { pos: points[04], r: 0.1 },
            Node { pos: points[05], r: 0.1 },
            Node { pos: points[06], r: 0.1 },
            Node { pos: points[07], r: 0.1 },
        ],
        edges: vec![
            Edge {start_node: points[00], end_node: points[01], r: 0.01},
            Edge {start_node: points[00], end_node: points[02], r: 0.01},
            Edge {start_node: points[00], end_node: points[04], r: 0.01},

            Edge {start_node: points[03], end_node: points[01], r: 0.01},
            Edge {start_node: points[03], end_node: points[02], r: 0.01},
            Edge {start_node: points[03], end_node: points[07], r: 0.01},

            Edge {start_node: points[05], end_node: points[01], r: 0.01},
            Edge {start_node: points[05], end_node: points[04], r: 0.01},
            Edge {start_node: points[05], end_node: points[07], r: 0.01},

            Edge {start_node: points[06], end_node: points[02], r: 0.01},
            Edge {start_node: points[06], end_node: points[04], r: 0.01},
            Edge {start_node: points[06], end_node: points[07], r: 0.01},
        ],
    }
}

pub fn create_4_cube() -> Object {
    let points = [
        Pos4D { x: -1.0, y: -1.0, z: -1.0, w: -1.0},
        Pos4D { x: -1.0, y: -1.0, z: -1.0, w:  1.0},
        Pos4D { x: -1.0, y: -1.0, z:  1.0, w: -1.0},
        Pos4D { x: -1.0, y: -1.0, z:  1.0, w:  1.0},
        Pos4D { x: -1.0, y:  1.0, z: -1.0, w: -1.0},
        Pos4D { x: -1.0, y:  1.0, z: -1.0, w:  1.0},
        Pos4D { x: -1.0, y:  1.0, z:  1.0, w: -1.0},
        Pos4D { x: -1.0, y:  1.0, z:  1.0, w:  1.0},
        Pos4D { x:  1.0, y: -1.0, z: -1.0, w: -1.0},
        Pos4D { x:  1.0, y: -1.0, z: -1.0, w:  1.0},
        Pos4D { x:  1.0, y: -1.0, z:  1.0, w: -1.0},
        Pos4D { x:  1.0, y: -1.0, z:  1.0, w:  1.0},
        Pos4D { x:  1.0, y:  1.0, z: -1.0, w: -1.0},
        Pos4D { x:  1.0, y:  1.0, z: -1.0, w:  1.0},
        Pos4D { x:  1.0, y:  1.0, z:  1.0, w: -1.0},
        Pos4D { x:  1.0, y:  1.0, z:  1.0, w:  1.0},
    ];

    Object {
        nodes: vec![
            Node { pos: points[00], r: 0.1 },
            Node { pos: points[01], r: 0.1 },
            Node { pos: points[02], r: 0.1 },
            Node { pos: points[03], r: 0.1 },
            Node { pos: points[04], r: 0.1 },
            Node { pos: points[05], r: 0.1 },
            Node { pos: points[06], r: 0.1 },
            Node { pos: points[07], r: 0.1 },
            Node { pos: points[08], r: 0.1 },
            Node { pos: points[09], r: 0.1 },
            Node { pos: points[10], r: 0.1 },
            Node { pos: points[11], r: 0.1 },
            Node { pos: points[12], r: 0.1 },
            Node { pos: points[13], r: 0.1 },
            Node { pos: points[14], r: 0.1 },
            Node { pos: points[15], r: 0.1 },
        ],
        edges: vec![
            Edge {start_node: points[00], end_node: points[01], r: 0.01},
            Edge {start_node: points[00], end_node: points[02], r: 0.01},
            Edge {start_node: points[00], end_node: points[04], r: 0.01},
            Edge {start_node: points[00], end_node: points[08], r: 0.01},

            Edge {start_node: points[03], end_node: points[01], r: 0.01},
            Edge {start_node: points[03], end_node: points[02], r: 0.01},
            Edge {start_node: points[03], end_node: points[07], r: 0.01},
            Edge {start_node: points[03], end_node: points[11], r: 0.01},

            Edge {start_node: points[05], end_node: points[01], r: 0.01},
            Edge {start_node: points[05], end_node: points[04], r: 0.01},
            Edge {start_node: points[05], end_node: points[07], r: 0.01},
            Edge {start_node: points[05], end_node: points[13], r: 0.01},

            Edge {start_node: points[06], end_node: points[02], r: 0.01},
            Edge {start_node: points[06], end_node: points[04], r: 0.01},
            Edge {start_node: points[06], end_node: points[07], r: 0.01},
            Edge {start_node: points[06], end_node: points[14], r: 0.01},

            Edge {start_node: points[09], end_node: points[01], r: 0.01},
            Edge {start_node: points[09], end_node: points[08], r: 0.01},
            Edge {start_node: points[09], end_node: points[11], r: 0.01},
            Edge {start_node: points[09], end_node: points[13], r: 0.01},

            Edge {start_node: points[10], end_node: points[02], r: 0.01},
            Edge {start_node: points[10], end_node: points[08], r: 0.01},
            Edge {start_node: points[10], end_node: points[11], r: 0.01},
            Edge {start_node: points[10], end_node: points[14], r: 0.01},

            Edge {start_node: points[12], end_node: points[04], r: 0.01},
            Edge {start_node: points[12], end_node: points[08], r: 0.01},
            Edge {start_node: points[12], end_node: points[13], r: 0.01},
            Edge {start_node: points[12], end_node: points[14], r: 0.01},

            Edge {start_node: points[15], end_node: points[07], r: 0.01},
            Edge {start_node: points[15], end_node: points[11], r: 0.01},
            Edge {start_node: points[15], end_node: points[13], r: 0.01},
            Edge {start_node: points[15], end_node: points[14], r: 0.01},
        ],
    }
}

pub fn create_3_sphere(res: i32) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let edges: Vec<Edge> = Vec::new();

    let phi = PI * (3.0 - (5.0_f64).sqrt());

    for i in 0..res {
        let y = 1.0 - (i as f64 / (res - 1) as f64) * 2.0;
        let r = (1.0 - y * y).sqrt();

        let theta = phi * i as f64;

        let x = theta.cos() * r;
        let z = theta.sin() * r;

        nodes.push(Node { pos: Pos4D { x, y, z, w: 0.0 }, r: 1.0 })
    }

    // for (i, _) in nodes.iter().enumerate().step_by(2) {
    //     edges.push(Edge { start_node: nodes[i].pos, end_node: nodes[i + 1].pos, r: 1.0 });
    // }

    Object { nodes, edges }
}

pub fn create_4_sphere(res: i32) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let edges: Vec<Edge> = Vec::new();

    let normal = Normal::new(0.0, 1.0).unwrap();
    
    for _ in 0..res {
        let pos = Pos4D { x: normal.sample(&mut rand::thread_rng()), y: normal.sample(&mut rand::thread_rng()), z: normal.sample(&mut rand::thread_rng()), w: normal.sample(&mut rand::thread_rng()) };
        let scaled_pos = pos * (1.0 / pos.len());

        nodes.push(Node { pos: scaled_pos, r: 1.0} );
    }

    Object { nodes, edges }
}