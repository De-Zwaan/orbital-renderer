use std::f64::consts::PI;
use std::vec;
use winit::dpi::PhysicalSize;

use crate::pos::*;
use crate::projection::Projection;
use crate::shapes::Color::*;
use crate::{matrix::*, print_coord_in_pixelbuffer};

#[derive(Clone, Copy)]
pub enum Color {
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Purple,
    White,
    Black,
    RGBA(u8, u8, u8, u8),
    RGB(u8, u8, u8),
}

impl Color {
    fn get_rgba(&self) -> [u8; 4] {
        match self {
            Color::Red => [0xff, 0x00, 0x00, 0xff],
            Color::Orange => [0xff, 0xaa, 0x00, 0xff],
            Color::Yellow => [0xaa, 0xaa, 0x00, 0xff],
            Color::Green => [0x00, 0xff, 0x00, 0xff],
            Color::Blue => [0x00, 0x00, 0xff, 0xff],
            Color::Purple => [0xaa, 0x00, 0xaa, 0xff],
            Color::White => [0xff, 0xff, 0xff, 0xff],
            Color::Black => [0x00, 0x00, 0x00, 0xff],
            Color::RGBA(r, g, b, a) => [*r, *g, *b, *a],
            Color::RGB(r, g, b) => [*r, *g, *b, 0xff],
        }
    }
}

pub struct Object {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,
}

#[derive(Clone, Copy)]
pub struct Node {
    pub pos: Pos4D,
    pub r: f64,
    pub color: Color,
}

#[derive(Clone, Copy)]
pub struct Edge {
    pub start_node: Pos4D,
    pub end_node: Pos4D,
    pub r: f64,
    pub color: Color,
}

#[derive(Clone, Copy)]
pub struct Face {
    pub node_a: Pos4D,
    pub node_b: Pos4D,
    pub node_c: Pos4D,
    pub r: f64,
    pub color: Color,
}

impl Object {
    /// Draw all edges, vertices and faces of the object
    pub fn draw(
        &self,
        screen: &mut [u8],
        size: PhysicalSize<u32>,
        projection_scale: f64,
        projection: Projection,
    ) {
        // Iterate over all edges, vertices and faces of the object and draw them
        self.edges.iter().for_each(|edge| {
            edge.draw(screen, size, projection, projection_scale);
        });

        self.nodes.iter().for_each(|node| {
            node.draw(screen, size, projection, projection_scale);
        });

        self.faces.iter().for_each(|face| {
            face.draw(screen, size, projection, projection_scale);
        });
    }
}

/// Trivial object transformations
pub trait Transform<T, V>
where
    Self: Sized,
{
    /// Rotate an object using a rotation matrix
    fn rotate_copy(&self, rotation_matrix: T) -> Self;

    /// Move an object using a vector
    fn move_copy(&self, vector: V) -> Self;

    /// Scale an object using a 1D scalar
    fn scale_copy(&self, scalar: f64) -> Self;
}

impl Transform<Matrix4x4, Pos4D> for Object {
    fn rotate_copy(&self, rotation_matrix: Matrix4x4) -> Self {
        let mut edges: Vec<Edge> = Vec::new();
        // Loop over all edges
        for edge in self.edges.iter() {
            edges.push(edge.rotate_copy(rotation_matrix));
        }

        let mut nodes: Vec<Node> = Vec::new();
        // Loop over all nodes
        for node in self.nodes.iter() {
            nodes.push(node.rotate_copy(rotation_matrix));
        }

        let mut faces: Vec<Face> = Vec::new();
        // Loop over all faces
        for face in self.faces.iter() {
            faces.push(face.rotate_copy(rotation_matrix));
        }

        Self {
            nodes,
            edges,
            faces,
        }
    }

    fn move_copy(&self, vector: Pos4D) -> Self {
        let mut edges: Vec<Edge> = Vec::new();
        // Loop over all edges
        for (_i, edge) in self.edges.iter().enumerate() {
            edges.push(edge.move_copy(vector));
        }

        let mut nodes: Vec<Node> = Vec::new();
        // Loop over all nodes
        for (_i, node) in self.nodes.iter().enumerate() {
            nodes.push(node.move_copy(vector));
        }

        let mut faces: Vec<Face> = Vec::new();
        // Loop over all faces
        for (_i, face) in self.faces.iter().enumerate() {
            faces.push(face.move_copy(vector));
        }

        Self {
            nodes,
            edges,
            faces,
        }
    }

    fn scale_copy(&self, scale: f64) -> Self {
        let mut edges: Vec<Edge> = Vec::new();
        // Loop over all edges
        for (_i, edge) in self.edges.iter().enumerate() {
            edges.push(edge.scale_copy(scale));
        }

        let mut nodes: Vec<Node> = Vec::new();
        // Loop over all nodes
        for (_i, node) in self.nodes.iter().enumerate() {
            nodes.push(node.scale_copy(scale));
        }

        let mut faces: Vec<Face> = Vec::new();
        // Loop over all faces
        for (_i, face) in self.faces.iter().enumerate() {
            faces.push(face.scale_copy(scale));
        }

        Self {
            nodes,
            edges,
            faces,
        }
    }
}

impl Transform<Matrix4x4, Pos4D> for Node {
    fn rotate_copy(&self, rotation_matrix: Matrix4x4) -> Self {
        Self {
            pos: rotation_matrix * self.pos,
            r: self.r,
            color: self.color,
        }
    }

    fn move_copy(&self, vector: Pos4D) -> Self {
        Self {
            pos: self.pos + vector,
            r: self.r,
            color: self.color,
        }
    }

    fn scale_copy(&self, scale: f64) -> Self {
        Self {
            pos: self.pos * scale,
            r: self.r,
            color: self.color,
        }
    }
}

impl Transform<Matrix4x4, Pos4D> for Edge {
    fn rotate_copy(&self, rotation_matrix: Matrix4x4) -> Self {
        let start_node: Pos4D = rotation_matrix * self.start_node;
        let end_node: Pos4D = rotation_matrix * self.end_node;

        Self {
            start_node,
            end_node,
            r: self.r,
            color: self.color,
        }
    }

    fn move_copy(&self, vector: Pos4D) -> Self {
        let start_node: Pos4D = self.start_node + vector;
        let end_node: Pos4D = self.end_node + vector;

        Self {
            start_node,
            end_node,
            r: self.r,
            color: self.color,
        }
    }

    fn scale_copy(&self, scale: f64) -> Self {
        let start_node: Pos4D = self.start_node * scale;
        let end_node: Pos4D = self.end_node * scale;

        Self {
            start_node,
            end_node,
            r: self.r,
            color: self.color,
        }
    }
}

impl Transform<Matrix4x4, Pos4D> for Face {
    fn rotate_copy(&self, rotation_matrix: Matrix4x4) -> Self {
        let node_a: Pos4D = rotation_matrix * self.node_a;
        let node_b: Pos4D = rotation_matrix * self.node_b;
        let node_c: Pos4D = rotation_matrix * self.node_c;

        Self {
            node_a,
            node_b,
            node_c,
            r: self.r,
            color: self.color,
        }
    }

    fn move_copy(&self, vector: Pos4D) -> Self {
        let node_a: Pos4D = self.node_a + vector;
        let node_b: Pos4D = self.node_b + vector;
        let node_c: Pos4D = self.node_c + vector;

        Self {
            node_a,
            node_b,
            node_c,
            r: self.r,
            color: self.color,
        }
    }

    fn scale_copy(&self, scale: f64) -> Self {
        let node_a: Pos4D = self.node_a * scale;
        let node_b: Pos4D = self.node_b * scale;
        let node_c: Pos4D = self.node_c * scale;

        Self {
            node_a,
            node_b,
            node_c,
            r: self.r,
            color: self.color,
        }
    }
}

trait Render {
    /// Determine the screen coordinates of objects using certain transformations and insert them into the pixelbuffer
    fn draw(
        &self,
        screen: &mut [u8],
        size: PhysicalSize<u32>,
        projection: Projection,
        projection_scale: f64,
    );

    /// Print a point to the screen with a certain y(square) radius
    fn print_point(
        x: i32,
        y: i32,
        r: i32,
        screen: &mut [u8],
        size: PhysicalSize<u32>,
        color: [u8; 4],
    ) {
        for x_off in -r..=r {
            for y_off in -r..=r {
                let x_p = x + x_off;
                let y_p = y + y_off;

                print_coord_in_pixelbuffer(x_p, y_p, screen, size, color)
            }
        }
    }
}

fn scale(pos: Pos4D, to_camera: Pos3D) -> f64 {
    // Find the angle between the origin to the point and the origin to the camera
    let pos_3d: Pos3D = Pos3D {
        x: pos.x,
        y: pos.y,
        z: pos.z,
    };

    // The smaller the angle to the camera, the larger the nodes are when drawn to the screen
    // This is meant to simulate a kind of reflection, where the light is large and behind the camera
    (to_camera * (1.0 / to_camera.len())) >> ((pos_3d * (1.0 / pos_3d.len())) * 2.0)
}

impl Render for Node {
    fn draw(
        &self,
        screen: &mut [u8],
        size: PhysicalSize<u32>,
        projection: Projection,
        projection_scale: f64,
    ) {
        if self.r as i32 <= 0 {
            return;
        };
        // if self.pos.w != self.pos.w.clamp(0.9, 1.1) {return};

        // Transform the Node to screen coordinates
        let pos: Pos2D = projection.project(self.pos, size, projection_scale);

        let r = scale(self.pos, projection.get_camera_pos()) * self.r;

        // Set the color of the points
        let rgba = self.color.get_rgba();
        // rgba[2] = (50.0 * (self.pos.w + 2.5)) as u8;

        // Draw small cubes around the point
        if r < 0.4 {
            return;
        };

        Self::print_point(pos.x as i32, pos.y as i32, r as i32, screen, size, rgba);
    }
}

impl Render for Edge {
    fn draw(
        &self,
        screen: &mut [u8],
        size: PhysicalSize<u32>,
        projection: Projection,
        projection_scale: f64,
    ) {
        if self.r as i32 <= 0 {
            return;
        };

        // Calculate the screen coordinates of the start and end points
        let start_point: Pos2D = projection.project(self.start_node, size, projection_scale);
        let end_point: Pos2D = projection.project(self.end_node, size, projection_scale);

        // Calculate vector for line connecting start and end point
        let edge = { [end_point.x - start_point.x, end_point.y - start_point.y] };

        let to_camera = projection.get_camera_pos();

        // Calculate the radius of the start and end points of the edge
        let start_point_r = scale(self.start_node, to_camera) * self.r;
        let end_point_r = scale(self.end_node, to_camera) * self.r;

        // Set 1 / the amount of points that compose an edge
        let resolution: f64 = 0.01;

        let mut rgba = self.color.get_rgba();

        for i in 0..=((1.0 / resolution) as i32) {
            let x_p = (edge[0] * i as f64 * resolution) as i32 + start_point.x as i32;
            let y_p = (edge[1] * i as f64 * resolution) as i32 + start_point.y as i32;

            // Interpolate the radius of the points making up the edges
            let r =
                (((end_point_r - start_point_r) * i as f64 * resolution) + start_point_r) as i32;

            // Change the blue channel of the edge based on the w coordiante
            rgba[2] = (50.0
                * (((self.end_node.w - self.start_node.w) * i as f64 * resolution
                    + self.start_node.w)
                    + 2.5)) as u8;

            Self::print_point(x_p, y_p, r, screen, size, rgba);
        }
    }
}

impl Render for Face {
    fn draw(
        &self,
        screen: &mut [u8],
        size: PhysicalSize<u32>,
        projection: Projection,
        projection_scale: f64,
    ) {
        let vector_a =
            projection.project_to_3d(self.node_b) + projection.project_to_3d(self.node_a) * -1.0;
        let vector_b =
            projection.project_to_3d(self.node_c) + projection.project_to_3d(self.node_a) * -1.0;

        // Get the normal vector of the surface by taking the cross product and normalise to a
        // length of 1
        let normal: Pos3D = vector_a ^ vector_b;
        let n_normal: Pos3D = normal * (1.0 / normal.len());

        let to_camera: Pos3D = Pos3D {
            x: -5.0,
            y: 5.0,
            z: -5.0,
        };

        // Let the brightness depend on the angle between the normal and the camera path
        // 1 if staight on, 0 if perpendicular and -1 if facing opposite
        let angle_to_camera: f64 = n_normal >> (to_camera * (1.0 / to_camera.len()));

        // Get the locations of the three nodes of the triangle
        let pos_a: Pos2D = projection.project(self.node_a, size, projection_scale);
        let pos_b: Pos2D = projection.project(self.node_b, size, projection_scale);
        let pos_c: Pos2D = projection.project(self.node_c, size, projection_scale);

        // Calculate 2d vectors between the points on the screen
        let a_to_b: Pos2D = pos_b + (pos_a * -1.0);
        let a_to_c: Pos2D = pos_c + (pos_a * -1.0);

        let mut rgba: [u8; 4] = self.color.get_rgba();

        // Change the alpha channel based on the angle between the camera and the surface
        rgba[2] = 255_u8 - (255.0 * angle_to_camera.clamp(0.0, 1.0)) as u8;

        let resolution: f64 = 0.1;

        let mut t: Vec<Pos2D> = Vec::new();

        const G: f64 = 1.0 / 1.32471795572;
        static ALPHA: Pos2D = Pos2D { x: G, y: G * G };

        for n in 1..((1.0 / resolution) as i32) {
            t.push(ALPHA * n as f64)
        }

        // http://extremelearning.com.au/evenly-distributing-points-in-a-triangle/
        // for (_, p) in t.iter().enumerate() {
        //     let mut pos: Pos2D = Pos2D { x: 0.0, y: 0.0 };
        //     if p.x + p.y < 1.0 {
        //         pos = pos_a + (Pos2D { x: 1.0, y: 0.0 } * p.x * 10.0) + (Pos2D { x: 1.0, y: 1.0 } * p.y * 10.0);
        //     } else {
        //         pos = pos_a + (Pos2D { x: 1.0, y: 1.0 } * (1.0 - p.x * 10.0)) + (Pos2D { x: 1.0, y: 1.0 } * (1.0 - p.y * 10.0));
        //     }
        //     Self::print_point(pos.x as i32, pos.y as i32, self.r as i32, screen, size, rgba)
        // }

        // Iterate over points on the surface of the face and print them to the screen
        for k1 in 1..((1.0 / resolution) as i32) {
            for k2 in 1..((1.0 / resolution) as i32) {
                if k1 as f64 * resolution + k2 as f64 * resolution > 1.0 {
                    break;
                }
                if angle_to_camera < 0.0 {
                    return;
                }

                let p =
                    pos_a + a_to_b * (k1 as f64 * resolution) + a_to_c * (k2 as f64 * resolution);

                Self::print_point(p.x as i32, p.y as i32, self.r as i32, screen, size, rgba);
            }
        }
    }
}

pub fn empty() -> Object {
    let nodes: Vec<Node> = vec![
        Node {
            pos: Pos4D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
            r: 1.0,
            color: White,
        },
        Node {
            pos: Pos4D {
                x: 1.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
            r: 1.0,
            color: Red,
        },
        Node {
            pos: Pos4D {
                x: 0.0,
                y: 1.0,
                z: 0.0,
                w: 0.0,
            },
            r: 1.0,
            color: Green,
        },
        Node {
            pos: Pos4D {
                x: 0.0,
                y: 0.0,
                z: 1.0,
                w: 0.0,
            },
            r: 1.0,
            color: Blue,
        },
        Node {
            pos: Pos4D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            r: 1.0,
            color: Purple,
        },
    ];

    let edges: Vec<Edge> = Vec::new();
    let faces: Vec<Face> = Vec::new();

    Object {
        nodes,
        edges,
        faces,
    }
}

pub fn create_3_cube(r: f64) -> Object {
    let points: [Pos4D; 8] = [
        Pos4D {
            x: r * -1.0,
            y: r * -1.0,
            z: r * -1.0,
            w: r * 0.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * -1.0,
            z: r * 1.0,
            w: r * 0.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * 1.0,
            z: r * -1.0,
            w: r * 0.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * 1.0,
            z: r * 1.0,
            w: r * 0.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * -1.0,
            z: r * -1.0,
            w: r * 0.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * -1.0,
            z: r * 1.0,
            w: r * 0.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * 1.0,
            z: r * -1.0,
            w: r * 0.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * 1.0,
            z: r * 1.0,
            w: r * 0.0,
        },
    ];

    Object {
        nodes: vec![
            Node {
                pos: points[0],
                r: 0.1,
                color: Yellow,
            },
            Node {
                pos: points[1],
                r: 0.1,
                color: Yellow,
            },
            Node {
                pos: points[2],
                r: 0.1,
                color: Yellow,
            },
            Node {
                pos: points[3],
                r: 0.1,
                color: Yellow,
            },
            Node {
                pos: points[4],
                r: 0.1,
                color: Yellow,
            },
            Node {
                pos: points[5],
                r: 0.1,
                color: Yellow,
            },
            Node {
                pos: points[6],
                r: 0.1,
                color: Yellow,
            },
            Node {
                pos: points[7],
                r: 0.1,
                color: Yellow,
            },
        ],
        edges: vec![
            Edge {
                start_node: points[0],
                end_node: points[1],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[0],
                end_node: points[2],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[0],
                end_node: points[4],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[3],
                end_node: points[1],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[3],
                end_node: points[2],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[3],
                end_node: points[7],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[5],
                end_node: points[1],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[5],
                end_node: points[4],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[5],
                end_node: points[7],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[6],
                end_node: points[2],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[6],
                end_node: points[4],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[6],
                end_node: points[7],
                r: 1.0,
                color: Purple,
            },
        ],
        faces: vec![
            Face {
                node_a: points[0],
                node_b: points[1],
                node_c: points[4],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[0],
                node_b: points[4],
                node_c: points[2],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[0],
                node_b: points[2],
                node_c: points[1],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[3],
                node_b: points[1],
                node_c: points[2],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[3],
                node_b: points[2],
                node_c: points[7],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[3],
                node_b: points[7],
                node_c: points[1],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[5],
                node_b: points[1],
                node_c: points[7],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[5],
                node_b: points[4],
                node_c: points[1],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[5],
                node_b: points[7],
                node_c: points[4],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[6],
                node_b: points[2],
                node_c: points[4],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[6],
                node_b: points[4],
                node_c: points[7],
                r: 1.0,
                color: Yellow,
            },
            Face {
                node_a: points[6],
                node_b: points[7],
                node_c: points[2],
                r: 1.0,
                color: Yellow,
            },
        ],
    }
}

pub fn create_4_cube(r: f64) -> Object {
    let points = [
        Pos4D {
            x: r * -1.0,
            y: r * -1.0,
            z: r * -1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * -1.0,
            z: r * -1.0,
            w: r * 1.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * -1.0,
            z: r * 1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * -1.0,
            z: r * 1.0,
            w: r * 1.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * 1.0,
            z: r * -1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * 1.0,
            z: r * -1.0,
            w: r * 1.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * 1.0,
            z: r * 1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * -1.0,
            y: r * 1.0,
            z: r * 1.0,
            w: r * 1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * -1.0,
            z: r * -1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * -1.0,
            z: r * -1.0,
            w: r * 1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * -1.0,
            z: r * 1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * -1.0,
            z: r * 1.0,
            w: r * 1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * 1.0,
            z: r * -1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * 1.0,
            z: r * -1.0,
            w: r * 1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * 1.0,
            z: r * 1.0,
            w: r * -1.0,
        },
        Pos4D {
            x: r * 1.0,
            y: r * 1.0,
            z: r * 1.0,
            w: r * 1.0,
        },
    ];

    Object {
        nodes: vec![
            Node {
                pos: points[0],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[1],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[2],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[3],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[4],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[5],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[6],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[7],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[8],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[9],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[10],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[11],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[12],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[13],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[14],
                r: 1.0,
                color: White,
            },
            Node {
                pos: points[15],
                r: 1.0,
                color: White,
            },
        ],
        edges: vec![
            Edge {
                start_node: points[0],
                end_node: points[1],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[0],
                end_node: points[2],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[0],
                end_node: points[4],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[0],
                end_node: points[8],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[3],
                end_node: points[1],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[3],
                end_node: points[2],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[3],
                end_node: points[7],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[3],
                end_node: points[11],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[5],
                end_node: points[1],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[5],
                end_node: points[4],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[5],
                end_node: points[7],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[5],
                end_node: points[13],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[6],
                end_node: points[2],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[6],
                end_node: points[4],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[6],
                end_node: points[7],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[6],
                end_node: points[14],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[9],
                end_node: points[1],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[9],
                end_node: points[8],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[9],
                end_node: points[11],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[9],
                end_node: points[13],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[10],
                end_node: points[2],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[10],
                end_node: points[8],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[10],
                end_node: points[11],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[10],
                end_node: points[14],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[12],
                end_node: points[4],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[12],
                end_node: points[8],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[12],
                end_node: points[13],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[12],
                end_node: points[14],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[15],
                end_node: points[7],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[15],
                end_node: points[11],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[15],
                end_node: points[13],
                r: 1.0,
                color: Purple,
            },
            Edge {
                start_node: points[15],
                end_node: points[14],
                r: 1.0,
                color: Purple,
            },
        ],
        faces: Vec::new(),
    }
}

pub fn create_3_sphere(res: i32) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let edges: Vec<Edge> = Vec::new();
    let faces: Vec<Face> = Vec::new();

    let phi = PI * (3.0 - (5.0_f64).sqrt());

    for i in 0..res {
        let y = 1.0 - (i as f64 / (res - 1) as f64) * 2.0;
        let r = (1.0 - y * y).sqrt();

        let theta = phi * i as f64;

        let x = theta.cos() * r;
        let z = theta.sin() * r;

        nodes.push(Node {
            pos: Pos4D { x, y, z, w: 0.0 },
            r: 1.0,
            color: Purple,
        })
    }

    Object {
        nodes,
        edges,
        faces,
    }
}

pub fn create_4_sphere(res: i32, r: f64) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let edges: Vec<Edge> = Vec::new();
    let faces: Vec<Face> = Vec::new();

    let res_per_plane = (res as f64).sqrt() as i32;

    // XZ plane
    for i in 0..res_per_plane {
        let cos_t: f64 = ((2.0 * PI) / res_per_plane as f64 * i as f64).cos();
        let sin_t: f64 = ((2.0 * PI) / res_per_plane as f64 * i as f64).sin();

        // rotating Z plane
        for j in 0..res_per_plane {
            let cos_r: f64 = ((2.0 * PI) / res_per_plane as f64 * j as f64).cos();
            let sin_r: f64 = ((2.0 * PI) / res_per_plane as f64 * j as f64).sin();

            // rotating W plane
            for k in 0..res_per_plane {
                let cos_s: f64 = ((2.0 * PI) / res_per_plane as f64 * k as f64).cos();
                let sin_s: f64 = ((2.0 * PI) / res_per_plane as f64 * k as f64).sin();

                let x: f64 = r * sin_t * sin_r * cos_s;
                let z: f64 = r * sin_t * sin_r * sin_s;

                let y: f64 = r * sin_t * cos_r;
                let w: f64 = r * cos_t;

                let pos = Pos4D { x, y, z, w };

                nodes.push(Node {
                    pos,
                    r: 1.0,
                    color: Purple,
                });
            }
        }
    }

    Object {
        nodes,
        edges,
        faces,
    }
}

pub fn create_torus(res: i32, r: f64) -> Object {
    let mut nodes: Vec<Node> = Vec::new();
    let edges: Vec<Edge> = Vec::new();
    let faces: Vec<Face> = Vec::new();

    let major_r: f64 = r;
    let minor_r: f64 = 0.5 * r;

    // XZ plane
    for t in 0..res {
        let cos_t: f64 = ((2.0 * PI) / res as f64 * t as f64).cos();
        let sin_t: f64 = ((2.0 * PI) / res as f64 * t as f64).sin();

        for p in 0..res {
            let cos_p: f64 = ((2.0 * PI) / res as f64 * p as f64).cos();
            let sin_p: f64 = ((2.0 * PI) / res as f64 * p as f64).sin();

            let x = (major_r + minor_r * cos_t) * sin_p;
            let y = (major_r + minor_r * cos_t) * cos_p;
            let z = minor_r * sin_t;
            let w = 0.0;

            let pos = Pos4D { x, y, z, w };

            nodes.push(Node {
                pos,
                r: 1.0,
                color: Purple,
            });
        }
    }

    Object {
        nodes,
        edges,
        faces,
    }
}
