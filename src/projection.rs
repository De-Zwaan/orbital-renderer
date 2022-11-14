use winit::dpi::PhysicalSize;

use crate::SCALE;
use crate::projection::Projection::*;
use crate::pos::*;
use crate::matrix::*;

#[derive(Clone, Copy)]
pub enum Projection {
    Perspective,
    Stereographic,
}

impl Projection {
    pub fn project(&self, pos: Pos4D, size: PhysicalSize<u32>) -> Pos2D {
        static SCREEN_MATRIX_3D: Matrix2x3 = Matrix2x3 {
            x: Pos3D { x:  0.866, y:  0.0, z: -0.866 },
            y: Pos3D { x: -0.5,   y:  -1.0, z: -0.5  },
        };
        
        match self {
            Perspective => {
                let scale = 2.0;
                let bound = size.width.min(size.height) as f64 / 2.0;
                let zratio = 0.9 + (pos.z / scale) * 0.3;

                Pos2D { 
                    x: (size.width as f64  / 2.0 + zratio * bound * (pos.x / scale)).floor(), 
                    y: (size.height as f64 / 2.0 - zratio * bound * (pos.y / scale)).floor(),
                }
            },
            Stereographic => {
                let pos_3d = Pos3D {
                    x: (pos.x / (2.0 + pos.w)), 
                    y: (pos.y / (2.0 + pos.w)), 
                    z: (pos.z / (2.0 + pos.w)),
                };

                (pos_3d * SCALE).to_screen_coords(SCREEN_MATRIX_3D, size)
            }
        }
    }
}


