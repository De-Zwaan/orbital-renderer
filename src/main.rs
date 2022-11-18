#![windows_subsystem = "windows"]
pub mod matrix;
pub mod pos;
pub mod projection;
pub mod shapes;

use pixels::{Error, PixelsBuilder, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use crate::pos::*;

#[allow(unused_imports)]
use shapes::{create_3_cube, create_3_sphere, create_4_cube, create_4_sphere, empty, Object};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 1000;

const SCALE: f64 = 200.0;

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();

    // Initialise the window
    let window = WindowBuilder::new()
        .with_title("Spinny Spinny")
        // .with_decorations(false)
        // .with_transparent(true)
        .with_always_on_top(true)
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .with_min_inner_size(LogicalSize::new(100, 100))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    // Create a surface texture to render to
    let surface_texture = SurfaceTexture::new(
        window.inner_size().width,
        window.inner_size().height,
        &window,
    );

    // Create a pixelarray
    let mut pixels: pixels::Pixels = PixelsBuilder::new(WIDTH, HEIGHT, surface_texture).build()?;

    let mut t: u64 = 0;

    let shape = create_3_cube(1.0);
    // let shape = create_4_cube(1.0);
    // let shape = create_3_sphere(1000);
    // let shape = create_4_sphere(3200, 1.8);
    // let shape = empty();

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                // println!("Window closed");
                control_flow.set_exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                // println!("Window resized");
                pixels.resize_buffer(new_size.width, new_size.height);
                pixels.resize_surface(new_size.width, new_size.height);
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                t += 1;

                let screen = pixels.get_frame();

                for (_i, p) in screen.chunks_exact_mut(4).enumerate() {
                    p.copy_from_slice(&[0x00, 0x00, 0x00, 0xff]);
                }

                // Draw objects
                shape.draw(screen, window.inner_size(), t);

                // Render result
                if pixels
                    .render()
                    .map_err(|e| println!("pixels.render() failed: {}", e))
                    .is_err()
                {
                    control_flow.set_exit();
                };
            }
            _ => (),
        }
    })
}
