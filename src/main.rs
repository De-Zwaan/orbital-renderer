use std::{
    f32::consts::PI,
    sync::{Arc, Mutex},
};

// Crates for window managment
use pixels::{PixelsBuilder, SurfaceTexture};
use winit::{
    dpi::PhysicalSize,
    error::EventLoopError,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

// Actual rendering code
use n_renderer::{
    pos::{RotationAxis, RotationPlane}, projection::{Projection, ProjectionType}, render::Screen, transform::Transform
};
use orbital_renderer::orbital::create_orbital;

const WIDTH: usize = 800;
const HEIGHT: usize = 800;

const SCALE: f32 = 10.0;

fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();

    // Initialise the window
    let window = WindowBuilder::new()
        .with_title("Spinny Spinny")
        .with_resizable(false)
        .with_inner_size(PhysicalSize::new(WIDTH as u32, HEIGHT as u32))
        .build(&event_loop)
        .unwrap();

    // Create a surface texture to render to
    let surface_texture = SurfaceTexture::new(
        window.inner_size().width,
        window.inner_size().height,
        &window,
    );

    // Create a pixelarray
    let mut pixels: pixels::Pixels =
        PixelsBuilder::new(WIDTH as u32, HEIGHT as u32, surface_texture)
            .build()
            .unwrap();

    // Create a pixelbuffer
    let screen = Arc::new(Mutex::new(Screen::new(WIDTH, HEIGHT)));

    // Generate shape
    let shape = &create_orbital((3, 2, 1), 50, 1.5, 0.03).scale(0.15);
    // let empty = &empty_3d().scale(0.1);

    let mut t = 0.0;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                // println!("Window closed");
                control_flow.exit();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                // println!("Window resized");
                let _ = pixels.resize_buffer(new_size.width, new_size.height);
                let _ = pixels.resize_surface(new_size.width, new_size.height);
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                {
                    let mut screen_lock = screen.lock().unwrap();
                    screen_lock.clear();
                }

                t += 0.1;

                // Transform the object
                let rotated_shape = shape.rotate(RotationPlane::get_rot_mat_3d(
                    RotationAxis::Z,
                    PI / 32.0 * t,
                ));

                // let rotated_empty = empty.rotate(RotationPlane::get_rot_mat_3d(
                //     RotationAxis::Y,
                //     PI / 32.0 * t,
                // ));

                // Draw the object
                rotated_shape.draw(
                    Arc::clone(&screen),
                    Projection::new(ProjectionType::Perspective, 1.0 / SCALE),
                );

                // rotated_empty.draw(
                //     Arc::clone(&screen),
                //     Projection::new(ProjectionType::Perspective, 1.0 / SCALE),
                // );

                {
                    let screen_lock = screen.lock().unwrap();
                    let screen_slice = screen_lock.get_slice();
                    pixels.frame_mut().copy_from_slice(screen_slice);
                }

                // Display the result on the screen
                if pixels
                    .render()
                    .map_err(|e| println!("pixels.render() failed: {}", e))
                    .is_err()
                {
                    control_flow.exit();
                };
            }
            _ => (),
        }
    })
}
