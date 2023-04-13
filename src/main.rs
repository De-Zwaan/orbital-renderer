use std::f64::consts::PI;

// Crates for window managment
use pixels::{Error, PixelsBuilder, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

// Actual rendering code
use n_renderer::{render::{Object, Transform}, pos::{RotationPlane}, projection::Projection::*};
use orbital_renderer::orbital::create_orbital;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;

const SCALE: f64 = 200.0;

fn main() -> Result<(), Error> {
    let event_loop = EventLoop::new();

    // Initialise the window
    let window = WindowBuilder::new()
        .with_title("Spinny Spinny")
        // .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top(true)
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
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
    
    let mut shape: Object = create_orbital(100, 0.1, 5.0, 0.05, (4, 3, 1));

    shape.scale(2.0);

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
                event: WindowEvent::Resized(_),
                ..
            } => {
                // println!("Window resized");
                // pixels.resize_buffer(new_size.width, new_size.height);
                // pixels.resize_surface(new_size.width, new_size.height);
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {

                let screen = pixels.get_frame();

                // Create an empty pixelbuffer to render to
                screen.chunks_exact_mut(4).for_each(|p| {
                    p.copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                });

                // Transform the object
                shape.rotate(RotationPlane::get_rot_mat_4d(RotationPlane::XZ, PI / 256.0));

                // Draw the object
                shape.draw(
                    screen,
                    window.inner_size(),
                    SCALE,
                    Perspective,
                );

                // Display the result on the screen
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
