use std::f32::consts::PI;

pub mod complex;
pub mod lookup;

use complex::{AbsArg, Complex, Exp, Split};
use n_renderer::{object::Object, pos::Pos3D, remove_duplicates, transform::Transform};

use crate::marching_cubes::run_marching_cubes;

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
    let r_nz = if r == 0.0 { 0.0001 } else { r };
    let t = (y / r_nz).acos();
    let xz_dist = (x * x + z * z).sqrt();
    let xz_dist_nz = if xz_dist == 0.0 { 0.0001 } else { xz_dist };
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
fn radial_wave_function((n, l, _m): (i32, i32, i32), a: f32) -> impl Fn(f32, f32, f32) -> Complex {
    move |x: f32, y: f32, z: f32| -> Complex {
        let (r, _t, _p) = cartesian_to_spherical(x, y, z);

        // General radial wave function
        let prefactor = (8.0 / (n as f32 * a).powi(3) * (n - l - 1).factorial() as f32
            / (2 * n * (n + l).factorial()) as f32)
            .sqrt();
        let exponential_part = (-r / (n as f32 * a)).exp();
        let power_part = (2.0 * r).powi(l) / (n as f32 * a).powi(l);
        let laguerre_part = laguerre_polynomials(2 * l + 1, n - l - 1, (2.0 * r) / (n as f32 * a));

        Complex(
            prefactor * exponential_part * power_part * laguerre_part,
            0.0,
        )
    }
}

/// Calculates the generalised/associated laguerre polynomials (L^k_n (x))for a given k, n and x
/// Uses the Rodrigues representation: https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
pub fn laguerre_polynomials(k: i32, n: i32, x: f32) -> f32 {
    (0..=n).map(|m| {
        (-1.0_f32).powi(m) * (n + k).factorial() as f32
            / ((n - m).factorial() * (k + m).factorial() * m.factorial()) as f32
            * x.powi(m)
    }).sum()
}

fn spherical_harmonic_normalization(l: i32, m: i32) -> f32 {
    ((2.0 * l as f32 + 1.0) / (4.0 * PI) * (l - m).factorial() as f32 / (l + m).factorial() as f32).sqrt()
}

fn spherical_harmonic(l: i32, m: i32, t: f32, p: f32) -> Complex {
    assert!(-l <= m, "-l <= m <= l");
    assert!( l >= m, "-l <= m <= l");

    let norm = spherical_harmonic_normalization(l, m);
    let legendre = legendre_polynomial(l, m, t.cos());
    let phase = Complex(0.0, m as f32 * p).exp();

    norm * legendre * phase
}

fn binom(x: f32, k: i32) -> f32 {
    (1..=k).map(|i| (x + 1.0) / i as f32 - 1.0).product()
}

fn positive_legendre_polynomial(l: i32, m: i32, x: f32) -> f32 {
    (-1.0_f32).powi(m) * 2.0_f32.powi(l) * (1.0 - x * x).powi(m).powf(0.5) * (m..=l)
        .map(|k| {
            k.factorial() as f32 / (k - m).factorial() as f32
                * x.powi(k - m)
                * binom(l as f32, k) as f32
                * binom((l + k - 1) as f32 / 2.0, l) as f32
        }).sum::<f32>()
}

fn legendre_polynomial(l: i32, m: i32, x: f32) -> f32 {
    if l >= 0 && m >= 0 {
        positive_legendre_polynomial(l, m, x)
    } else if m < 0 {
        (-1.0_f32).powi(-m) * (l + m).factorial() as f32 / (l - m).factorial() as f32
        * positive_legendre_polynomial(l, -m, x)
    } else {
        0.0
    }
}

fn angular_wave_function((_n, l, m): (i32, i32, i32)) -> impl Fn(f32, f32, f32) -> Complex {
    move |x: f32, y: f32, z: f32| -> Complex {
        let (_r, t, p) = cartesian_to_spherical(x, y, z);
        spherical_harmonic(l, m, t, p)
    }
}

fn psi(qn: (i32, i32, i32), a: f32) -> impl Fn(f32, f32, f32) -> Complex {
    move |x: f32, y: f32, z: f32| -> Complex {
        radial_wave_function(qn, a)(x, y, z) * angular_wave_function(qn)(x, y, z)
    }
}

#[allow(unused)]
pub fn create_orbital((n, l, m): (i32, i32, i32), res: usize, min: f32, a: f32) -> Object<Pos3D> {
    // Assert that the user has entered valid quantum numbers
    assert!(n >= 1, "n = 1,2,...");
    assert!(l >= 0 && l <= n - 1, "l = 0,1,2,...,n-1");
    assert!(m >= -l && m <= l, "m = -l,-l+1,...,-1,0,1,...,l-1,l");

    let s = (2, 0, 0);
    let pz = (2, 1, 0);
    let px = (2, 1, 1);
    let py = (2, 1, -1);

    // let psi_d = psi((n, l, m), a);
    // let function = psi;
    // let psi_B = psi(px, sc_B, a);

    let psi_s = psi(s, a);
    let psi_x = psi(px, a);
    let psi_y = psi(py, a);
    let psi_z = psi(pz, a);

    // let sp3_1 = Complex(0.5, 0.0) * (psi_s + psi_x + psi_y + psi_z);
    //Complex(0.5, 0.0) * (psi_s(x, y, z) + psi_x(x, y, z) + psi_y(x, y, z) + psi_z(x, y, z))
    
    let function = |x: f32, y: f32, z: f32| {
        psi((n, l, m), a)(x, y, z).Re() * Complex(1.0, 0.0)
    };
    generate_orbital(function, res, min)
}

fn generate_orbital(
    func: impl Fn(f32, f32, f32) -> Complex,
    res: usize,
    min: f32,
) -> Object<Pos3D> {
    // Scale down the cube based on the resolution
    let normalized_func = move |pos: Pos3D| -> Complex {
        func(pos.x / res as f32, pos.y / res as f32, pos.z / res as f32)
    };

    let cutoff = move |q: Complex| q.abs() >= min;

    // Actually run the marching cube algorithm
    let (nodes, faces) = run_marching_cubes(normalized_func, (res, res, res), cutoff);

    // Assemble the object from thom the vertices/faces resulting from the algorithm
    let object = Object { nodes, faces };

    // Remove duplicate vertices to reduce the performance impact
    println!(
        "Object contains {} points before optimisation...",
        object.nodes.len()
    );
    let object = remove_duplicates(object);
    println!(
        "Object contains {} points after optimisation...",
        object.nodes.len()
    );
    let object = object
        .scale(1.0 / res as f32);

    object
}
