use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Complex(pub f64, pub f64);

#[allow(non_snake_case)]
pub trait Split {
    fn Re(self) -> f64;
    fn Im(self) -> f64;
}

impl Split for Complex {
    fn Re(self) -> f64 {
        self.0
    }

    fn Im(self) -> f64 {
        self.1
    }
}

impl Add for Complex {
    type Output = Complex;

    fn add(self, rhs: Self) -> Self::Output {
        Complex(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Add<f64> for Complex {
    type Output = Complex;

    fn add(self, rhs: f64) -> Self::Output {
        Complex(self.0 + rhs, self.1)
    }
}

impl Add<Complex> for f64 {
    type Output = Complex;

    fn add(self, rhs: Complex) -> Self::Output {
        Complex(self + rhs.0, rhs.1)
    }
}

impl Sub for Complex {
    type Output = Complex;

    fn sub(self, rhs: Self) -> Self::Output {
        Complex(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl Sub<f64> for Complex {
    type Output = Complex;

    fn sub(self, rhs: f64) -> Self::Output {
        Complex(self.0 - rhs, self.1)
    }
}

impl Sub<Complex> for f64 {
    type Output = Complex;

    fn sub(self, rhs: Complex) -> Self::Output {
        Complex(self - rhs.0, rhs.1)
    }
}

impl Mul for Complex {
    type Output = Complex;

    fn mul(self, rhs: Self) -> Self::Output {
        Complex(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;

    fn mul(self, rhs: f64) -> Self::Output {
        Complex(self.0 * rhs, self.1 * rhs)
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;

    fn mul(self, rhs: Complex) -> Self::Output {
        Complex(self * rhs.0, self * rhs.1)
    }
}

impl Div for Complex {
    type Output = Complex;

    fn div(self, rhs: Self) -> Self::Output {
        Complex(
            (self.0 * rhs.0 + self.1 * rhs.1) / (rhs.0 * rhs.0 + rhs.1 * rhs.1),
            -(rhs.0 * self.1 - self.0 * rhs.1) / (rhs.0 * rhs.0 + rhs.1 * rhs.1),
        )
    }
}

impl Div<f64> for Complex {
    type Output = Complex;

    fn div(self, rhs: f64) -> Self::Output {
        Complex(self.0 / rhs, self.1 / rhs)
    }
}

impl Div<Complex> for f64 {
    type Output = Complex;

    fn div(self, rhs: Complex) -> Self::Output {
        Complex(
            (self * rhs.0) / (rhs.0 * rhs.0 + rhs.1 * rhs.1),
            (self * rhs.1) / (rhs.0 * rhs.0 + rhs.1 * rhs.1),
        )
    }
}

pub trait Exp<Rhs = Self> {
    type Output;

    fn exp(self) -> Self::Output;
}

impl Exp for Complex {
    type Output = Complex;

    fn exp(self) -> Self::Output {
        Complex(self.0.exp() * self.1.cos(), self.0.exp() * self.1.sin())
    }
}
pub trait AbsArg {
    type Output;

    fn abs(self) -> Self::Output;
    fn arg(self) -> Self::Output;
}

impl AbsArg for Complex {
    type Output = f64;

    fn abs(self) -> Self::Output {
        (self.0 * self.0 + self.1 * self.1).sqrt()
    }

    fn arg(self) -> Self::Output {
        self.0.atan2(self.1)
    }
}

pub trait Conjugate {
    type Output;

    fn conjugate(self) -> Self::Output;
}

impl Conjugate for Complex {
    type Output = Complex;

    fn conjugate(self) -> Self::Output {
        Complex(self.0, -self.1)
    }
}