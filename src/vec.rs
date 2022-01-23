use std::ops;

#[derive(Default, PartialEq, Debug, Clone, Copy)]
pub struct Vec3(pub f64, pub f64, pub f64);

impl Vec3 {
    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f64 {
        self.0 * self.0 + self.1 * self.1 + self.2 * self.2
    }

    pub fn unit_vector(&self) -> Vec3 {
        self / self.length()
    }

    pub fn near_zero(&self) -> bool {
        let s = 1e-8;
        self.0.abs() < s && self.1.abs() < s && self.2.abs() < s
    }
}

impl ops::Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl ops::Add<&Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl ops::AddAssign<Vec3> for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        *self = *self + rhs
    }
}

impl ops::Sub<Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl ops::Sub<&Vec3> for Vec3 {
    type Output = Vec3;

    fn sub(self, rhs: &Vec3) -> Self::Output {
        Vec3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl ops::SubAssign<Vec3> for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        *self = *self - rhs
    }
}

impl ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2)
    }
}

impl ops::Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec3(self.0 * rhs, self.1 * rhs, self.2 * rhs)
    }
}

impl ops::MulAssign<Vec3> for Vec3 {
    fn mul_assign(&mut self, rhs: Vec3) {
        *self = *self * rhs
    }
}

impl ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Self::Output {
        rhs * self
    }
}

impl ops::Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}

impl ops::Div<f64> for &Vec3 {
    type Output = Vec3;

    fn div(self, rhs: f64) -> Self::Output {
        Vec3(self.0 / rhs, self.1 / rhs, self.2 / rhs)
    }
}

impl ops::MulAssign<f64> for Vec3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.0 *= rhs;
        self.1 *= rhs;
        self.2 *= rhs;
    }
}

impl ops::DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, rhs: f64) {
        self.0 /= rhs;
        self.1 /= rhs;
        self.2 /= rhs;
    }
}

impl ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Self::Output {
        Vec3(-self.0, -self.1, -self.2)
    }
}

pub fn dot(lhs: Vec3, rhs: Vec3) -> f64 {
    lhs.0 * rhs.0 + lhs.1 * rhs.1 + lhs.2 * rhs.2
}

pub fn cross(lhs: Vec3, rhs: Vec3) -> Vec3 {
    Vec3(
        lhs.1 * rhs.2 - lhs.2 * rhs.1,
        lhs.2 * rhs.0 - lhs.0 * rhs.2,
        lhs.0 * rhs.1 - lhs.1 * rhs.0,
    )
}

pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * dot(v, n) * n
}

pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = dot(-uv, n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + n * cos_theta);
    let a = 1.0 - r_out_perp.length_squared();
    let b = a.abs().sqrt();
    let r_out_parallel = n * -b;
    r_out_perp + r_out_parallel
}

#[cfg(test)]
mod test {
    use super::*;

    // Perform the correct epsilon comparion for floats
    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if ($x - $y).abs() > $d {
                panic!();
            }
        };
    }

    #[test]
    fn create_default_vec3() {
        let v = Vec3::default();
        assert_delta!(v.0, 0.0, f64::EPSILON);
        assert_delta!(v.1, 0.0, f64::EPSILON);
        assert_delta!(v.2, 0.0, f64::EPSILON);
    }

    #[test]
    fn create_normal_vec3() {
        let v = Vec3(0.1, 0.2, 0.3);
        assert_delta!(v.0, 0.1, f64::EPSILON);
        assert_delta!(v.1, 0.2, f64::EPSILON);
        assert_delta!(v.2, 0.3, f64::EPSILON);
    }

    #[test]
    fn length() {
        let v = Vec3(2.0, 2.0, 2.0);
        assert_delta!(v.length_squared(), 12.0, f64::EPSILON);
        assert_delta!(v.length(), 12.0_f64.sqrt(), f64::EPSILON);
    }

    #[test]
    fn add() {
        let v = Vec3::default() + Vec3(1.0, 1.0, 1.0);
        assert_eq!(v, Vec3(1.0, 1.0, 1.0));
    }

    #[test]
    fn sub() {
        let v = Vec3(1.0, 1.0, 1.0) - Vec3(1.0, 1.0, 1.0);
        assert_eq!(v, Vec3(0.0, 0.0, 0.0));
    }

    #[test]
    fn mul() {
        let v = Vec3(3.0, 2.0, 1.0) * Vec3(1.0, 2.0, 3.0);
        assert_eq!(v, Vec3(3.0, 4.0, 3.0));
    }

    #[test]
    fn div() {
        let v = Vec3(3.0, 2.0, 1.5) / 2.0;
        assert_eq!(v, Vec3(1.5, 1.0, 0.75));
    }
}
