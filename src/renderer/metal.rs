use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::renderer::hit_record::HitRecord;
use crate::renderer::material::Material;
use crate::renderer::ray::Ray;

pub struct Metal {
    pub albedo: Vec3,
    pub fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<(Ray, Vec3)> {
        let v = ray.dir.normalize();
        let n = rec.normal;
        let reflected = v - 2.0 * Vec3::dot(v, n) * n;

        if Vec3::dot(reflected, rec.normal) > 0.0 {
            return Some((
                Ray::new(
                    rec.p,
                    reflected + self.fuzz * Ray::random_in_unit_sphere(rng),
                ),
                self.albedo,
            ));
        }

        return None;
    }
}
