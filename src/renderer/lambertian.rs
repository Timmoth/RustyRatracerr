use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::renderer::hit_record::HitRecord;
use crate::renderer::material::Material;
use crate::renderer::ray::Ray;

pub struct Lambertian {
    pub albedo: Vec3,
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<(Ray, Vec3)> {
        let scatter_direction = rec.normal + Ray::random_in_unit_sphere(rng);

        let abs = scatter_direction.abs();
        if abs.x < 1e-8 && abs.y < 1e-8 && abs.z < 1e-8 {
            return Some((
                Ray {
                    pos: rec.p,
                    dir: rec.normal,
                },
                self.albedo,
            ));
        }

        return Some((
            Ray {
                pos: rec.p,
                dir: scatter_direction,
            },
            self.albedo,
        ));
    }
}
