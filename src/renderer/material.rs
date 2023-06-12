use glam::Vec3;
use rand::rngs::ThreadRng;

use crate::renderer::hit_record::HitRecord;
use crate::renderer::ray::Ray;

pub trait Material {
    fn scatter(
        &self,
        ray: &Ray,
        hit_record: &HitRecord,
        rng: &mut ThreadRng,
    ) -> Option<(Ray, Vec3)>;
}
