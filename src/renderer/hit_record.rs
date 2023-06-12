use glam::Vec3;

use std::rc::Rc;

use crate::renderer::material::Material;
use crate::renderer::ray::Ray;

pub struct HitRecord {
    pub p: Vec3,
    pub normal: Vec3,
    pub t: f32,
    pub front_face: bool,
    pub material: Rc<dyn Material + 'static>,
}

impl HitRecord {
    pub fn new(
        r: &Ray,
        normal: Vec3,
        t: f32,
        p: Vec3,
        material: Rc<dyn Material + 'static>,
    ) -> Self {
        let front_face = Vec3::dot(r.dir, normal) < 0.0;
        HitRecord {
            t: t,
            p: p,
            normal: if front_face { normal } else { -normal },
            front_face: front_face,
            material: material,
        }
    }
}
