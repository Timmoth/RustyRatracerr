use crate::renderer::hit_record::HitRecord;
use crate::renderer::material::Material;
use crate::renderer::ray::Hittable;
use crate::renderer::ray::Ray;
use glam::Vec3;
use std::rc::Rc;

impl Sphere {
    pub fn new(center: Vec3, radius: f32, material: Rc<dyn Material + 'static>) -> Self {
        Sphere {
            center: center,
            radius: radius,
            material: material,
        }
    }
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
    pub material: Rc<dyn Material + 'static>,
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = r.pos - self.center;
        let a = r.dir.length_squared();

        let half_b = Vec3::dot(oc, r.dir);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let p = r.at(root);
        return Some(HitRecord::new(
            r,
            (p - self.center) / self.radius,
            root,
            p,
            self.material.clone(),
        ));
    }
}
