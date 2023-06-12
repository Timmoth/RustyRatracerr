use crate::renderer::hit_record::HitRecord;
use glam::Vec3;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;

pub struct Ray {
    pub pos: Vec3,
    pub dir: Vec3,
}

pub trait Hittable {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

impl Ray {
    pub fn new(pos: Vec3, dir: Vec3) -> Self {
        return Ray { pos: pos, dir: dir };
    }

    pub fn at(&self, t: f32) -> Vec3 {
        return self.pos + t * self.dir;
    }

    pub fn color(&self, hittable: &impl Hittable, rng: &mut ThreadRng, depth: i32) -> Vec3 {
        if depth <= 0 {
            return Vec3::new(0.0, 0.0, 0.0);
        }

        let hit = hittable.hit(self, 0.001, std::f32::INFINITY);
        if hit.is_some() {
            // Draw collision surface normal
            let rec = hit.unwrap();
            let cc = rec.material.scatter(self, &rec, rng);

            if cc.is_some() {
                let ccc = cc.unwrap();
                return ccc.1 * ccc.0.color(hittable, rng, depth - 1);
            }

            return Vec3::new(0.0, 0.0, 0.0);
        }

        // Background gradient
        let unit_direction = self.dir.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0);
    }

    pub fn random_in_unit_sphere(rng: &mut ThreadRng) -> Vec3 {
        loop {
            let p = Vec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            );

            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }

    pub fn random_in_hemisphere(p: Vec3, normal: Vec3) -> Vec3 {
        if Vec3::dot(p, normal) > 0.0 {
            return p;
        }

        return -p;
    }
}
