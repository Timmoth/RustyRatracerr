use crate::renderer::hit_record::HitRecord;
use crate::renderer::material::Material;
use crate::renderer::ray::Ray;
use glam::Vec3;
use rand::rngs::ThreadRng;
use rand::Rng;

pub struct Dielectric {
    pub index_of_refraction: f32,
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<(Ray, Vec3)> {
        let refraction_ratio = if rec.front_face {
            1.0 / self.index_of_refraction
        } else {
            self.index_of_refraction
        };

        let unit_direction = ray.dir.normalize();

        let cos_theta = Vec3::dot(-unit_direction, rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction = if cannot_refract
            || Dielectric::reflectance(cos_theta, refraction_ratio) > rng.gen_range(0.0..1.0)
        {
            Dielectric::reflect(unit_direction, rec.normal)
        } else {
            Dielectric::refract(unit_direction, rec.normal, refraction_ratio)
        };

        return Some((Ray::new(rec.p, direction), Vec3::new(1.0, 1.0, 1.0)));
    }
}

impl Dielectric {
    pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
        let cos_theta = Vec3::dot(-uv, n).min(1.0);
        let r_out_perp = etai_over_etat * (uv + cos_theta * n);
        let r_out_parallel = (1.0 - r_out_perp.length_squared()).abs().sqrt() * -n;
        return r_out_perp + r_out_parallel;
    }

    pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
        return v - 2.0 * Vec3::dot(v, n) * n;
    }

    pub fn reflectance(cosine: f32, refraction_ratio: f32) -> f32 {
        // Use Schlick's approximation for reflectance.
        let mut r0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio);
        r0 = r0 * r0;
        return r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0);
    }
}
