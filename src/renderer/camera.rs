use glam::Vec3;
use rand::rngs::ThreadRng;
use rand::Rng;

use crate::renderer::ray::Ray;

pub struct Camera {
    origin: Vec3,
    bottom_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    lens_radius: f32,
    u: Vec3,
    v: Vec3,
}

impl Camera {
    pub fn new(
        look_from: Vec3,
        look_at: Vec3,
        v_up: Vec3,
        vertical_fov: f32,
        aspect_ratio: f32,
        aperture: f32,
        focus_dist: f32,
    ) -> Self {
        let h = (vertical_fov / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).normalize();
        let u = Vec3::cross(v_up, w).normalize();
        let v = Vec3::cross(w, u);

        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;

        let bottom_left = look_from - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;

        Camera {
            origin: look_from,
            horizontal: focus_dist * viewport_width * u,
            vertical: focus_dist * viewport_height * v,
            bottom_left,
            lens_radius: aperture / 2.0,
            u,
            v,
        }
    }

    pub fn get_ray(&self, s: f32, t: f32, rng: &mut ThreadRng) -> Ray {
        let rd = self.lens_radius * Camera::random_in_unit_disk(rng);
        let offset = self.u * rd.x + self.v * rd.y;
        return Ray {
            pos: self.origin + offset,
            dir: self.bottom_left + s * self.horizontal + t * self.vertical - self.origin - offset,
        };
    }

    pub fn random_in_unit_disk(rng: &mut ThreadRng) -> Vec3 {
        loop {
            let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if p.length_squared() < 1.0 {
                return p;
            }
        }
    }
}
