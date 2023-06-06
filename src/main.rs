use glam::Vec3;
use image::{ImageBuffer, Rgb};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;

const ASPECT_RATIO: f32 = 16.0 / 9.0;
const WIDTH: u32 = 400;
const HEIGHT: u32 = (WIDTH as f32 / ASPECT_RATIO) as u32;

const SAMPLES_PER_PIXEL: u32 = 40;
const MAX_DEPTH: i32 = 10;

fn main() {
    println!("Begin render {}x{}", WIDTH, HEIGHT);

    // Image
    let mut image_buffer = ImageBuffer::new(WIDTH, HEIGHT);

    // World
    let hittable_list: HittableList = HittableList {
        objects: Vec::from([
            Box::new(Sphere {
                center: Vec3::new(0.0, 0.0, -1.0),
                radius: 0.5,
            }) as Box<_>,
            Box::new(Sphere {
                center: Vec3::new(0.0, -100.5, -1.0),
                radius: 100.0,
            }) as Box<_>,
        ]),
    };

    // Camera
    let camera: Camera = Camera::new();
    let mut rng = rand::thread_rng();

    let pixel_multiplier = Vec3::new(255.0, 255.0, 255.0);
    let samples = Vec3::new(
        SAMPLES_PER_PIXEL as f32,
        SAMPLES_PER_PIXEL as f32,
        SAMPLES_PER_PIXEL as f32,
    );

    for y in 0..HEIGHT {
        println!("Line {}:", y);
        for x in 0..WIDTH {
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let u = (x as f32 + rng.gen_range(0.0..1.0)) / (WIDTH as f32 - 1.0);
                let v = (y as f32 + rng.gen_range(0.0..1.0)) / (HEIGHT as f32 - 1.0);

                let r = camera.get_ray(u, v);
                pixel_color += r.color(&hittable_list, &mut rng, MAX_DEPTH);
            }

            pixel_color /= samples;

            image_buffer.put_pixel(
                x,
                (HEIGHT - 1) - y,
                Rgb([
                    (pixel_color.x.sqrt() * 256.0) as u8,
                    (pixel_color.y.sqrt() * 256.0) as u8,
                    (pixel_color.z.sqrt() * 256.0) as u8,
                ]),
            );
        }
    }

    // Save image to file
    let file_name = format!("./images/raytrace-{}x{}.png", WIDTH, HEIGHT);
    image_buffer.save(file_name).unwrap();
}

pub struct Camera {
    origin: Vec3,
    bottom_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    pub fn new() -> Self {
        let viewport_height = 2.0;
        let viewport_width = ASPECT_RATIO * viewport_height;
        let focal_length = 1.0;

        let origin = Vec3::new(0.0, 0.0, 0.0);
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let bottom_left =
            origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

        Camera {
            origin,
            horizontal,
            vertical,
            bottom_left,
        }
    }

    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        return Ray {
            pos: self.origin,
            dir: self.bottom_left + u * self.horizontal + v * self.vertical - self.origin,
        };
    }
}

pub struct Ray {
    pos: Vec3,
    dir: Vec3,
}

pub struct HitRecord {
    p: Vec3,
    normal: Vec3,
    t: f32,
}

impl HitRecord {
    pub fn new(r: &Ray, normal: Vec3, t: f32, p: Vec3) -> Self {
        let front_face = Vec3::dot(r.dir, normal) < 0.0;
        HitRecord {
            t: t,
            p: p,
            normal: if front_face { normal } else { -normal },
        }
    }
}

pub trait Hittable {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
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
        return Some(HitRecord::new(r, (p - self.center) / self.radius, root, p));
    }
}

pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable>>,
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for hittable in self.objects.iter() {
            let h = hittable.hit(r, t_min, t_max);
            if h.is_some() {
                let hh = h.unwrap();
                if hh.t < closest_so_far {
                    closest_so_far = hh.t;
                    closest_hit = Some(hh);
                }
            }
        }

        return closest_hit;
    }
}

impl Ray {
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
            let target = rec.p + Ray::random_in_unit_sphere(rng, rec.normal);
            return 0.5
                * Ray {
                    pos: rec.p,
                    dir: target - rec.p,
                }
                .color(hittable, rng, depth - 1);
        }

        // Background gradient
        let unit_direction = self.dir.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0);
    }

    pub fn hit_sphere(&self, center: Vec3, radius: f32) -> f32 {
        let oc = self.pos - center;
        let a = self.dir.length_squared();

        let half_b = Vec3::dot(oc, self.dir);
        let c = oc.length_squared() - radius * radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return -1.0;
        } else {
            return (-half_b - discriminant.sqrt()) / a;
        }
    }

    pub fn random_in_unit_sphere(rng: &mut ThreadRng, normal: Vec3) -> Vec3 {
        let die = Uniform::from(-1.0..1.0);

        while true {
            let p = Vec3::new(die.sample(rng), die.sample(rng), die.sample(rng));

            if p.length_squared() < 1.0 {
                if (Vec3::dot(p, normal) > 0.0) {
                    return p;
                }

                return -p;
            }
        }

        return Vec3::new(0.0, 0.0, 0.0);
    }
}
