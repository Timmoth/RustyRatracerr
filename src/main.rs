use glam::Vec3;
use image::{ImageBuffer, Rgb};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::rc::Rc;

const ASPECT_RATIO: f32 = 16.0 / 9.0;
const WIDTH: u32 = 1000;
const HEIGHT: u32 = (WIDTH as f32 / ASPECT_RATIO) as u32;

const SAMPLES_PER_PIXEL: u32 = 200;
const MAX_DEPTH: i32 = 100;

fn main() {
    println!("Begin render {}x{}", WIDTH, HEIGHT);

    // Image
    let mut image_buffer = ImageBuffer::new(WIDTH, HEIGHT);
    // Camera
    let camera: Camera = Camera::new(2.4, ASPECT_RATIO);
    let mut rng = rand::thread_rng();

    // World
    let mut hittable_list: HittableList = HittableList {
        objects: Vec::from([
            Box::new(Sphere::new(
                Vec3::new(-1.0, 0.0, -1.0),
                0.5,
                Rc::new(Metal {
                    albedo: Vec3::new(0.5, 0.5, 0.5),
                    fuzz: 0.3,
                }) as Rc<_>,
            )) as Box<_>,
            Box::new(Sphere::new(
                Vec3::new(0.0, 0.0, -2.0),
                0.5,
                Rc::new(Metal {
                    albedo: Vec3::new(0.5, 0.5, 0.5),
                    fuzz: 0.01,
                }) as Rc<_>,
            )) as Box<_>,
            Box::new(Sphere::new(
                Vec3::new(0.0, 0.0, -0.5),
                0.5,
                Rc::new(Dielectric {
                    index_of_refraction: 1.1,
                }) as Rc<_>,
            )) as Box<_>,
            Box::new(Sphere::new(
                Vec3::new(1.0, 0.0, -1.0),
                0.5,
                Rc::new(Metal {
                    albedo: Vec3::new(0.5, 0.5, 0.5),
                    fuzz: 0.0,
                }) as Rc<_>,
            )) as Box<_>,
            Box::new(Sphere::new(
                Vec3::new(0.0, -100.5, -1.0),
                100.0,
                Rc::new(Lambertian {
                    albedo: Vec3::new(0.5, 0.5, 0.5),
                }) as Rc<_>,
            )) as Box<_>,
        ]),
    };

    // for n in 0..100 {
    //     hittable_list.objects.insert(
    //         0,
    //         Box::new(Sphere {
    //             center: Vec3::new(
    //                 rng.gen_range(0.0..60.0) - 30.0,
    //                 rng.gen_range(0.0..20.0),
    //                 -rng.gen_range(0.0..30.0) + 5.0,
    //             ),
    //             radius: rng.gen_range(0.1..0.4),
    //             material: Rc::new(Metal {
    //                 albedo: Vec3::new(0.5, 0.5, 0.5),
    //             }),
    //         }) as Box<_>,
    //     );
    // }

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
    pub fn new(vertical_fov: f32, aspect_ratio: f32) -> Self {
        let viewport_height = (vertical_fov / 2.0).tan();
        let viewport_width = aspect_ratio * viewport_height;
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
    front_face: bool,
    material: Rc<dyn Material + 'static>,
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

pub trait Hittable {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

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
    center: Vec3,
    radius: f32,
    material: Rc<dyn Material + 'static>,
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

            if (cc.is_some()) {
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

    pub fn random_in_unit_sphere(rng: &mut ThreadRng) -> Vec3 {
        let die = Uniform::from(-1.0..1.0);

        while true {
            let p = Vec3::new(die.sample(rng), die.sample(rng), die.sample(rng));

            if p.length_squared() < 1.0 {
                return p;
            }
        }

        return Vec3::new(0.0, 0.0, 0.0);
    }

    pub fn random_in_hemisphere(p: Vec3, normal: Vec3) -> Vec3 {
        if (Vec3::dot(p, normal) > 0.0) {
            return p;
        }

        return -p;
    }
}

pub trait Material {
    fn scatter(
        &self,
        r_in: &Ray,
        hitRecord: &HitRecord,
        rng: &mut ThreadRng,
    ) -> Option<(Ray, Vec3)>;
}

pub struct Lambertian {
    pub albedo: Vec3,
}

impl Material for Lambertian {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<(Ray, Vec3)> {
        let scatter_direction = rec.normal + Ray::random_in_unit_sphere(rng);

        let abs = scatter_direction.abs();
        if (abs.x < 1e-8 && abs.y < 1e-8 && abs.z < 1e-8) {
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

pub struct Metal {
    pub albedo: Vec3,
    pub fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<(Ray, Vec3)> {
        let v = r_in.dir.normalize();
        let n = rec.normal;
        let reflected = v - 2.0 * Vec3::dot(v, n) * n;

        if (Vec3::dot(reflected, rec.normal) > 0.0) {
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

pub struct Dielectric {
    pub index_of_refraction: f32,
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<(Ray, Vec3)> {
        let refraction_ratio = if rec.front_face {
            (1.0 / self.index_of_refraction)
        } else {
            self.index_of_refraction
        };

        let unit_direction = r_in.dir.normalize();

        let cos_theta = Vec3::dot(-unit_direction, rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction = if (cannot_refract
            || Dielectric::reflectance(cos_theta, refraction_ratio) > rng.gen_range(0.0..1.0))
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
