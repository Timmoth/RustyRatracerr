use glam::Vec3;
use image::{ImageBuffer, Rgb};
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use std::rc::Rc;

use mpi::topology::SystemCommunicator;
use mpi::traits::*;
use std::process;
use std::thread::sleep;
use std::time;
use std::time::{Duration, SystemTime};

const ASPECT_RATIO: f32 = 16.0 / 9.0;
const WIDTH: usize = 1920;
const HEIGHT: usize = (WIDTH as f32 / ASPECT_RATIO) as usize;

const SAMPLES_PER_PIXEL: u32 = 3000;
const MAX_DEPTH: i32 = 100;

fn main() {
    println!("Begin render {}x{}", WIDTH, HEIGHT);

    // Open MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    if (world.rank() == 0) {
        RunMaster(world, WIDTH, HEIGHT);
    } else {
        RunWorker(world, WIDTH, HEIGHT);
    }
}

fn RunMaster(world: SystemCommunicator, width: usize, height: usize) {
    println!("Starting");
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    let mut linesOut: i32 = 0;
    let mut linesIn: i32 = 0;

    let mut imageBuffer = image::ImageBuffer::new(width as u32, height as u32);
    let mut data: Vec<u32> = vec![0u32; width];
    let timer = SystemTime::now();

    loop {
        let r = world.any_process().receive_into(&mut data);

        if (r.tag() != 0) {
            linesIn += 1;
            println!("{} Finished line: {}", r.source_rank(), r.tag());
            let y = r.tag() as u32;
            for x in 0..width {
                let pixel = imageBuffer.get_pixel_mut(x as u32, height as u32 - y);
                let rgb = data[x];

                let red = (rgb >> 16) & 255;
                let green = rgb >> 8 & 255;
                let blue = rgb & 255;

                *pixel = image::Rgb([red as u8, green as u8, blue as u8]);
            }
        }

        world.process_at_rank(r.source_rank()).send(&mut linesOut);
        linesOut += 1;

        if (linesIn >= (height - 1) as i32) {
            // Save image to file
            let file_name = format!("raytrace-{}x{}.png", width, height);
            imageBuffer.save(file_name).unwrap();

            // Status update
            match timer.elapsed() {
                Ok(elapsed) => {
                    println!("Finished {}ms", elapsed.as_millis());
                }
                Err(e) => {
                    println!("Error: {e:?}");
                }
            }

            world.abort(0);
            break;
        }
    }
}

fn RunWorker(world: SystemCommunicator, width: usize, height: usize) {
    let size = world.size() as usize;
    let rank = world.rank() as usize;
    let root_process = world.process_at_rank(0);

    let mut lineIndex = 0;
    // Used to store pixel values

    let mut data: Vec<u32> = vec![0u32; width];

    root_process.send_with_tag(&data, lineIndex);
    root_process.receive_into(&mut lineIndex);

    loop {
        if (lineIndex >= height as i32) {
            println!("Exit node {}", rank);
            break;
        }

        let mut y = lineIndex;
        println!("{} Working on line: {}", rank, y);

        // Calculate each pixel
        let timer = SystemTime::now();

        // Camera
        let look_from = Vec3::new(13.0, 2.0, 3.0);
        let look_at = Vec3::new(0.0, 0.0, 0.0);
        let camera: Camera = Camera::new(
            look_from,
            look_at,
            Vec3::new(0.0, 1.0, 0.0),
            0.35,
            ASPECT_RATIO,
            0.1,
            10.0,
        );
        let mut rng = rand::thread_rng();

        // World
        let mut hittable_list: HittableList = HittableList {
            objects: Vec::from([
                Box::new(Sphere::new(
                    Vec3::new(4.0, 1.0, 0.0),
                    1.0,
                    Rc::new(Metal {
                        albedo: Vec3::new(0.8, 0.8, 0.8),
                        fuzz: 0.0,
                    }) as Rc<_>,
                )) as Box<_>,
                Box::new(Sphere::new(
                    Vec3::new(0.0, 1.0, 0.0),
                    1.0,
                    Rc::new(Dielectric {
                        index_of_refraction: 1.5,
                    }) as Rc<_>,
                )) as Box<_>,
                Box::new(Sphere::new(
                    Vec3::new(-4.0, 1.0, 0.0),
                    -1.0,
                    Rc::new(Dielectric {
                        index_of_refraction: 1.5,
                    }) as Rc<_>,
                )) as Box<_>,
                Box::new(Sphere::new(
                    Vec3::new(0.0, -1000.0, -1.0),
                    1000.0,
                    Rc::new(Lambertian {
                        albedo: Vec3::new(0.5, 0.5, 0.5),
                    }) as Rc<_>,
                )) as Box<_>,
            ]),
        };

        let mut r = ChaCha8Rng::seed_from_u64(2);

        for a in -11..11 {
            for b in -11..11 {
                let choose_mat = r.gen_range(0.0..1.0);
                let center = Vec3::new(
                    a as f32 + 0.9 * r.gen_range(0.0..1.0),
                    0.2,
                    b as f32 + 0.9 * r.gen_range(0.0..1.0),
                );

                if ((center - Vec3::new(4.0, 0.2, 0.0)).length() < 0.9) {
                    continue;
                }

                if (choose_mat < 0.7) {
                    hittable_list.objects.insert(
                        0,
                        Box::new(Sphere {
                            center: center,
                            radius: 0.2,
                            material: Rc::new(Lambertian {
                                albedo: Vec3::new(
                                    r.gen_range(0.0..1.0),
                                    r.gen_range(0.0..1.0),
                                    r.gen_range(0.0..1.0),
                                ),
                            }),
                        }) as Box<_>,
                    );
                } else if (choose_mat < 0.9) {
                    hittable_list.objects.insert(
                        0,
                        Box::new(Sphere {
                            center: center,
                            radius: 0.2,
                            material: Rc::new(Metal {
                                fuzz: r.gen_range(0.0..0.5),
                                albedo: Vec3::new(
                                    r.gen_range(0.5..1.0),
                                    r.gen_range(0.5..1.0),
                                    r.gen_range(0.5..1.0),
                                ),
                            }),
                        }) as Box<_>,
                    );
                } else {
                    hittable_list.objects.insert(
                        0,
                        Box::new(Sphere {
                            center: center,
                            radius: 0.2,
                            material: Rc::new(Dielectric {
                                index_of_refraction: 1.5,
                            }),
                        }) as Box<_>,
                    );
                }
            }
        }

        let pixel_multiplier = Vec3::new(255.0, 255.0, 255.0);
        let samples = Vec3::new(
            SAMPLES_PER_PIXEL as f32,
            SAMPLES_PER_PIXEL as f32,
            SAMPLES_PER_PIXEL as f32,
        );

        for x in 0..WIDTH {
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
            for _ in 0..SAMPLES_PER_PIXEL {
                let u = (x as f32 + rng.gen_range(0.0..1.0)) / (WIDTH as f32 - 1.0);
                let v = (y as f32 + rng.gen_range(0.0..1.0)) / (HEIGHT as f32 - 1.0);

                let r = camera.get_ray(u, v, &mut rng);
                pixel_color += r.color(&hittable_list, &mut rng, MAX_DEPTH);
            }

            pixel_color /= samples;

            let r = (pixel_color.x.sqrt() * 255.0) as u32;
            let g = (pixel_color.y.sqrt() * 255.0) as u32;
            let b = (pixel_color.z.sqrt() * 255.0) as u32;
            // Store the color value based on the escape time
            data[x] = r << 16 | g << 8 | b
        }

        // Status update
        match timer.elapsed() {
            Ok(elapsed) => {
                println!("Rank{} {}/{} {}ms", rank, y, height, elapsed.as_millis());
            }
            Err(e) => {
                println!("Error: {e:?}");
            }
        }

        root_process.send_with_tag(&data, y);
        root_process.receive_into(&mut lineIndex);
    }
}

pub struct Camera {
    origin: Vec3,
    bottom_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    lens_radius: f32,
    u: Vec3,
    v: Vec3,
    w: Vec3,
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
            w,
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
        while true {
            let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
            if (p.length_squared() >= 1.0) {
                continue;
            }
            return p;
        }

        return Vec3::new(0.0, 0.0, 0.0);
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
