use glam::Vec3;
use rand::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use std::rc::Rc;

use crate::renderer::camera::Camera;
use crate::renderer::dielectric::Dielectric;
use crate::renderer::hittable_list::HittableList;
use crate::renderer::lambertian::Lambertian;
use crate::renderer::metal::Metal;
use crate::renderer::ray::Hittable;
use crate::renderer::sphere::Sphere;

use mpi::topology::SystemCommunicator;
use mpi::traits::*;
use std::time::SystemTime;

const ASPECT_RATIO: f32 = 16.0 / 9.0;
const WIDTH: usize = 1000;
const HEIGHT: usize = (WIDTH as f32 / ASPECT_RATIO) as usize;

const SAMPLES_PER_PIXEL: u32 = 1000;
const MAX_DEPTH: i32 = 60;
mod renderer;

fn main() {
    // Initialize Open MPI
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    if world.rank() == 0 {
        run_master(world, WIDTH, HEIGHT);
    } else {
        run_worker(world, WIDTH, HEIGHT);
    }
}

fn run_master(world: SystemCommunicator, width: usize, height: usize) {
    println!("Begin render {}x{}", width, height);

    let mut rows_dispatched: i32 = 0; // The number of rows that have been dispatched to workers
    let mut rows_completed: i32 = 0; // The number of rows that have been completed by workers

    // Used to save the rendered image
    let mut output_image_buffer = image::ImageBuffer::new(width as u32, height as u32);
    // Stores row pixel data produced by each worker.
    let mut worker_row_buffer: Vec<u32> = vec![0u32; width];
    // Used to time how long the render takes
    let timer = SystemTime::now();

    loop {
        // Wait for the next worker to produce a new row
        let status = world.any_process().receive_into(&mut worker_row_buffer);
        let worker_index = status.source_rank();
        let row_index = status.tag() as u32;

        if row_index != 0 {
            rows_completed += 1;
            for x in 0..width {
                // Get the pixel data from the worker row buffer
                let rgb = worker_row_buffer[x];
                let red = (rgb >> 16) & 255;
                let green = rgb >> 8 & 255;
                let blue = rgb & 255;

                // Store the pixel data in the output image buffer
                let pixel = output_image_buffer.get_pixel_mut(x as u32, height as u32 - row_index);
                *pixel = image::Rgb([red as u8, green as u8, blue as u8]);
            }
        }

        // Send the worker a new row to render
        world
            .process_at_rank(worker_index)
            .send(&mut rows_dispatched);
        rows_dispatched += 1;

        if rows_completed >= (height - 1) as i32 {
            // Save image to file
            let file_name = format!("raytrace-{}x{}_{}.png", width, height, SAMPLES_PER_PIXEL);
            output_image_buffer.save(file_name).unwrap();

            // Output how long the render took
            match timer.elapsed() {
                Ok(elapsed) => {
                    println!("Finished {}ms", elapsed.as_millis());
                }
                Err(e) => {
                    println!("Error: {e:?}");
                }
            }

            // Exit the program
            world.abort(0);
        }
    }
}

fn run_worker(world: SystemCommunicator, width: usize, height: usize) {
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

    let mut scene_rng = ChaCha8Rng::seed_from_u64(2);

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = scene_rng.gen_range(0.0..1.0);
            let center = Vec3::new(
                a as f32 + 0.9 * scene_rng.gen_range(0.0..1.0),
                0.2,
                b as f32 + 0.9 * scene_rng.gen_range(0.0..1.0),
            );

            if (center - Vec3::new(4.0, 0.2, 0.0)).length() < 0.9 {
                continue;
            }

            if choose_mat < 0.7 {
                hittable_list.objects.insert(
                    0,
                    Box::new(Sphere {
                        center: center,
                        radius: 0.2,
                        material: Rc::new(Lambertian {
                            albedo: Vec3::new(
                                scene_rng.gen_range(0.0..1.0),
                                scene_rng.gen_range(0.0..1.0),
                                scene_rng.gen_range(0.0..1.0),
                            ),
                        }),
                    }) as Box<_>,
                );
            } else if choose_mat < 0.9 {
                hittable_list.objects.insert(
                    0,
                    Box::new(Sphere {
                        center: center,
                        radius: 0.2,
                        material: Rc::new(Metal {
                            fuzz: scene_rng.gen_range(0.0..0.5),
                            albedo: Vec3::new(
                                scene_rng.gen_range(0.5..1.0),
                                scene_rng.gen_range(0.5..1.0),
                                scene_rng.gen_range(0.5..1.0),
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

    let worker_index = world.rank() as usize;
    let root_process = world.process_at_rank(0);

    // The index of the current row being processed by the worker
    let mut row_index = 0;

    // Used to store pixel values being rendered for the current row
    let mut row_image_buffer: Vec<u32> = vec![0u32; width];

    // Initially send an empty buffer to the root process to let it know that the worker has started
    root_process.send_with_tag(&row_image_buffer, row_index);
    // Wait for the root process to send a row index to be rendered
    root_process.receive_into(&mut row_index);

    loop {
        if row_index >= height as i32 {
            // If the worker sends an index that is larger than the height of the image, the program has finished
            println!("Exit node {}", worker_index);
            break;
        }

        // Used to time how long this row took to render
        let timer = SystemTime::now();

        render(
            row_image_buffer.as_mut_slice(),
            row_index,
            &hittable_list,
            &camera,
        );

        // Status update
        match timer.elapsed() {
            Ok(elapsed) => {
                println!(
                    "worker {} completed row {}/{} in {}ms",
                    worker_index,
                    row_index,
                    height,
                    elapsed.as_millis()
                );
            }
            Err(e) => {
                println!("Error: {e:?}");
            }
        }

        // Tell the root process to send a new row to render
        root_process.send_with_tag(&row_image_buffer, row_index);
        // Wait for the root process to send a row index to be rendered
        root_process.receive_into(&mut row_index);
    }
}

fn render(
    row_image_buffer: &mut [u32],
    row_index: i32,
    hittable_list: &impl Hittable,
    camera: &Camera,
) {
    let mut rng = rand::thread_rng();

    let samples = Vec3::new(
        SAMPLES_PER_PIXEL as f32,
        SAMPLES_PER_PIXEL as f32,
        SAMPLES_PER_PIXEL as f32,
    );

    for x in 0..WIDTH {
        let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
        for _ in 0..SAMPLES_PER_PIXEL {
            let u = (x as f32 + rng.gen_range(0.0..1.0)) / (WIDTH as f32 - 1.0);
            let v = (row_index as f32 + rng.gen_range(0.0..1.0)) / (HEIGHT as f32 - 1.0);

            let r = camera.get_ray(u, v, &mut rng);
            pixel_color += r.color(hittable_list, &mut rng, MAX_DEPTH);
        }

        pixel_color /= samples;

        let r = (pixel_color.x.sqrt() * 255.0) as u32;
        let g = (pixel_color.y.sqrt() * 255.0) as u32;
        let b = (pixel_color.z.sqrt() * 255.0) as u32;

        row_image_buffer[x] = r << 16 | g << 8 | b
    }
}
