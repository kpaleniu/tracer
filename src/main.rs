mod vec;

extern crate image;
extern crate piston_window;

use core::f64::consts::PI;
use image::ImageBuffer;
use image::Rgba;
use piston_window::types::Color;
use piston_window::*;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::Rng;
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use vec::Vec3;

const ASPECT_RATIO: f64 = 16.0 / 9.0;
const WIDTH: u32 = 640;
const HEIGHT: u32 = (WIDTH as f64 / ASPECT_RATIO) as u32;

#[derive(Debug)]
pub struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + t * self.direction
    }
}

impl From<Vec3> for Color {
    fn from(v: Vec3) -> Color {
        [v.0 as f32, v.1 as f32, v.2 as f32, 1.0]
    }
}

fn fragment_to_pixel(frag: Vec3, samples_per_pixel: i32) -> Rgba<u8> {
    let scale = 1.0 / samples_per_pixel as f64;
    let r = frag.0 * scale;
    let g = frag.1 * scale;
    let b = frag.2 * scale;

    Rgba([
        (256.0 * r.clamp(0.0, 0.999)) as u8,
        (256.0 * g.clamp(0.0, 0.999)) as u8,
        (256.0 * b.clamp(0.0, 0.999)) as u8,
        255,
    ])
}

struct Camera {
    origin: Vec3,
    bottom_left: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new(
        look_from: Vec3,
        look_at: Vec3,
        vertical_up: Vec3,
        vertical_fov: f64,
        aspect_ratio: f64,
    ) -> Self {
        let h = (vertical_fov.deg_to_rad() / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).unit_vector();
        let u = vec::cross(vertical_up, -w).unit_vector();
        let v = vec::cross(w, u);

        let origin = look_from;
        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        Self {
            origin,
            bottom_left: origin - horizontal / 2.0 - vertical / 2.0 - w,
            horizontal,
            vertical,
        }
    }

    fn ray(&self, u: f64, v: f64) -> Ray {
        let direction = self.bottom_left + u * self.horizontal + v * self.vertical - self.origin;
        Ray::new(self.origin, direction)
    }
}

#[derive(Clone)]
struct Hit {
    point: Vec3,
    normal: Vec3,
    material: Arc<dyn Material + Send + Sync>,
    t: f64,
    front_face: bool,
}

trait Hittable {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit>;
}

#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: f64,
    material: Arc<dyn Material + Send + Sync>,
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let oc = r.origin - self.center;
        let a = r.direction.length_squared();
        let half_b = vec::dot(oc, r.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();
        let root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            let root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }

        let point = r.at(root);
        let outward_normal = (point - self.center) / self.radius;
        let (normal, front_face) = if vec::dot(r.direction, outward_normal) < 0.0 {
            (outward_normal, true)
        } else {
            (-outward_normal, false)
        };

        Some(Hit {
            point,
            normal,
            t: root,
            front_face,
            material: self.material.clone(),
        })
    }
}

trait Material {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> (Ray, Vec3, bool);
}

struct Lambertian {
    albedo: Vec3,
}

impl Material for Lambertian {
    fn scatter(&self, _r_in: &Ray, hit: &Hit) -> (Ray, Vec3, bool) {
        let mut scatter_direction = hit.normal + random_in_unit_sphere();
        if scatter_direction.near_zero() {
            scatter_direction = hit.normal;
        }
        let scattered = Ray::new(hit.point, scatter_direction);
        (scattered, self.albedo, true)
    }
}

struct Metal {
    albedo: Vec3,
    fuzz: f64,
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> (Ray, Vec3, bool) {
        let reflected = vec::reflect(r_in.direction.unit_vector(), hit.normal);
        let scattered = Ray::new(hit.point, reflected + self.fuzz * random_in_unit_sphere());
        let did_scatter = vec::dot(scattered.direction, hit.normal) > 0.0;
        (scattered, self.albedo, did_scatter)
    }
}

struct Dielectric {
    ir: f64,
}

impl Dielectric {
    fn reflectance(cosine: f64, ref_index: f64) -> f64 {
        let r0 = (1.0 - ref_index) / (1.0 + ref_index);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> (Ray, Vec3, bool) {
        let attentuation = Vec3(1.0, 1.0, 1.0);
        let refraction_ratio = if hit.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };
        let unit_direction = r_in.direction.unit_vector();
        let cos_theta = vec::dot(-unit_direction, hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let mut rng = rand::thread_rng();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction =
            if cannot_refract || Self::reflectance(cos_theta, refraction_ratio) > rng.gen() {
                vec::reflect(unit_direction, hit.normal)
            } else {
                vec::refract(unit_direction, hit.normal, refraction_ratio)
            };
        let scattered = Ray::new(hit.point, direction);
        (scattered, attentuation, true)
    }
}

struct Scene {
    components: Vec<Arc<dyn Hittable + Send + Sync>>,
}

impl Scene {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut h = None;
        let mut closest = t_max;
        for c in &self.components {
            if let Some(result) = c.hit(r, t_min, closest) {
                h = Some(result.clone());
                closest = result.t;
            }
        }
        h
    }
}

fn random_in_unit_sphere() -> Vec3 {
    let mut rng = rand::thread_rng();
    let dist = Uniform::from(0.0..1.0);
    let theta = 2.0 * PI * dist.sample(&mut rng);
    let phi = (1.0 - 2.0 * dist.sample(&mut rng)).acos();
    let x = phi.sin() * theta.cos();
    let y = phi.sin() * theta.sin();
    let z = phi.cos();
    Vec3(x, y, z)
}

fn raycast(r: &Ray, scene: &Scene, depth: u32) -> Vec3 {
    if depth == 0 {
        return Vec3(0.0, 0.0, 0.0);
    }

    if let Some(hit) = scene.hit(r, 0.0001, f64::INFINITY) {
        let (ray, attenuation, did_scatter) = hit.material.scatter(r, &hit);
        if did_scatter {
            return attenuation * raycast(&ray, scene, depth - 1);
        }
        return Vec3(0.0, 0.0, 0.0);
    }

    let d = r.direction.unit_vector();
    let t = 0.5 * (d.1 + 1.0);

    // blend black and blue
    (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0)
}

struct ComputeContext {
    handle: JoinHandle<()>,
    cancel: Sender<()>,
}

fn compute<'a>(
    range: &'a [u32],
    camera: Arc<Camera>,
    scene: Arc<Scene>,
    samples_per_pixel: i32,
    max_depth: u32,
    color_tx: Sender<Vec<(u32, u32, Rgba<u8>)>>,
) -> ComputeContext {
    let (cancel_tx, cancel_rx) = mpsc::channel();
    let range = range.to_owned();
    let handle = thread::spawn(move || {
        let mut rng = rand::thread_rng();

        let mut line = Vec::<(u32, u32, Rgba<u8>)>::new();
        for i in range {
            let x = i % WIDTH;
            let y = i / WIDTH;

            let mut color = Vec3::default();
            for _ in 0..samples_per_pixel {
                let u = (rng.gen::<f64>() + x as f64) / (WIDTH - 1) as f64;
                let v = (rng.gen::<f64>() + y as f64) / (HEIGHT - 1) as f64;
                let ray = camera.ray(u, v);
                color += raycast(&ray, &scene, max_depth);
            }
            line.push((x, y, fragment_to_pixel(color, samples_per_pixel)));

            if cancel_rx.try_recv().is_ok() {
                break;
            }

            if i % (20 * WIDTH) == 0 {
                color_tx.send(line.clone()).unwrap();
                line.clear();
            }
        }

        if !line.is_empty() {
            color_tx.send(line.clone()).unwrap();
        }
    });
    ComputeContext {
        handle,
        cancel: cancel_tx,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut frame_buffer = ImageBuffer::from_pixel(WIDTH, HEIGHT, image::Rgba([0, 0, 0, 255]));

    let mut window: PistonWindow = WindowSettings::new("Raytracer", [WIDTH, HEIGHT])
        .exit_on_esc(true)
        .build()?;

    let mut texture_context = window.create_texture_context();
    let texture_settings = TextureSettings::new();

    let mut texture = Texture::from_image(&mut texture_context, &frame_buffer, &texture_settings)?;

    let (color_tx, color_rx) = mpsc::channel();

    let camera = Arc::new(Camera::new(
        Vec3(-2.0, 2.0, 1.0),
        Vec3(0.0, 0.0, -1.0),
        Vec3(0.0, 1.0, 0.0),
        20.0,
        ASPECT_RATIO,
    ));
    let samples_per_pixel = 128;
    let max_depth = 64;

    let material_ground = Arc::new(Lambertian {
        albedo: Vec3(0.8, 0.8, 0.0),
    });
    let material_center = Arc::new(Lambertian {
        albedo: Vec3(0.1, 0.2, 0.5),
    });
    let material_left = Arc::new(Dielectric { ir: 1.5 });
    let material_right = Arc::new(Metal {
        albedo: Vec3(0.8, 0.6, 0.2),
        fuzz: 0.0,
    });

    let scene = Arc::new(Scene {
        components: vec![
            Arc::new(Sphere {
                center: Vec3(0.0, -100.5, -1.0),
                radius: 100.0,
                material: material_ground.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3(0.0, 0.0, -1.0),
                radius: 0.5,
                material: material_center.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3(-1.0, 0.0, -1.0),
                radius: 0.5,
                material: material_left.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3(-1.0, 0.0, -1.0),
                radius: -0.45,
                material: material_left.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3(1.0, 0.0, -1.0),
                radius: 0.5,
                material: material_right.clone(),
            }),
        ],
    });
    let mut contexts = Vec::new();
    let full: Vec<_> = (0..WIDTH * HEIGHT).collect();
    let chunk_size = (WIDTH * HEIGHT / 8).try_into().unwrap();

    for chunk in full.chunks(chunk_size) {
        let context = compute(
            chunk,
            camera.clone(),
            scene.clone(),
            samples_per_pixel,
            max_depth,
            color_tx.clone(),
        );
        contexts.push(context);
    }

    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g, device| {
            if let Ok(line) = color_rx.try_recv() {
                for (x, y, px) in line {
                    frame_buffer.put_pixel(x, y, px);
                }
                texture.update(&mut texture_context, &frame_buffer).unwrap();
                texture_context.encoder.flush(device);
            }
            clear([1.0; 4], g);
            image(&texture, c.transform, g);
        });
    }

    for context in contexts {
        let _cancel = context.cancel.send(());
        context.handle.join().unwrap();
    }

    Ok(())
}