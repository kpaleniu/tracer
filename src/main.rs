extern crate image;
extern crate piston_window;

use core::f32::consts::PI;
use glam::Vec3A as Vec3;
use image::ImageBuffer;
use image::Rgba;
use piston_window::*;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::Rng;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::ops::Add;
use std::sync::mpsc::channel;
use std::sync::Arc;
use std::time::Instant;

const ASPECT_RATIO: f32 = 16.0 / 9.0;
const WIDTH: u32 = 640;
const HEIGHT: u32 = (WIDTH as f32 / ASPECT_RATIO) as u32;

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    v - 2.0 * v.dot(n) * n
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = (-uv).dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + n * cos_theta);
    let a = 1.0 - r_out_perp.length_squared();
    let b = a.abs().sqrt();
    let r_out_parallel = n * -b;
    r_out_perp + r_out_parallel
}

fn near_zero(v: Vec3) -> bool {
    let s = 1e-8;
    v.x.abs() < s && v.y.abs() < s && v.z.abs() < s
}

#[derive(Debug)]
pub struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + t * self.direction
    }
}

fn fragment_to_pixel(frag: Vec3, samples_per_pixel: i32) -> Rgba<u8> {
    let scale = 1.0 / samples_per_pixel as f32;
    let r = frag.x * scale;
    let g = frag.y * scale;
    let b = frag.z * scale;

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
        vertical_fov: f32,
        aspect_ratio: f32,
    ) -> Self {
        let h = (vertical_fov.deg_to_rad() / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).normalize();
        let u = vertical_up.cross(-w).normalize();
        let v = w.cross(u);

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

    fn ray(&self, u: f32, v: f32) -> Ray {
        let direction = self.bottom_left + u * self.horizontal + v * self.vertical - self.origin;
        Ray::new(self.origin, direction)
    }
}

#[derive(Clone)]
struct Hit {
    point: Vec3,
    normal: Vec3,
    material: Arc<dyn Material + Send + Sync>,
    t: f32,
    front_face: bool,
}

trait Hittable {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit>;
}

#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: f32,
    material: Arc<dyn Material + Send + Sync>,
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
        let oc = r.origin - self.center;
        let a = r.direction.length_squared();
        let half_b = oc.dot(r.direction);
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
        let (normal, front_face) = if r.direction.dot(outward_normal) < 0.0 {
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
        if near_zero(scatter_direction) {
            scatter_direction = hit.normal;
        }
        let scattered = Ray::new(hit.point, scatter_direction);
        (scattered, self.albedo, true)
    }
}

struct Metal {
    albedo: Vec3,
    fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> (Ray, Vec3, bool) {
        let reflected = reflect(r_in.direction.normalize(), hit.normal);
        let scattered = Ray::new(hit.point, reflected + self.fuzz * random_in_unit_sphere());
        let did_scatter = scattered.direction.dot(hit.normal) > 0.0;
        (scattered, self.albedo, did_scatter)
    }
}

struct Dielectric {
    ir: f32,
}

impl Dielectric {
    fn reflectance(cosine: f32, ref_index: f32) -> f32 {
        let r0 = (1.0 - ref_index) / (1.0 + ref_index);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, r_in: &Ray, hit: &Hit) -> (Ray, Vec3, bool) {
        let attentuation = Vec3::new(1.0, 1.0, 1.0);
        let refraction_ratio = if hit.front_face {
            1.0 / self.ir
        } else {
            self.ir
        };
        let unit_direction = r_in.direction.normalize();
        let cos_theta = (-unit_direction).dot(hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let mut rng = rand::thread_rng();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction =
            if cannot_refract || Self::reflectance(cos_theta, refraction_ratio) > rng.gen() {
                reflect(unit_direction, hit.normal)
            } else {
                refract(unit_direction, hit.normal, refraction_ratio)
            };
        let scattered = Ray::new(hit.point, direction);
        (scattered, attentuation, true)
    }
}

struct Scene {
    components: Vec<Arc<dyn Hittable + Send + Sync>>,
}

impl Scene {
    fn hit(&self, r: &Ray, t_min: f32, t_max: f32) -> Option<Hit> {
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
    Vec3::new(x, y, z)
}

fn raycast(r: &Ray, scene: &Scene, depth: u32) -> Vec3 {
    if depth == 0 {
        return Vec3::ZERO;
    }

    if let Some(hit) = scene.hit(r, 0.0001, f32::INFINITY) {
        let (ray, attenuation, did_scatter) = hit.material.scatter(r, &hit);
        if did_scatter {
            return attenuation * raycast(&ray, scene, depth - 1);
        }
        return Vec3::new(0.0, 0.0, 0.0);
    }

    let d = r.direction.normalize();
    let t = 0.5 * (d.y + 1.0);

    // blend black and blue
    (1.0 - t) * Vec3::ONE + t * Vec3::new(0.5, 0.7, 1.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut frame_buffer = ImageBuffer::from_pixel(WIDTH, HEIGHT, image::Rgba([0, 0, 0, 255]));

    let mut window: PistonWindow = WindowSettings::new("Raytracer", [WIDTH, HEIGHT])
        .exit_on_esc(true)
        .build()?;

    let mut texture_context = window.create_texture_context();
    let texture_settings = TextureSettings::new();

    let mut texture = Texture::from_image(&mut texture_context, &frame_buffer, &texture_settings)?;

    let assets = find_folder::Search::ParentsThenKids(3, 3)
        .for_folder("assets")
        .unwrap();

    let mut glyphs = window
        .load_font(assets.join("FiraSans-Regular.ttf"))
        .unwrap();

    let camera = Arc::new(Camera::new(
        Vec3::new(-2.0, 2.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 1.0, 0.0),
        20.0,
        ASPECT_RATIO,
    ));
    let samples_per_pixel = 128;
    let max_depth = 64;

    let material_ground = Arc::new(Lambertian {
        albedo: Vec3::new(0.8, 0.8, 0.0),
    });
    let material_center = Arc::new(Lambertian {
        albedo: Vec3::new(0.1, 0.2, 0.5),
    });
    let material_left = Arc::new(Dielectric { ir: 1.5 });
    let material_right = Arc::new(Metal {
        albedo: Vec3::new(0.8, 0.6, 0.2),
        fuzz: 0.0,
    });

    let scene = Arc::new(Scene {
        components: vec![
            Arc::new(Sphere {
                center: Vec3::new(0.0, -100.5, -1.0),
                radius: 100.0,
                material: material_ground.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3::new(0.0, 0.0, -1.0),
                radius: 0.5,
                material: material_center.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3::new(-1.0, 0.0, -1.0),
                radius: 0.5,
                material: material_left.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3::new(-1.0, 0.0, -1.0),
                radius: -0.45,
                material: material_left.clone(),
            }),
            Arc::new(Sphere {
                center: Vec3::new(1.0, 0.0, -1.0),
                radius: 0.5,
                material: material_right.clone(),
            }),
        ],
    });

    let (tx, rx) = channel();
    rayon::spawn(move || {
        let now = Instant::now();
        let _res = (0..WIDTH * HEIGHT)
            .into_par_iter()
            .map(|i| {
                let x = i % WIDTH;
                let y = i / WIDTH;
                let color = (0..samples_per_pixel)
                    .into_par_iter()
                    .map_init(
                        || rand::thread_rng(),
                        |rng, _| {
                            let u = (rng.gen::<f32>() + x as f32) / (WIDTH - 1) as f32;
                            let v = (rng.gen::<f32>() + y as f32) / (HEIGHT - 1) as f32;
                            let ray = camera.ray(u, v);
                            raycast(&ray, &scene, max_depth)
                        },
                    )
                    .reduce_with(Add::add)
                    .unwrap();
                (x, y, fragment_to_pixel(color, samples_per_pixel))
            })
            .try_for_each_with(tx, move |tx, chunk| tx.send(chunk));
        println!("took {}s", now.elapsed().as_secs());
    });

    let mut show_frame_time = false;
    while let Some(e) = window.next() {
        if let Some(Button::Keyboard(key)) = e.press_args() {
            if key == Key::F {
                show_frame_time = !show_frame_time;
            }
        };

        window.draw_2d(&e, |c, g, device| {
            let mut updated = false;
            let mut iter = rx.try_iter();
            let now = Instant::now();
            while let Some((x, y, px)) = iter.next() {
                if now.elapsed().as_millis() > 16 {
                    break;
                }
                frame_buffer.put_pixel(x, y, px);
                updated = true;
            }
            if updated {
                texture.update(&mut texture_context, &frame_buffer).unwrap();
                texture_context.encoder.flush(device);
            }
            clear([1.0; 4], g);
            image(&texture, c.transform, g);

            if show_frame_time {
                let transform = c.transform.trans(10.0, 100.0);
                let millis = format!("{} us", now.elapsed().as_micros());
                text::Text::new_color([1.0, 1.0, 1.0, 1.0], 32)
                    .draw(&millis, &mut glyphs, &c.draw_state, transform, g)
                    .unwrap();

                // Update glyphs before rendering.
                glyphs.factory.encoder.flush(device);
            }
        });
    }

    Ok(())
}
