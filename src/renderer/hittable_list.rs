use crate::renderer::hit_record::HitRecord;
use crate::renderer::ray::Hittable;
use crate::renderer::ray::Ray;

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
