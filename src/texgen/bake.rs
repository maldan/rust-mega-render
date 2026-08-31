//! Bake graph maps into a [`crate::Material`] on a scene.

use crate::material::{HeightMode, Material, MaterialMaps};
use crate::scene::Scene;
use crate::store::Handle;
use crate::texture::Texture;

use super::gpu::GpuEval;
use super::graph::TexGraph;
use crate::io::material::MaterialBytesError;
use crate::MaterialFile;
/// Insert albedo / MR / normal / height textures and a standard PBR material.
pub fn insert_maps(
    scene: &mut Scene,
    albedo: Vec<u8>,
    mr: Vec<u8>,
    normal: Vec<u8>,
    height: Vec<u8>,
    res: u32,
    tess_factor: u32,
    displacement: f32,
    height_mode: HeightMode,
) -> Handle<Material> {
    let res = res.max(1);
    let albedo_h = scene.textures.insert(tex(res, albedo, true));
    let mr_h = scene.textures.insert(tex(res, mr, false));
    let nrm_h = scene.textures.insert(tex(res, normal, false));
    let hgt_h = scene.textures.insert(tex(res, height, false));
    let mut mat = Material::new([1.0, 1.0, 1.0, 1.0], 1.0, 1.0);
    mat.maps = MaterialMaps::Single {
        albedo: Some(albedo_h),
        normal: Some(nrm_h),
        metallic_roughness: Some(mr_h),
    };
    if displacement > 0.0 {
        mat.height = Some(hgt_h);
        mat.displacement_scale = displacement;
        mat.tess_factor = tess_factor.clamp(1, 32);
        mat.height_mode = height_mode;
    }
    scene.materials.insert(mat)
}

fn tex(res: u32, rgba: Vec<u8>, srgb: bool) -> Texture {
    Texture {
        id: Texture::new_id(),
        width: res,
        height: res,
        rgba,
        version: 1,
        srgb,
        dirty: None,
        gpu_resident: false,
    }
}

impl GpuEval {
    /// Evaluate `graph` and insert a material into `scene`.
    pub fn bake_into_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        graph: &TexGraph,
        res: u32,
    ) -> Handle<Material> {
        let ((albedo, mr, normal, height), _) = self.eval_material(
            device,
            queue,
            &graph.nodes,
            &graph.links,
            &graph.output_id,
            res,
        );
        let (tess, disp, mode) = graph
            .output()
            .map(|n| {
                (
                    n.tess_factor.clamp(1, 32) as u32,
                    n.displacement.max(0.0),
                    n.height_mode,
                )
            })
            .unwrap_or((32, 0.0, HeightMode::Tessellate));
        insert_maps(
            scene, albedo, mr, normal, height, res, tess, disp, mode,
        )
    }

    /// Bake a procedural `MAT ` recipe into `scene`. Needs a `TEXG` chunk.
    pub fn instantiate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &mut Scene,
        file: &MaterialFile,
    ) -> Result<Handle<Material>, MaterialBytesError> {
        let Some(proc) = &file.graph else {
            return Err(MaterialBytesError::NotProcedural);
        };
        let h = self.bake_into_scene(device, queue, scene, &proc.graph, proc.resolution);
        if let Some(mat) = scene.materials.get_mut(h) {
            mat.albedo = file.albedo;
            mat.metallic = file.metallic;
            mat.roughness = file.roughness;
            mat.sss_strength = file.sss_strength;
            mat.sss_color = file.sss_color;
            mat.sss_curvature = file.sss_curvature;
            mat.alpha_cutoff = file.alpha_cutoff;
            mat.shading_model = file.shading_model;
        }
        Ok(h)
    }
}
