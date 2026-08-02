use glam::Vec3;

pub struct Mesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    /// xyz = tangent, w = bitangent sign
    pub tangents: Vec<[f32; 4]>,
    pub joints: Option<Vec<[u16; 4]>>,
    pub weights: Option<Vec<[f32; 4]>>,
    pub indices: Vec<u32>,
    pub version: u64,
}

impl Mesh {
    pub fn new(
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
    ) -> Self {
        let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
        Self {
            positions,
            normals,
            uvs,
            tangents,
            joints: None,
            weights: None,
            indices,
            version: 1,
        }
    }

    pub fn mark_changed(&mut self) {
        self.tangents = compute_tangents(&self.positions, &self.normals, &self.uvs, &self.indices);
        self.version += 1;
    }
}

fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let n = positions.len();
    let mut tan1 = vec![Vec3::ZERO; n];
    let mut tan2 = vec![Vec3::ZERO; n];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let uv0 = uvs.get(i0).copied().unwrap_or([0.0, 0.0]);
        let uv1 = uvs.get(i1).copied().unwrap_or([0.0, 0.0]);
        let uv2 = uvs.get(i2).copied().unwrap_or([0.0, 0.0]);

        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let du1 = uv1[0] - uv0[0];
        let dv1 = uv1[1] - uv0[1];
        let du2 = uv2[0] - uv0[0];
        let dv2 = uv2[1] - uv0[1];
        let det = du1 * dv2 - du2 * dv1;
        if det.abs() < 1e-8 {
            continue;
        }
        let r = 1.0 / det;
        let sdir = (e1 * dv2 - e2 * dv1) * r;
        let tdir = (e2 * du1 - e1 * du2) * r;
        tan1[i0] += sdir;
        tan1[i1] += sdir;
        tan1[i2] += sdir;
        tan2[i0] += tdir;
        tan2[i1] += tdir;
        tan2[i2] += tdir;
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let nrm = Vec3::from_array(normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]));
        let t = tan1[i];
        let mut tangent = t - nrm * nrm.dot(t);
        if tangent.length_squared() < 1e-8 {
            let helper = if nrm.x.abs() > 0.9 {
                Vec3::Y
            } else {
                Vec3::X
            };
            tangent = nrm.cross(helper);
        }
        let tangent = tangent.normalize_or_zero();
        let w = if nrm.cross(tangent).dot(tan2[i]) < 0.0 {
            -1.0
        } else {
            1.0
        };
        out.push([tangent.x, tangent.y, tangent.z, w]);
    }
    out
}
