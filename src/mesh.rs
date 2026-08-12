use glam::Vec3;

/// Relative morph / shape-key target (glTF-style deltas from basis).
#[derive(Clone, Debug)]
pub struct MorphTarget {
    pub name: String,
    /// Per-vertex position deltas; length must match basis vertex count.
    pub position_deltas: Vec<[f32; 3]>,
    /// Optional normal deltas. Empty → normals rebuilt from positions after blend.
    pub normal_deltas: Vec<[f32; 3]>,
}

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
    /// Rest positions. Empty until morphs are used (then captured from `positions`).
    pub basis_positions: Vec<[f32; 3]>,
    /// Rest normals (same lifecycle as `basis_positions`).
    pub basis_normals: Vec<[f32; 3]>,
    pub morph_targets: Vec<MorphTarget>,
    /// Weight per morph target in `[0, 1]` (clamped on apply).
    pub morph_weights: Vec<f32>,
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
            basis_positions: Vec::new(),
            basis_normals: Vec::new(),
            morph_targets: Vec::new(),
            morph_weights: Vec::new(),
        }
    }

    pub fn mark_changed(&mut self) {
        self.tangents = compute_tangents(&self.positions, &self.normals, &self.uvs, &self.indices);
        self.version += 1;
    }

    /// Capture current `positions`/`normals` as basis if not yet stored.
    pub fn ensure_basis(&mut self) {
        if self.basis_positions.is_empty() {
            self.basis_positions = self.positions.clone();
            self.basis_normals = if self.normals.len() == self.positions.len() {
                self.normals.clone()
            } else {
                vec![[0.0, 1.0, 0.0]; self.positions.len()]
            };
        }
    }

    /// Add a zero-delta shape key. Returns its index.
    pub fn add_shape_key(&mut self, name: impl Into<String>) -> usize {
        self.ensure_basis();
        let n = self.basis_positions.len();
        self.morph_targets.push(MorphTarget {
            name: name.into(),
            position_deltas: vec![[0.0, 0.0, 0.0]; n],
            normal_deltas: Vec::new(),
        });
        self.morph_weights.push(0.0);
        self.morph_targets.len() - 1
    }

    pub fn remove_shape_key(&mut self, index: usize) {
        if index >= self.morph_targets.len() {
            return;
        }
        self.morph_targets.remove(index);
        if index < self.morph_weights.len() {
            self.morph_weights.remove(index);
        }
        if self.morph_targets.is_empty() {
            // Restore basis into display buffers.
            if !self.basis_positions.is_empty() {
                self.positions = self.basis_positions.clone();
                self.normals = self.basis_normals.clone();
                self.mark_changed();
            }
        } else {
            self.apply_morphs();
        }
    }

    pub fn set_morph_weight(&mut self, index: usize, weight: f32) {
        if let Some(w) = self.morph_weights.get_mut(index) {
            *w = weight.clamp(0.0, 1.0);
            self.apply_morphs();
        }
    }

    /// `positions` / `normals` = basis + Σ weightᵢ · Δᵢ. Bumps `version`.
    pub fn apply_morphs(&mut self) {
        self.ensure_basis();
        let n = self.basis_positions.len();
        if n == 0 {
            return;
        }

        if self.morph_targets.is_empty() {
            self.positions = self.basis_positions.clone();
            self.normals = self.basis_normals.clone();
            self.mark_changed();
            return;
        }

        // Keep weight vec in sync.
        while self.morph_weights.len() < self.morph_targets.len() {
            self.morph_weights.push(0.0);
        }
        self.morph_weights.truncate(self.morph_targets.len());

        let mut positions = self.basis_positions.clone();
        let use_normal_deltas = self
            .morph_targets
            .iter()
            .any(|t| t.normal_deltas.len() == n);
        let mut normals = if use_normal_deltas {
            self.basis_normals.clone()
        } else {
            Vec::new()
        };

        for (ti, target) in self.morph_targets.iter().enumerate() {
            let w = self.morph_weights.get(ti).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            if w.abs() < 1e-8 {
                continue;
            }
            let count = target.position_deltas.len().min(n);
            for i in 0..count {
                let d = target.position_deltas[i];
                positions[i][0] += d[0] * w;
                positions[i][1] += d[1] * w;
                positions[i][2] += d[2] * w;
            }
            if use_normal_deltas && target.normal_deltas.len() == n {
                for i in 0..n {
                    let d = target.normal_deltas[i];
                    normals[i][0] += d[0] * w;
                    normals[i][1] += d[1] * w;
                    normals[i][2] += d[2] * w;
                }
            }
        }

        if use_normal_deltas {
            for nrm in &mut normals {
                let v = Vec3::from_array(*nrm);
                *nrm = if v.length_squared() > 1e-10 {
                    v.normalize().to_array()
                } else {
                    [0.0, 1.0, 0.0]
                };
            }
        } else {
            // Weld by position so UV-split verts keep smooth shading (no seam hard edges).
            normals = recompute_normals_welded(&positions, &self.indices);
        }

        self.positions = positions;
        self.normals = normals;
        self.mark_changed();
    }

    /// Add a mesh-local offset into one shape key's position deltas (sculpt helper).
    pub fn add_shape_delta(&mut self, target: usize, vertex: usize, delta: [f32; 3]) {
        let Some(t) = self.morph_targets.get_mut(target) else {
            return;
        };
        if vertex >= t.position_deltas.len() {
            return;
        }
        let d = &mut t.position_deltas[vertex];
        d[0] += delta[0];
        d[1] += delta[1];
        d[2] += delta[2];
    }
}

/// Face-area normals, then average across vertices that share the same position
/// (UV / sharp-edge splits). Prevents hard creases along UV seams after morph.
fn recompute_normals_welded(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let n = positions.len();
    let mut acc = vec![Vec3::ZERO; n];
    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= n || i1 >= n || i2 >= n {
            continue;
        }
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let face = (p1 - p0).cross(p2 - p0);
        acc[i0] += face;
        acc[i1] += face;
        acc[i2] += face;
    }

    // Quantized position → list of vertex indices (UV splits share a cell).
    use std::collections::HashMap;
    let mut groups: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    const Q: f32 = 1e5; // ~0.01mm in meters; stable for face-scale models
    for (i, p) in positions.iter().enumerate() {
        let key = (
            (p[0] * Q).round() as i32,
            (p[1] * Q).round() as i32,
            (p[2] * Q).round() as i32,
        );
        groups.entry(key).or_default().push(i);
    }

    let mut out = vec![[0.0, 1.0, 0.0]; n];
    for idxs in groups.values() {
        let mut sum = Vec3::ZERO;
        for &i in idxs {
            sum += acc[i];
        }
        let nrm = if sum.length_squared() > 1e-12 {
            sum.normalize().to_array()
        } else {
            [0.0, 1.0, 0.0]
        };
        for &i in idxs {
            out[i] = nrm;
        }
    }
    out
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
