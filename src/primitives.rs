use super::Mesh;

pub fn plane(w: f32, h: f32) -> Mesh {
    let hw = w * 0.5;
    let hh = h * 0.5;
    let n = [0.0, 1.0, 0.0];
    Mesh::new(
        vec![
            [-hw, 0.0, hh],
            [hw, 0.0, hh],
            [hw, 0.0, -hh],
            [-hw, 0.0, -hh],
        ],
        vec![n, n, n, n],
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![0, 1, 2, 0, 2, 3],
    )
}

pub fn cube(size: f32) -> Mesh {
    let s = size * 0.5;
    let positions = vec![
        // +Y
        [-s, s, s],
        [s, s, s],
        [s, s, -s],
        [-s, s, -s],
        // -Y
        [-s, -s, -s],
        [s, -s, -s],
        [s, -s, s],
        [-s, -s, s],
        // +Z
        [-s, -s, s],
        [s, -s, s],
        [s, s, s],
        [-s, s, s],
        // -Z
        [s, -s, -s],
        [-s, -s, -s],
        [-s, s, -s],
        [s, s, -s],
        // +X
        [s, -s, s],
        [s, -s, -s],
        [s, s, -s],
        [s, s, s],
        // -X
        [-s, -s, -s],
        [-s, -s, s],
        [-s, s, s],
        [-s, s, -s],
    ];
    let face_n = [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    let mut normals = Vec::with_capacity(24);
    for n in face_n {
        normals.extend_from_slice(&[n, n, n, n]);
    }
    let uv = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let mut uvs = Vec::with_capacity(24);
    for _ in 0..6 {
        uvs.extend_from_slice(&uv);
    }
    let mut indices = Vec::with_capacity(36);
    for f in 0..6u32 {
        let i = f * 4;
        indices.extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]);
    }
    Mesh::new(positions, normals, uvs, indices)
}

/// Cube with `segs×segs` quads per face (hard edges, planar UV).
pub fn cube_subdiv(size: f32, segs: u32) -> Mesh {
    let s = size * 0.5;
    let segs = segs.max(1);
    let faces: [([[f32; 3]; 4], [f32; 3]); 6] = [
        ([[-s, s, s], [s, s, s], [s, s, -s], [-s, s, -s]], [0.0, 1.0, 0.0]),
        (
            [[-s, -s, -s], [s, -s, -s], [s, -s, s], [-s, -s, s]],
            [0.0, -1.0, 0.0],
        ),
        (
            [[-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]],
            [0.0, 0.0, 1.0],
        ),
        (
            [[s, -s, -s], [-s, -s, -s], [-s, s, -s], [s, s, -s]],
            [0.0, 0.0, -1.0],
        ),
        (
            [[s, -s, s], [s, -s, -s], [s, s, -s], [s, s, s]],
            [1.0, 0.0, 0.0],
        ),
        (
            [[-s, -s, -s], [-s, -s, s], [-s, s, s], [-s, s, -s]],
            [-1.0, 0.0, 0.0],
        ),
    ];
    let uv_c = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let cols = segs + 1;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();
    for (corners, n) in faces {
        let base = positions.len() as u32;
        for v in 0..cols {
            let tv = v as f32 / segs as f32;
            for u in 0..cols {
                let tu = u as f32 / segs as f32;
                let a = lerp3(corners[0], corners[1], tu);
                let b = lerp3(corners[3], corners[2], tu);
                positions.push(lerp3(a, b, tv));
                normals.push(n);
                let ua = lerp2(uv_c[0], uv_c[1], tu);
                let ub = lerp2(uv_c[3], uv_c[2], tu);
                uvs.push(lerp2(ua, ub, tv));
            }
        }
        for y in 0..segs {
            for x in 0..segs {
                let i = base + y * cols + x;
                indices.extend_from_slice(&[i, i + 1, i + cols + 1, i, i + cols + 1, i + cols]);
            }
        }
    }
    Mesh::new(positions, normals, uvs, indices)
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

pub fn sphere(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let sectors = sectors.max(3);
    let stacks = stacks.max(2);
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for y in 0..=stacks {
        let v = y as f32 / stacks as f32;
        let phi = v * std::f32::consts::PI;
        let (sp, cp) = phi.sin_cos();
        for x in 0..=sectors {
            let u = x as f32 / sectors as f32;
            let theta = u * std::f32::consts::TAU;
            let (st, ct) = theta.sin_cos();
            let n = [st * sp, cp, ct * sp];
            positions.push([radius * n[0], radius * n[1], radius * n[2]]);
            normals.push(n);
            uvs.push([u, v]);
        }
    }
    let mut indices = Vec::new();
    let row = sectors + 1;
    for y in 0..stacks {
        for x in 0..sectors {
            let i = y * row + x;
            let a = i;
            let b = i + row;
            // CW from outside (LH / FrontFace::Cw)
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    Mesh::new(positions, normals, uvs, indices)
}
