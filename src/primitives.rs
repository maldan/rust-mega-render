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
