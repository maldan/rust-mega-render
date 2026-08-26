use crate::mesh::{compute_tangents, Mesh, MorphTarget};

const MAGIC: &[u8; 4] = b"MESH";
const VERSION: u16 = 1;
const FLAG_QUANTIZED: u16 = 1 << 0;
const MAX_SLOT: u32 = 255;

const ID_POS: [u8; 4] = *b"POS ";
const ID_NRM: [u8; 4] = *b"NRM ";
const ID_UV: [u8; 4] = *b"UV  ";
const ID_TAN: [u8; 4] = *b"TAN ";
const ID_COL: [u8; 4] = *b"COL ";
const ID_JNT: [u8; 4] = *b"JNT ";
const ID_WGT: [u8; 4] = *b"WGT ";
const ID_IDX: [u8; 4] = *b"IDX ";
const ID_MRPH: [u8; 4] = *b"MRPH";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshBytesError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u16),
    Quantized,
    Duplicate(&'static str),
    DuplicateSlot { channel: &'static str, slot: u32 },
    SizeMismatch,
    VertexCountMismatch,
    BadIndices,
    MorphWithoutPositions,
    BadUtf8,
    SlotTooLarge(u32),
}

impl std::fmt::Display for MeshBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated MESH blob"),
            Self::BadMagic => write!(f, "not a MESH blob"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported MESH version {v}"),
            Self::Quantized => write!(f, "QUANTIZED MESH is not supported in version 1"),
            Self::Duplicate(id) => write!(f, "duplicate {id} chunk"),
            Self::DuplicateSlot { channel, slot } => {
                write!(f, "duplicate {channel} slot {slot}")
            }
            Self::SizeMismatch => write!(f, "chunk size does not match count"),
            Self::VertexCountMismatch => write!(f, "vertex attribute counts differ"),
            Self::BadIndices => write!(f, "invalid index buffer"),
            Self::MorphWithoutPositions => write!(f, "MRPH chunk without POS"),
            Self::BadUtf8 => write!(f, "morph name is not UTF-8"),
            Self::SlotTooLarge(s) => write!(f, "attribute slot {s} exceeds 255"),
        }
    }
}

impl std::error::Error for MeshBytesError {}

pub fn to_bytes(mesh: &Mesh) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());

    let positions = if mesh.basis_positions.is_empty() {
        mesh.positions.as_slice()
    } else {
        mesh.basis_positions.as_slice()
    };
    let normals = if mesh.basis_normals.is_empty() {
        mesh.normals.as_slice()
    } else {
        mesh.basis_normals.as_slice()
    };
    // Tangents on the live mesh follow the blended pose; omit when rest ≠ display.
    let write_tan = mesh.basis_positions.is_empty() && !mesh.tangents.is_empty();

    write_unslotted(&mut out, ID_POS, 12, positions.len(), |p| {
        write_f32s(p, positions.as_flattened());
    });
    write_unslotted(&mut out, ID_NRM, 12, normals.len(), |p| {
        write_f32s(p, normals.as_flattened());
    });
    for (slot, col) in mesh.uvs.iter().enumerate() {
        write_slotted(&mut out, ID_UV, slot as u32, 8, col.len(), |p| {
            write_f32s(p, col.as_flattened());
        });
    }
    if write_tan {
        write_unslotted(&mut out, ID_TAN, 16, mesh.tangents.len(), |p| {
            write_f32s(p, mesh.tangents.as_flattened());
        });
    }
    for (slot, col) in mesh.colors.iter().enumerate() {
        write_slotted(&mut out, ID_COL, slot as u32, 16, col.len(), |p| {
            write_f32s(p, col.as_flattened());
        });
    }
    for (slot, col) in mesh.joints.iter().enumerate() {
        write_slotted(&mut out, ID_JNT, slot as u32, 8, col.len(), |p| {
            for j in col {
                for c in j {
                    p.extend_from_slice(&c.to_le_bytes());
                }
            }
        });
    }
    for (slot, col) in mesh.weights.iter().enumerate() {
        write_slotted(&mut out, ID_WGT, slot as u32, 16, col.len(), |p| {
            write_f32s(p, col.as_flattened());
        });
    }
    write_unslotted(&mut out, ID_IDX, 4, mesh.indices.len(), |p| {
        for i in &mesh.indices {
            p.extend_from_slice(&i.to_le_bytes());
        }
    });
    for target in &mesh.morph_targets {
        write_morph(&mut out, target);
    }
    out
}

fn write_unslotted(
    out: &mut Vec<u8>,
    id: [u8; 4],
    stride: usize,
    count: usize,
    fill: impl FnOnce(&mut Vec<u8>),
) {
    if count == 0 {
        return;
    }
    let mut payload = Vec::with_capacity(4 + count * stride);
    payload.extend_from_slice(&(count as u32).to_le_bytes());
    fill(&mut payload);
    write_chunk(out, id, &payload);
}

fn write_slotted(
    out: &mut Vec<u8>,
    id: [u8; 4],
    slot: u32,
    stride: usize,
    count: usize,
    fill: impl FnOnce(&mut Vec<u8>),
) {
    if count == 0 {
        return;
    }
    let mut payload = Vec::with_capacity(8 + count * stride);
    payload.extend_from_slice(&slot.to_le_bytes());
    payload.extend_from_slice(&(count as u32).to_le_bytes());
    fill(&mut payload);
    write_chunk(out, id, &payload);
}

fn write_morph(out: &mut Vec<u8>, target: &MorphTarget) {
    if target.position_deltas.is_empty() && target.normal_deltas.is_empty() {
        return;
    }
    let name = target.name.as_bytes();
    let name_len = u16::try_from(name.len()).unwrap_or(u16::MAX);
    let name = &name[..name_len as usize];
    let name_section = 2 + name.len();
    let name_pad = (4 - (name_section % 4)) % 4;
    let has_nrm = !target.normal_deltas.is_empty();
    let count = if has_nrm {
        target.position_deltas.len().min(target.normal_deltas.len())
    } else {
        target.position_deltas.len()
    };
    if count == 0 {
        return;
    }

    let mut payload = Vec::new();
    payload.extend_from_slice(&name_len.to_le_bytes());
    payload.extend_from_slice(name);
    payload.resize(name_section + name_pad, 0);
    let flags: u32 = if has_nrm { 1 } else { 0 };
    payload.extend_from_slice(&flags.to_le_bytes());
    payload.extend_from_slice(&(count as u32).to_le_bytes());
    write_f32s(&mut payload, target.position_deltas[..count].as_flattened());
    if has_nrm {
        write_f32s(&mut payload, target.normal_deltas[..count].as_flattened());
    }
    write_chunk(out, ID_MRPH, &payload);
}

fn write_chunk(out: &mut Vec<u8>, id: [u8; 4], payload: &[u8]) {
    out.extend_from_slice(&id);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    while out.len() % 4 != 0 {
        out.push(0);
    }
}

fn write_f32s(out: &mut Vec<u8>, xs: &[f32]) {
    out.reserve(xs.len() * 4);
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], MeshBytesError> {
        if self.remaining() < n {
            return Err(MeshBytesError::Truncated);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16, MeshBytesError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, MeshBytesError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn read_f32s(data: &[u8]) -> Result<Vec<f32>, MeshBytesError> {
    if data.len() % 4 != 0 {
        return Err(MeshBytesError::SizeMismatch);
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for c in data.chunks_exact(4) {
        out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(out)
}

fn pack3(v: &[f32]) -> Result<Vec<[f32; 3]>, MeshBytesError> {
    if v.len() % 3 != 0 {
        return Err(MeshBytesError::SizeMismatch);
    }
    Ok(v.chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect())
}

fn pack2(v: &[f32]) -> Result<Vec<[f32; 2]>, MeshBytesError> {
    if v.len() % 2 != 0 {
        return Err(MeshBytesError::SizeMismatch);
    }
    Ok(v.chunks_exact(2).map(|c| [c[0], c[1]]).collect())
}

fn pack4(v: &[f32]) -> Result<Vec<[f32; 4]>, MeshBytesError> {
    if v.len() % 4 != 0 {
        return Err(MeshBytesError::SizeMismatch);
    }
    Ok(v.chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect())
}

fn note_count(n_verts: &mut Option<u32>, n: u32) -> Result<(), MeshBytesError> {
    if n == 0 {
        return Ok(());
    }
    match *n_verts {
        None => {
            *n_verts = Some(n);
            Ok(())
        }
        Some(v) if v == n => Ok(()),
        Some(_) => Err(MeshBytesError::VertexCountMismatch),
    }
}

fn put_column<T: Default>(cols: &mut Vec<Vec<T>>, slot: u32, data: Vec<T>) -> Result<(), MeshBytesError> {
    if slot > MAX_SLOT {
        return Err(MeshBytesError::SlotTooLarge(slot));
    }
    let i = slot as usize;
    if cols.len() <= i {
        cols.resize_with(i + 1, Vec::new);
    }
    if !cols[i].is_empty() {
        return Err(MeshBytesError::DuplicateSlot {
            channel: "slot",
            slot,
        });
    }
    cols[i] = data;
    Ok(())
}

pub fn from_bytes(bytes: &[u8]) -> Result<Mesh, MeshBytesError> {
    if bytes.len() < 8 {
        return Err(MeshBytesError::Truncated);
    }
    if &bytes[0..4] != MAGIC {
        return Err(MeshBytesError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != VERSION {
        return Err(MeshBytesError::UnsupportedVersion(version));
    }
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    if flags & FLAG_QUANTIZED != 0 {
        return Err(MeshBytesError::Quantized);
    }

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut tangents = Vec::new();
    let mut colors: Vec<Vec<[f32; 4]>> = Vec::new();
    let mut joints: Vec<Vec<[u16; 4]>> = Vec::new();
    let mut weights: Vec<Vec<[f32; 4]>> = Vec::new();
    let mut indices = Vec::new();
    let mut morph_targets = Vec::new();
    let mut saw_pos = false;
    let mut saw_nrm = false;
    let mut saw_tan = false;
    let mut saw_idx = false;
    let mut n_verts = None;
    let mut file = Reader {
        bytes,
        pos: 8,
    };

    while file.pos < bytes.len() {
        if file.remaining() < 8 {
            return Err(MeshBytesError::Truncated);
        }
        let id = file.take(4)?;
        let id = [id[0], id[1], id[2], id[3]];
        let size = file.u32()? as usize;
        let payload = file.take(size)?;
        let aligned = (file.pos + 3) & !3;
        if aligned <= bytes.len() {
            file.pos = aligned;
        } else {
            file.pos = bytes.len();
        }

        match id {
            ID_POS => {
                if saw_pos {
                    return Err(MeshBytesError::Duplicate("POS"));
                }
                saw_pos = true;
                let (count, body) = read_unslotted(payload, 12)?;
                note_count(&mut n_verts, count)?;
                positions = pack3(&read_f32s(body)?)?;
            }
            ID_NRM => {
                if saw_nrm {
                    return Err(MeshBytesError::Duplicate("NRM"));
                }
                saw_nrm = true;
                let (count, body) = read_unslotted(payload, 12)?;
                note_count(&mut n_verts, count)?;
                normals = pack3(&read_f32s(body)?)?;
            }
            ID_TAN => {
                if saw_tan {
                    return Err(MeshBytesError::Duplicate("TAN"));
                }
                saw_tan = true;
                let (count, body) = read_unslotted(payload, 16)?;
                note_count(&mut n_verts, count)?;
                tangents = pack4(&read_f32s(body)?)?;
            }
            ID_IDX => {
                if saw_idx {
                    return Err(MeshBytesError::Duplicate("IDX"));
                }
                saw_idx = true;
                let (count, body) = read_unslotted(payload, 4)?;
                if count % 3 != 0 {
                    return Err(MeshBytesError::BadIndices);
                }
                if body.len() != count as usize * 4 {
                    return Err(MeshBytesError::SizeMismatch);
                }
                indices = body
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
            }
            ID_UV => {
                let (slot, count, body) = read_slotted(payload, 8)?;
                note_count(&mut n_verts, count)?;
                let col = pack2(&read_f32s(body)?)?;
                put_column(&mut uvs, slot, col).map_err(|_| MeshBytesError::DuplicateSlot {
                    channel: "UV",
                    slot,
                })?;
            }
            ID_COL => {
                let (slot, count, body) = read_slotted(payload, 16)?;
                note_count(&mut n_verts, count)?;
                let col = pack4(&read_f32s(body)?)?;
                put_column(&mut colors, slot, col).map_err(|_| MeshBytesError::DuplicateSlot {
                    channel: "COL",
                    slot,
                })?;
            }
            ID_WGT => {
                let (slot, count, body) = read_slotted(payload, 16)?;
                note_count(&mut n_verts, count)?;
                let col = pack4(&read_f32s(body)?)?;
                put_column(&mut weights, slot, col).map_err(|_| MeshBytesError::DuplicateSlot {
                    channel: "WGT",
                    slot,
                })?;
            }
            ID_JNT => {
                let (slot, count, body) = read_slotted(payload, 8)?;
                note_count(&mut n_verts, count)?;
                if body.len() != count as usize * 8 {
                    return Err(MeshBytesError::SizeMismatch);
                }
                let col = body
                    .chunks_exact(8)
                    .map(|c| {
                        [
                            u16::from_le_bytes([c[0], c[1]]),
                            u16::from_le_bytes([c[2], c[3]]),
                            u16::from_le_bytes([c[4], c[5]]),
                            u16::from_le_bytes([c[6], c[7]]),
                        ]
                    })
                    .collect();
                put_column(&mut joints, slot, col).map_err(|_| MeshBytesError::DuplicateSlot {
                    channel: "JNT",
                    slot,
                })?;
            }
            ID_MRPH => {
                let target = read_morph(payload)?;
                note_count(&mut n_verts, target.position_deltas.len() as u32)?;
                morph_targets.push(target);
            }
            _ => {}
        }
    }

    if !morph_targets.is_empty() && positions.is_empty() {
        return Err(MeshBytesError::MorphWithoutPositions);
    }
    if let Some(n) = n_verts {
        for &i in &indices {
            if i >= n {
                return Err(MeshBytesError::BadIndices);
            }
        }
    } else if !indices.is_empty() {
        return Err(MeshBytesError::BadIndices);
    }

    let mut basis_positions = Vec::new();
    let mut basis_normals = Vec::new();
    if !morph_targets.is_empty() {
        basis_positions = positions.clone();
        basis_normals = if normals.len() == positions.len() {
            normals.clone()
        } else {
            vec![[0.0, 1.0, 0.0]; positions.len()]
        };
    }
    let morph_weights = vec![0.0; morph_targets.len()];

    if tangents.is_empty() && !positions.is_empty() {
        let uv0: &[[f32; 2]] = uvs.first().map(|c| c.as_slice()).unwrap_or(&[]);
        tangents = compute_tangents(&positions, &normals, uv0, &indices);
    }

    Ok(Mesh {
        positions,
        normals,
        uvs,
        tangents,
        colors,
        joints,
        weights,
        indices,
        version: 1,
        basis_positions,
        basis_normals,
        morph_targets,
        morph_weights,
    })
}

fn read_unslotted(payload: &[u8], stride: usize) -> Result<(u32, &[u8]), MeshBytesError> {
    if payload.len() < 4 {
        return Err(MeshBytesError::Truncated);
    }
    let count = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let expected = 4 + count as usize * stride;
    if payload.len() != expected {
        return Err(MeshBytesError::SizeMismatch);
    }
    Ok((count, &payload[4..]))
}

fn read_slotted(payload: &[u8], stride: usize) -> Result<(u32, u32, &[u8]), MeshBytesError> {
    if payload.len() < 8 {
        return Err(MeshBytesError::Truncated);
    }
    let slot = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let count = u32::from_le_bytes(payload[4..8].try_into().unwrap());
    if slot > MAX_SLOT {
        return Err(MeshBytesError::SlotTooLarge(slot));
    }
    let expected = 8 + count as usize * stride;
    if payload.len() != expected {
        return Err(MeshBytesError::SizeMismatch);
    }
    Ok((slot, count, &payload[8..]))
}

fn read_morph(payload: &[u8]) -> Result<MorphTarget, MeshBytesError> {
    let mut r = Reader {
        bytes: payload,
        pos: 0,
    };
    let name_len = r.u16()? as usize;
    let name_bytes = r.take(name_len)?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| MeshBytesError::BadUtf8)?
        .to_string();
    let name_section = 2 + name_len;
    let name_pad = (4 - (name_section % 4)) % 4;
    let _ = r.take(name_pad)?;
    let morph_flags = r.u32()?;
    let count = r.u32()? as usize;
    let has_nrm = morph_flags & 1 != 0;
    let expected = name_section
        + name_pad
        + 8
        + count * 12
        + if has_nrm { count * 12 } else { 0 };
    if payload.len() != expected {
        return Err(MeshBytesError::SizeMismatch);
    }
    let pos_bytes = r.take(count * 12)?;
    let position_deltas = pack3(&read_f32s(pos_bytes)?)?;
    let normal_deltas = if has_nrm {
        let nrm_bytes = r.take(count * 12)?;
        pack3(&read_f32s(nrm_bytes)?)?
    } else {
        Vec::new()
    };
    Ok(MorphTarget {
        name,
        position_deltas,
        normal_deltas,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(r: Result<Mesh, MeshBytesError>) -> MeshBytesError {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected MESH decode error"),
        }
    }

    fn tri() -> Mesh {
        Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0.0, 0.0, 1.0]; 3],
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn empty_roundtrip() {
        let m = Mesh::new(vec![], vec![], vec![], vec![]);
        let back = Mesh::from_bytes(&m.to_bytes()).unwrap();
        assert!(back.positions.is_empty());
        assert!(back.indices.is_empty());
    }

    #[test]
    fn spec_example_triangle() {
        let mut bytes = Vec::new();
        bytes.extend(b"MESH");
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(0u16.to_le_bytes());
        bytes.extend(b"POS ");
        bytes.extend(40u32.to_le_bytes());
        bytes.extend(3u32.to_le_bytes());
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bytes.extend(c.to_le_bytes());
            }
        }
        bytes.extend(b"IDX ");
        bytes.extend(16u32.to_le_bytes());
        bytes.extend(3u32.to_le_bytes());
        for i in [0u32, 1, 2] {
            bytes.extend(i.to_le_bytes());
        }
        let m = Mesh::from_bytes(&bytes).unwrap();
        assert_eq!(m.positions, vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(m.indices, vec![0, 1, 2]);
        assert_eq!(m.tangents.len(), 3);
    }

    #[test]
    fn roundtrip_attrs_and_sparse_uv() {
        let mut m = tri();
        m.uvs = vec![vec![], vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]];
        m.colors = vec![vec![[1.0, 0.0, 0.0, 1.0]; 3]];
        m.joints = vec![vec![[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]];
        m.weights = vec![vec![[1.0, 0.0, 0.0, 0.0]; 3]];
        let back = Mesh::from_bytes(&m.to_bytes()).unwrap();
        assert_eq!(back.positions, m.positions);
        assert_eq!(back.uvs[1], m.uvs[1]);
        assert!(back.uvs[0].is_empty());
        assert_eq!(back.colors[0], m.colors[0]);
        assert_eq!(back.joints[0], m.joints[0]);
        assert_eq!(back.weights[0], m.weights[0]);
        assert_eq!(back.indices, m.indices);
    }

    #[test]
    fn morph_uses_basis_and_skips_weights() {
        let mut m = tri();
        m.ensure_basis();
        m.add_shape_key("smile");
        m.add_shape_delta(0, 0, [0.1, 0.0, 0.0]);
        m.set_morph_weight(0, 1.0);
        assert_ne!(m.positions[0], m.basis_positions[0]);
        let back = Mesh::from_bytes(&m.to_bytes()).unwrap();
        assert_eq!(back.positions, m.basis_positions);
        assert_eq!(back.basis_positions, m.basis_positions);
        assert_eq!(back.morph_targets[0].name, "smile");
        assert_eq!(back.morph_targets[0].position_deltas[0], [0.1, 0.0, 0.0]);
        assert_eq!(back.morph_weights, vec![0.0]);
    }

    #[test]
    fn unknown_chunk_skipped() {
        let mut bytes = tri().to_bytes();
        bytes.extend(b"ZZZZ");
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        Mesh::from_bytes(&bytes).unwrap();
    }

    #[test]
    fn duplicate_pos_fails() {
        let mut bytes = tri().to_bytes();
        let pos = {
            let mut p = Vec::new();
            write_unslotted(&mut p, ID_POS, 12, 3, |b| {
                write_f32s(b, &[0.0; 9]);
            });
            p
        };
        bytes.extend(pos);
        assert_eq!(err(Mesh::from_bytes(&bytes)), MeshBytesError::Duplicate("POS"));
    }

    #[test]
    fn quantized_rejected() {
        let mut bytes = Mesh::new(vec![], vec![], vec![], vec![]).to_bytes();
        bytes[6] = 1;
        assert_eq!(err(Mesh::from_bytes(&bytes)), MeshBytesError::Quantized);
    }
}
