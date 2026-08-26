use crate::node::{Node, Transform};
use crate::skin::Skin;
use crate::store::Store;
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;

const MAGIC: &[u8; 4] = b"SKIN";
const VERSION: u16 = 1;
const FLAG_QUANTIZED: u16 = 1 << 0;

const ID_IBM: [u8; 4] = *b"IBM ";
const ID_NAME: [u8; 4] = *b"NAME";
const ID_PAR: [u8; 4] = *b"PAR ";
const ID_TRS: [u8; 4] = *b"TRS ";

/// Decoded `SKIN` blob (`docs/skin.md`). `Skin.joints` are not in the file.
#[derive(Clone)]
pub struct SkinFile {
    pub inverse_bind: Vec<Mat4>,
    /// Empty if the `NAME` chunk was omitted.
    pub names: Vec<String>,
    /// Empty if `PAR ` was omitted. `-1` = root.
    pub parents: Vec<i32>,
    /// Empty if `TRS ` was omitted.
    pub locals: Vec<Transform>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkinBytesError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u16),
    Quantized,
    Duplicate(&'static str),
    SizeMismatch,
    CountMismatch,
    HierarchyWithoutBind,
    BadParent,
    BadUtf8,
}

impl std::fmt::Display for SkinBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated SKIN blob"),
            Self::BadMagic => write!(f, "not a SKIN blob"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported SKIN version {v}"),
            Self::Quantized => write!(f, "QUANTIZED SKIN is not supported in version 1"),
            Self::Duplicate(id) => write!(f, "duplicate {id} chunk"),
            Self::SizeMismatch => write!(f, "chunk size does not match count"),
            Self::CountMismatch => write!(f, "skin chunk counts differ"),
            Self::HierarchyWithoutBind => write!(f, "NAME/PAR/TRS without IBM"),
            Self::BadParent => write!(f, "invalid joint parent"),
            Self::BadUtf8 => write!(f, "joint name is not UTF-8"),
        }
    }
}

impl std::error::Error for SkinBytesError {}

impl SkinFile {
    pub fn from_skin(skin: &Skin, nodes: &Store<Node>) -> Self {
        let n = skin.inverse_bind.len();
        let mut names = vec![String::new(); n];
        let mut parents = vec![-1i32; n];
        let mut locals = vec![Transform::default(); n];
        let mut index_of: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, h) in skin.joints.iter().enumerate() {
            index_of.insert(h.key(), i);
        }
        for (i, h) in skin.joints.iter().enumerate() {
            if i >= n {
                break;
            }
            let Some(node) = nodes.get(*h) else {
                continue;
            };
            names[i] = node.name.clone();
            locals[i] = node.local;
            if let Some(p) = node.parent {
                if let Some(&pi) = index_of.get(&p.key()) {
                    parents[i] = pi as i32;
                }
            }
        }
        Self {
            inverse_bind: skin.inverse_bind.clone(),
            names,
            parents,
            locals,
        }
    }

    pub fn into_skin(self, nodes: &mut Store<Node>) -> Skin {
        spawn_joints(self, nodes)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        to_bytes(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SkinBytesError> {
        from_bytes(bytes)
    }
}

fn spawn_joints(file: SkinFile, nodes: &mut Store<Node>) -> Skin {
    let n = file.inverse_bind.len();
    let mut joints = Vec::with_capacity(n);
    for i in 0..n {
        let name = file.names.get(i).cloned().unwrap_or_default();
        let local = file.locals.get(i).copied().unwrap_or_default();
        joints.push(nodes.insert(Node {
            name,
            parent: None,
            local,
            mesh: None,
            material: None,
            skin: None,
            visible: true,
        }));
    }
    for (i, &p) in file.parents.iter().enumerate() {
        if p >= 0 && i < joints.len() {
            let parent = joints[p as usize];
            if let Some(node) = nodes.get_mut(joints[i]) {
                node.parent = Some(parent);
            }
        }
    }
    Skin {
        joints,
        inverse_bind: file.inverse_bind,
    }
}

pub fn to_bytes(file: &SkinFile) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());

    let n = file.inverse_bind.len();
    if n == 0 {
        return out;
    }

    let mut payload = Vec::with_capacity(4 + n * 64);
    payload.extend_from_slice(&(n as u32).to_le_bytes());
    for m in &file.inverse_bind {
        write_f32s(&mut payload, &m.to_cols_array());
    }
    write_chunk(&mut out, ID_IBM, &payload);

    let names = if file.names.len() == n {
        file.names.clone()
    } else {
        vec![String::new(); n]
    };
    write_chunk(&mut out, ID_NAME, &encode_names(&names));

    let mut par = Vec::with_capacity(4 + n * 4);
    par.extend_from_slice(&(n as u32).to_le_bytes());
    if file.parents.len() == n {
        for p in &file.parents {
            par.extend_from_slice(&p.to_le_bytes());
        }
    } else {
        for _ in 0..n {
            par.extend_from_slice(&(-1i32).to_le_bytes());
        }
    }
    write_chunk(&mut out, ID_PAR, &par);

    let mut trs = Vec::with_capacity(4 + n * 40);
    trs.extend_from_slice(&(n as u32).to_le_bytes());
    for i in 0..n {
        let t = file.locals.get(i).copied().unwrap_or_default();
        write_f32s(&mut trs, &t.translation.to_array());
        write_f32s(&mut trs, &t.rotation.to_array());
        write_f32s(&mut trs, &t.scale.to_array());
    }
    write_chunk(&mut out, ID_TRS, &trs);
    out
}

pub fn from_bytes(bytes: &[u8]) -> Result<SkinFile, SkinBytesError> {
    if bytes.len() < 8 {
        return Err(SkinBytesError::Truncated);
    }
    if &bytes[0..4] != MAGIC {
        return Err(SkinBytesError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != VERSION {
        return Err(SkinBytesError::UnsupportedVersion(version));
    }
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    if flags & FLAG_QUANTIZED != 0 {
        return Err(SkinBytesError::Quantized);
    }

    let mut inverse_bind = Vec::new();
    let mut names = Vec::new();
    let mut parents = Vec::new();
    let mut locals = Vec::new();
    let mut saw_ibm = false;
    let mut saw_name = false;
    let mut saw_par = false;
    let mut saw_trs = false;
    let mut n_joints = None;
    let mut file = Reader {
        bytes,
        pos: 8,
    };

    while file.pos < bytes.len() {
        if file.remaining() < 8 {
            return Err(SkinBytesError::Truncated);
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
            ID_IBM => {
                if saw_ibm {
                    return Err(SkinBytesError::Duplicate("IBM"));
                }
                saw_ibm = true;
                let (count, body) = read_counted(payload, 64)?;
                note_count(&mut n_joints, count)?;
                inverse_bind = read_mats(body, count as usize)?;
            }
            ID_NAME => {
                if saw_name {
                    return Err(SkinBytesError::Duplicate("NAME"));
                }
                saw_name = true;
                names = read_names(payload, &mut n_joints)?;
            }
            ID_PAR => {
                if saw_par {
                    return Err(SkinBytesError::Duplicate("PAR"));
                }
                saw_par = true;
                let (count, body) = read_counted(payload, 4)?;
                note_count(&mut n_joints, count)?;
                parents = read_i32s(body, count as usize)?;
            }
            ID_TRS => {
                if saw_trs {
                    return Err(SkinBytesError::Duplicate("TRS"));
                }
                saw_trs = true;
                let (count, body) = read_counted(payload, 40)?;
                note_count(&mut n_joints, count)?;
                locals = read_trs(body, count as usize)?;
            }
            _ => {}
        }
    }

    if (saw_name || saw_par || saw_trs) && !saw_ibm {
        return Err(SkinBytesError::HierarchyWithoutBind);
    }
    if saw_par {
        check_parents(&parents)?;
    }

    Ok(SkinFile {
        inverse_bind,
        names,
        parents,
        locals,
    })
}

fn encode_names(names: &[String]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&(names.len() as u32).to_le_bytes());
    for name in names {
        let bytes = name.as_bytes();
        let name_len = u16::try_from(bytes.len()).unwrap_or(u16::MAX);
        let bytes = &bytes[..name_len as usize];
        payload.extend_from_slice(&name_len.to_le_bytes());
        payload.extend_from_slice(bytes);
        let section = 2 + bytes.len();
        let pad = (4 - (section % 4)) % 4;
        payload.resize(payload.len() + pad, 0);
    }
    payload
}

fn read_names(payload: &[u8], n_joints: &mut Option<u32>) -> Result<Vec<String>, SkinBytesError> {
    let mut r = Reader {
        bytes: payload,
        pos: 0,
    };
    let count = r.u32()?;
    note_count(n_joints, count)?;
    let mut names = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name_len = r.u16()? as usize;
        let raw = r.take(name_len)?;
        let name = std::str::from_utf8(raw)
            .map_err(|_| SkinBytesError::BadUtf8)?
            .to_string();
        let section = 2 + name_len;
        let pad = (4 - (section % 4)) % 4;
        let _ = r.take(pad)?;
        names.push(name);
    }
    if r.pos != payload.len() {
        return Err(SkinBytesError::SizeMismatch);
    }
    Ok(names)
}

fn read_counted(payload: &[u8], stride: usize) -> Result<(u32, &[u8]), SkinBytesError> {
    if payload.len() < 4 {
        return Err(SkinBytesError::Truncated);
    }
    let count = u32::from_le_bytes(payload[0..4].try_into().unwrap());
    let expected = 4 + count as usize * stride;
    if payload.len() != expected {
        return Err(SkinBytesError::SizeMismatch);
    }
    Ok((count, &payload[4..]))
}

fn read_mats(body: &[u8], count: usize) -> Result<Vec<Mat4>, SkinBytesError> {
    let mut out = Vec::with_capacity(count);
    for chunk in body.chunks_exact(64) {
        let mut a = [0f32; 16];
        for (i, c) in chunk.chunks_exact(4).enumerate() {
            a[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        out.push(Mat4::from_cols_array(&a));
    }
    Ok(out)
}

fn read_i32s(body: &[u8], count: usize) -> Result<Vec<i32>, SkinBytesError> {
    let mut out = Vec::with_capacity(count);
    for c in body.chunks_exact(4) {
        out.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(out)
}

fn read_trs(body: &[u8], count: usize) -> Result<Vec<Transform>, SkinBytesError> {
    let mut out = Vec::with_capacity(count);
    for chunk in body.chunks_exact(40) {
        let mut f = [0f32; 10];
        for (i, c) in chunk.chunks_exact(4).enumerate() {
            f[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        out.push(Transform {
            translation: Vec3::new(f[0], f[1], f[2]),
            rotation: Quat::from_xyzw(f[3], f[4], f[5], f[6]),
            scale: Vec3::new(f[7], f[8], f[9]),
        });
    }
    Ok(out)
}

fn note_count(n: &mut Option<u32>, count: u32) -> Result<(), SkinBytesError> {
    if count == 0 {
        return Ok(());
    }
    match *n {
        None => {
            *n = Some(count);
            Ok(())
        }
        Some(v) if v == count => Ok(()),
        Some(_) => Err(SkinBytesError::CountMismatch),
    }
}

fn check_parents(parents: &[i32]) -> Result<(), SkinBytesError> {
    let n = parents.len() as i32;
    for (i, &p) in parents.iter().enumerate() {
        if p == -1 {
            continue;
        }
        if p < 0 || p >= n || p == i as i32 {
            return Err(SkinBytesError::BadParent);
        }
    }
    let mut color = vec![0u8; parents.len()];
    for i in 0..parents.len() {
        if dfs_parent(parents, &mut color, i) {
            return Err(SkinBytesError::BadParent);
        }
    }
    Ok(())
}

fn dfs_parent(parents: &[i32], color: &mut [u8], i: usize) -> bool {
    match color[i] {
        1 => return true,
        2 => return false,
        _ => {}
    }
    color[i] = 1;
    let p = parents[i];
    if p >= 0 && dfs_parent(parents, color, p as usize) {
        return true;
    }
    color[i] = 2;
    false
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

    fn take(&mut self, n: usize) -> Result<&'a [u8], SkinBytesError> {
        if self.remaining() < n {
            return Err(SkinBytesError::Truncated);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16, SkinBytesError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, SkinBytesError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(r: Result<SkinFile, SkinBytesError>) -> SkinBytesError {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected SKIN decode error"),
        }
    }

    #[test]
    fn empty_roundtrip() {
        let nodes = Store::default();
        let bytes = Skin {
            joints: Vec::new(),
            inverse_bind: Vec::new(),
        }
        .to_bytes(&nodes);
        let mut out = Store::default();
        let back = Skin::from_bytes(&bytes, &mut out).unwrap();
        assert!(back.inverse_bind.is_empty());
        assert!(back.joints.is_empty());
        assert_eq!(&bytes[..4], b"SKIN");
    }

    #[test]
    fn spec_identity_ibm() {
        let mut bytes = Vec::new();
        bytes.extend(b"SKIN");
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(0u16.to_le_bytes());
        bytes.extend(b"IBM ");
        bytes.extend(68u32.to_le_bytes());
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(Mat4::IDENTITY.to_cols_array().iter().flat_map(|f| f.to_le_bytes()));
        let file = SkinFile::from_bytes(&bytes).unwrap();
        assert_eq!(file.inverse_bind.len(), 1);
        assert_eq!(file.inverse_bind[0], Mat4::IDENTITY);
    }

    #[test]
    fn hierarchy_roundtrip() {
        let file = SkinFile {
            inverse_bind: vec![Mat4::IDENTITY, Mat4::from_translation(Vec3::Y)],
            names: vec!["root".into(), "hip".into()],
            parents: vec![-1, 0],
            locals: vec![
                Transform::default(),
                Transform {
                    translation: Vec3::Y,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                },
            ],
        };
        let back = SkinFile::from_bytes(&file.to_bytes()).unwrap();
        assert_eq!(back.names, file.names);
        assert_eq!(back.parents, file.parents);
        assert_eq!(back.locals[1].translation, Vec3::Y);
        assert_eq!(back.inverse_bind.len(), 2);
    }

    #[test]
    fn node_roundtrip() {
        let mut nodes = Store::default();
        let root = nodes.insert(Node {
            name: "root".into(),
            parent: None,
            local: Transform::from_translation(Vec3::X),
            mesh: None,
            material: None,
            skin: None,
            visible: true,
        });
        let child = nodes.insert(Node {
            name: "child".into(),
            parent: Some(root),
            local: Transform::from_translation(Vec3::Y),
            mesh: None,
            material: None,
            skin: None,
            visible: true,
        });
        let skin = Skin {
            joints: vec![root, child],
            inverse_bind: vec![Mat4::IDENTITY, Mat4::IDENTITY],
        };
        let bytes = skin.to_bytes(&nodes);
        let mut out = Store::default();
        let back = Skin::from_bytes(&bytes, &mut out).unwrap();
        assert_eq!(back.joints.len(), 2);
        assert_eq!(out.get(back.joints[0]).unwrap().name, "root");
        assert_eq!(out.get(back.joints[1]).unwrap().name, "child");
        assert_eq!(
            out.get(back.joints[1]).unwrap().parent.map(|h| h.key()),
            Some(back.joints[0].key())
        );
        assert_eq!(out.get(back.joints[0]).unwrap().local.translation, Vec3::X);
    }

    #[test]
    fn name_without_ibm_fails() {
        let mut bytes = Vec::new();
        bytes.extend(b"SKIN");
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(0u16.to_le_bytes());
        let names = encode_names(&["a".into()]);
        bytes.extend(b"NAME");
        bytes.extend((names.len() as u32).to_le_bytes());
        bytes.extend(names);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        assert_eq!(
            err(SkinFile::from_bytes(&bytes)),
            SkinBytesError::HierarchyWithoutBind
        );
    }

    #[test]
    fn parent_cycle_fails() {
        let file = SkinFile {
            inverse_bind: vec![Mat4::IDENTITY, Mat4::IDENTITY],
            names: Vec::new(),
            parents: vec![1, 0],
            locals: Vec::new(),
        };
        assert_eq!(
            err(SkinFile::from_bytes(&file.to_bytes())),
            SkinBytesError::BadParent
        );
    }

    #[test]
    fn quantized_rejected() {
        let nodes = Store::default();
        let mut bytes = Skin {
            joints: Vec::new(),
            inverse_bind: Vec::new(),
        }
        .to_bytes(&nodes);
        bytes[6] = 1;
        assert_eq!(err(SkinFile::from_bytes(&bytes)), SkinBytesError::Quantized);
    }

    #[test]
    fn unknown_chunk_skipped() {
        let nodes = Store::default();
        let mut bytes = Skin {
            joints: Vec::new(),
            inverse_bind: vec![Mat4::IDENTITY],
        }
        .to_bytes(&nodes);
        bytes.extend(b"ZZZZ");
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        SkinFile::from_bytes(&bytes).unwrap();
    }
}
