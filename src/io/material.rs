use crate::material::{HairShading, Material, MaterialMaps, ShadingModel};
use crate::texgen::{TexGraphBytesError, TexGraphFile};
use crate::texture::TextureStore;

const MAGIC: &[u8; 4] = b"MAT ";
const VERSION: u16 = 1;
const FLAG_QUANTIZED: u16 = 1 << 0;

const ID_PBR: [u8; 4] = *b"PBR ";
const ID_SSS: [u8; 4] = *b"SSS ";
const ID_CUT: [u8; 4] = *b"CUT ";
const ID_HAIR: [u8; 4] = *b"HAIR";
const ID_ALB: [u8; 4] = *b"ALB ";
const ID_NRM: [u8; 4] = *b"NRM ";
const ID_MR: [u8; 4] = *b"MR  ";
const ID_UALB: [u8; 4] = *b"UALB";
const ID_UNRM: [u8; 4] = *b"UNRM";
const ID_UMR: [u8; 4] = *b"UMR ";
const ID_TEXG: [u8; 4] = *b"TEXG";

/// Recipe from a `MAT ` blob. Texture slots are string ids, not handles.
#[derive(Clone, Debug)]
pub struct MaterialFile {
    pub albedo: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub sss_strength: f32,
    pub sss_color: [f32; 3],
    pub sss_curvature: f32,
    pub alpha_cutoff: f32,
    pub shading_model: ShadingModel,
    pub maps: MaterialFileMaps,
    /// Procedural maps. If set, bake this instead of resolving `maps` ids.
    pub graph: Option<TexGraphFile>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaterialFileMaps {
    Single {
        albedo: Option<String>,
        normal: Option<String>,
        metallic_roughness: Option<String>,
    },
    Udim {
        albedo: Vec<(u32, String)>,
        normal: Vec<(u32, String)>,
        metallic_roughness: Vec<(u32, String)>,
    },
}

impl Default for MaterialFile {
    fn default() -> Self {
        let m = Material::default();
        Self {
            albedo: m.albedo,
            metallic: m.metallic,
            roughness: m.roughness,
            sss_strength: m.sss_strength,
            sss_color: m.sss_color,
            sss_curvature: m.sss_curvature,
            alpha_cutoff: m.alpha_cutoff,
            shading_model: m.shading_model,
            maps: MaterialFileMaps::Single {
                albedo: None,
                normal: None,
                metallic_roughness: None,
            },
            graph: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterialBytesError {
    Truncated,
    BadMagic,
    UnsupportedVersion(u16),
    Quantized,
    Duplicate(&'static str),
    MixedMaps,
    SizeMismatch,
    BadUtf8,
    NotProcedural,
    Graph(TexGraphBytesError),
}

impl std::fmt::Display for MaterialBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated MAT blob"),
            Self::BadMagic => write!(f, "not a MAT blob"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported MAT version {v}"),
            Self::Quantized => write!(f, "QUANTIZED MAT is not supported in version 1"),
            Self::Duplicate(id) => write!(f, "duplicate {id} chunk"),
            Self::MixedMaps => write!(f, "single and UDIM map chunks in one MAT"),
            Self::SizeMismatch => write!(f, "chunk size does not match payload"),
            Self::BadUtf8 => write!(f, "texture id is not UTF-8"),
            Self::NotProcedural => write!(f, "MAT has no TEXG graph"),
            Self::Graph(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MaterialBytesError {}

impl MaterialFile {
    pub fn from_material(mat: &Material, textures: &TextureStore) -> Self {
        let maps = match &mat.maps {
            MaterialMaps::Single {
                albedo,
                normal,
                metallic_roughness,
            } => MaterialFileMaps::Single {
                albedo: id_of(textures, *albedo),
                normal: id_of(textures, *normal),
                metallic_roughness: id_of(textures, *metallic_roughness),
            },
            MaterialMaps::Udim {
                albedo,
                normal,
                metallic_roughness,
            } => MaterialFileMaps::Udim {
                albedo: udim_ids(textures, albedo),
                normal: udim_ids(textures, normal),
                metallic_roughness: udim_ids(textures, metallic_roughness),
            },
        };
        Self {
            albedo: mat.albedo,
            metallic: mat.metallic,
            roughness: mat.roughness,
            sss_strength: mat.sss_strength,
            sss_color: mat.sss_color,
            sss_curvature: mat.sss_curvature,
            alpha_cutoff: mat.alpha_cutoff,
            shading_model: mat.shading_model,
            maps,
            graph: None,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        to_bytes(self)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MaterialBytesError> {
        from_bytes(bytes)
    }

    pub fn is_procedural(&self) -> bool {
        self.graph.is_some()
    }

    /// Texture catalog keys referenced by this recipe (order: albedo, normal, MR).
    pub fn texture_ids(&self) -> Vec<&str> {
        let mut out = Vec::new();
        match &self.maps {
            MaterialFileMaps::Single {
                albedo,
                normal,
                metallic_roughness,
            } => {
                push_id(&mut out, albedo.as_deref());
                push_id(&mut out, normal.as_deref());
                push_id(&mut out, metallic_roughness.as_deref());
            }
            MaterialFileMaps::Udim {
                albedo,
                normal,
                metallic_roughness,
            } => {
                for (_, id) in albedo {
                    push_id(&mut out, Some(id));
                }
                for (_, id) in normal {
                    push_id(&mut out, Some(id));
                }
                for (_, id) in metallic_roughness {
                    push_id(&mut out, Some(id));
                }
            }
        }
        out
    }
}

fn push_id<'a>(out: &mut Vec<&'a str>, id: Option<&'a str>) {
    let Some(id) = id else {
        return;
    };
    if id.is_empty() || out.contains(&id) {
        return;
    }
    out.push(id);
}

fn id_of(textures: &TextureStore, h: Option<crate::Handle<crate::Texture>>) -> Option<String> {
    let h = h?;
    let tex = textures.get(h)?;
    if tex.id.is_empty() {
        None
    } else {
        Some(tex.id.clone())
    }
}

fn udim_ids(
    textures: &TextureStore,
    tiles: &[(u32, crate::Handle<crate::Texture>)],
) -> Vec<(u32, String)> {
    tiles
        .iter()
        .filter_map(|(udim, h)| {
            let tex = textures.get(*h)?;
            if tex.id.is_empty() {
                None
            } else {
                Some((*udim, tex.id.clone()))
            }
        })
        .collect()
}

pub fn to_bytes(file: &MaterialFile) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());

    let mut pbr = Vec::new();
    write_f32s(&mut pbr, &file.albedo);
    write_f32s(&mut pbr, &[file.metallic, file.roughness]);
    write_chunk(&mut out, ID_PBR, &pbr);

    if file.sss_strength > 0.0 {
        let mut sss = Vec::new();
        write_f32s(&mut sss, &[file.sss_strength]);
        write_f32s(&mut sss, &file.sss_color);
        write_f32s(&mut sss, &[file.sss_curvature]);
        write_chunk(&mut out, ID_SSS, &sss);
    }
    if file.alpha_cutoff > 0.0 {
        let mut cut = Vec::new();
        write_f32s(&mut cut, &[file.alpha_cutoff]);
        write_chunk(&mut out, ID_CUT, &cut);
    }
    if let ShadingModel::Hair(h) = file.shading_model {
        write_chunk(&mut out, ID_HAIR, &encode_hair(&h));
    }

    match &file.maps {
        MaterialFileMaps::Single {
            albedo,
            normal,
            metallic_roughness,
        } => {
            write_id_chunk(&mut out, ID_ALB, albedo.as_deref());
            write_id_chunk(&mut out, ID_NRM, normal.as_deref());
            write_id_chunk(&mut out, ID_MR, metallic_roughness.as_deref());
        }
        MaterialFileMaps::Udim {
            albedo,
            normal,
            metallic_roughness,
        } => {
            write_udim_chunk(&mut out, ID_UALB, albedo);
            write_udim_chunk(&mut out, ID_UNRM, normal);
            write_udim_chunk(&mut out, ID_UMR, metallic_roughness);
        }
    }
    if let Some(graph) = &file.graph {
        write_chunk(&mut out, ID_TEXG, &graph.to_bytes());
    }
    out
}

pub fn from_bytes(bytes: &[u8]) -> Result<MaterialFile, MaterialBytesError> {
    if bytes.len() < 8 {
        return Err(MaterialBytesError::Truncated);
    }
    if &bytes[0..4] != MAGIC {
        return Err(MaterialBytesError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != VERSION {
        return Err(MaterialBytesError::UnsupportedVersion(version));
    }
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    if flags & FLAG_QUANTIZED != 0 {
        return Err(MaterialBytesError::Quantized);
    }

    let mut file = MaterialFile::default();
    let mut saw_pbr = false;
    let mut saw_sss = false;
    let mut saw_cut = false;
    let mut saw_hair = false;
    let mut saw_alb = false;
    let mut saw_nrm = false;
    let mut saw_mr = false;
    let mut saw_ualb = false;
    let mut saw_unrm = false;
    let mut saw_umr = false;
    let mut saw_texg = false;
    let mut single = MaterialFileMaps::Single {
        albedo: None,
        normal: None,
        metallic_roughness: None,
    };
    let mut udim = MaterialFileMaps::Udim {
        albedo: Vec::new(),
        normal: Vec::new(),
        metallic_roughness: Vec::new(),
    };
    let mut used_single = false;
    let mut used_udim = false;

    let mut r = Reader {
        bytes,
        pos: 8,
    };
    while r.remaining() > 0 {
        if r.remaining() < 8 {
            return Err(MaterialBytesError::Truncated);
        }
        let id = r.take(4)?;
        let id = [id[0], id[1], id[2], id[3]];
        let size = r.u32()? as usize;
        let payload = r.take(size)?;
        let aligned = (r.pos + 3) & !3;
        if aligned <= bytes.len() {
            r.pos = aligned;
        } else {
            r.pos = bytes.len();
        }

        match id {
            ID_PBR => {
                if saw_pbr {
                    return Err(MaterialBytesError::Duplicate("PBR"));
                }
                saw_pbr = true;
                let f = read_f32s(payload, 6)?;
                file.albedo = [f[0], f[1], f[2], f[3]];
                file.metallic = f[4];
                file.roughness = f[5];
            }
            ID_SSS => {
                if saw_sss {
                    return Err(MaterialBytesError::Duplicate("SSS"));
                }
                saw_sss = true;
                let f = read_f32s(payload, 5)?;
                file.sss_strength = f[0];
                file.sss_color = [f[1], f[2], f[3]];
                file.sss_curvature = f[4];
            }
            ID_CUT => {
                if saw_cut {
                    return Err(MaterialBytesError::Duplicate("CUT"));
                }
                saw_cut = true;
                let f = read_f32s(payload, 1)?;
                file.alpha_cutoff = f[0];
            }
            ID_HAIR => {
                if saw_hair {
                    return Err(MaterialBytesError::Duplicate("HAIR"));
                }
                saw_hair = true;
                file.shading_model = ShadingModel::Hair(read_hair(payload)?);
            }
            ID_ALB => {
                dup(&mut saw_alb, "ALB")?;
                used_single = true;
                if let MaterialFileMaps::Single { albedo, .. } = &mut single {
                    *albedo = Some(read_id(payload)?);
                }
            }
            ID_NRM => {
                dup(&mut saw_nrm, "NRM")?;
                used_single = true;
                if let MaterialFileMaps::Single { normal, .. } = &mut single {
                    *normal = Some(read_id(payload)?);
                }
            }
            ID_MR => {
                dup(&mut saw_mr, "MR")?;
                used_single = true;
                if let MaterialFileMaps::Single {
                    metallic_roughness, ..
                } = &mut single
                {
                    *metallic_roughness = Some(read_id(payload)?);
                }
            }
            ID_UALB => {
                dup(&mut saw_ualb, "UALB")?;
                used_udim = true;
                if let MaterialFileMaps::Udim { albedo, .. } = &mut udim {
                    *albedo = read_udim(payload)?;
                }
            }
            ID_UNRM => {
                dup(&mut saw_unrm, "UNRM")?;
                used_udim = true;
                if let MaterialFileMaps::Udim { normal, .. } = &mut udim {
                    *normal = read_udim(payload)?;
                }
            }
            ID_UMR => {
                dup(&mut saw_umr, "UMR")?;
                used_udim = true;
                if let MaterialFileMaps::Udim {
                    metallic_roughness, ..
                } = &mut udim
                {
                    *metallic_roughness = read_udim(payload)?;
                }
            }
            ID_TEXG => {
                dup(&mut saw_texg, "TEXG")?;
                file.graph = Some(
                    TexGraphFile::from_bytes(payload).map_err(MaterialBytesError::Graph)?,
                );
            }
            _ => {}
        }
    }

    if used_single && used_udim {
        return Err(MaterialBytesError::MixedMaps);
    }
    file.maps = if used_udim { udim } else { single };
    Ok(file)
}

fn dup(saw: &mut bool, name: &'static str) -> Result<(), MaterialBytesError> {
    if *saw {
        return Err(MaterialBytesError::Duplicate(name));
    }
    *saw = true;
    Ok(())
}

fn encode_hair(h: &HairShading) -> Vec<u8> {
    let mut p = Vec::new();
    write_f32s(
        &mut p,
        &[
            h.primary_shift,
            h.secondary_shift,
            h.primary_exponent,
            h.secondary_exponent,
            h.secondary_tint[0],
            h.secondary_tint[1],
            h.secondary_tint[2],
            h.secondary_strength,
            h.tip_fade,
            h.cutout_fringe,
        ],
    );
    p.extend_from_slice(&(u32::from(h.soft_blend)).to_le_bytes());
    p
}

fn read_hair(payload: &[u8]) -> Result<HairShading, MaterialBytesError> {
    if payload.len() != 44 {
        return Err(MaterialBytesError::SizeMismatch);
    }
    let f = read_f32s(&payload[..40], 10)?;
    let flags = u32::from_le_bytes(payload[40..44].try_into().unwrap());
    Ok(HairShading {
        primary_shift: f[0],
        secondary_shift: f[1],
        primary_exponent: f[2],
        secondary_exponent: f[3],
        secondary_tint: [f[4], f[5], f[6]],
        secondary_strength: f[7],
        tip_fade: f[8],
        cutout_fringe: f[9],
        soft_blend: flags & 1 != 0,
    })
}

fn write_id_chunk(out: &mut Vec<u8>, id: [u8; 4], tex_id: Option<&str>) {
    let Some(tex_id) = tex_id.filter(|s| !s.is_empty()) else {
        return;
    };
    write_chunk(out, id, &encode_id(tex_id));
}

fn encode_id(id: &str) -> Vec<u8> {
    let bytes = id.as_bytes();
    let n = u16::try_from(bytes.len()).unwrap_or(u16::MAX);
    let bytes = &bytes[..n as usize];
    let mut p = Vec::with_capacity(2 + bytes.len());
    p.extend_from_slice(&n.to_le_bytes());
    p.extend_from_slice(bytes);
    p
}

fn read_id(payload: &[u8]) -> Result<String, MaterialBytesError> {
    if payload.len() < 2 {
        return Err(MaterialBytesError::Truncated);
    }
    let n = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    if payload.len() < 2 + n {
        return Err(MaterialBytesError::SizeMismatch);
    }
    std::str::from_utf8(&payload[2..2 + n])
        .map(|s| s.to_string())
        .map_err(|_| MaterialBytesError::BadUtf8)
}

fn write_udim_chunk(out: &mut Vec<u8>, id: [u8; 4], tiles: &[(u32, String)]) {
    if tiles.is_empty() {
        return;
    }
    let mut p = Vec::new();
    p.extend_from_slice(&(tiles.len() as u32).to_le_bytes());
    for (udim, tex_id) in tiles {
        p.extend_from_slice(&udim.to_le_bytes());
        p.extend_from_slice(&encode_id(tex_id));
        while p.len() % 4 != 0 {
            p.push(0);
        }
    }
    write_chunk(out, id, &p);
}

fn read_udim(payload: &[u8]) -> Result<Vec<(u32, String)>, MaterialBytesError> {
    let mut r = Reader {
        bytes: payload,
        pos: 0,
    };
    let count = r.u32()? as usize;
    let mut tiles = Vec::with_capacity(count);
    for _ in 0..count {
        let udim = r.u32()?;
        let n = r.u16()? as usize;
        let raw = r.take(n)?;
        let id = std::str::from_utf8(raw)
            .map_err(|_| MaterialBytesError::BadUtf8)?
            .to_string();
        let section = r.pos;
        let pad = (4 - (section % 4)) % 4;
        let _ = r.take(pad)?;
        tiles.push((udim, id));
    }
    if r.pos != payload.len() {
        return Err(MaterialBytesError::SizeMismatch);
    }
    Ok(tiles)
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

fn read_f32s(payload: &[u8], n: usize) -> Result<Vec<f32>, MaterialBytesError> {
    if payload.len() < n * 4 {
        return Err(MaterialBytesError::SizeMismatch);
    }
    Ok(payload[..n * 4]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], MaterialBytesError> {
        if self.remaining() < n {
            return Err(MaterialBytesError::Truncated);
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16, MaterialBytesError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, MaterialBytesError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::Texture;

    fn err(r: Result<MaterialFile, MaterialBytesError>) -> MaterialBytesError {
        match r {
            Err(e) => e,
            Ok(_) => panic!("expected MAT decode error"),
        }
    }

    #[test]
    fn default_roundtrip() {
        let textures = TextureStore::default();
        let bytes = Material::default().to_bytes(&textures);
        let file = MaterialFile::from_bytes(&bytes).unwrap();
        assert_eq!(&bytes[..4], b"MAT ");
        assert_eq!(file.albedo, [1.0, 1.0, 1.0, 1.0]);
        assert!(file.texture_ids().is_empty());
        match file.maps {
            MaterialFileMaps::Single {
                albedo: None,
                normal: None,
                metallic_roughness: None,
            } => {}
            _ => panic!("expected empty single maps"),
        }
    }

    #[test]
    fn maps_and_hair_roundtrip() {
        let mut textures = TextureStore::default();
        let mut alb = Texture::solid(255, 0, 0, 255);
        alb.id = "char/albedo".into();
        let mut nrm = Texture::solid_linear(128, 128, 255, 255);
        nrm.id = "char/nrm".into();
        let ha = textures.insert(alb);
        let hn = textures.insert(nrm);

        let mut mat = Material::new([0.2, 0.3, 0.4, 1.0], 0.1, 0.6);
        mat.maps = MaterialMaps::Single {
            albedo: Some(ha),
            normal: Some(hn),
            metallic_roughness: None,
        };
        mat.alpha_cutoff = 0.4;
        mat.shading_model = ShadingModel::Hair(HairShading {
            soft_blend: true,
            ..HairShading::default()
        });

        let file = MaterialFile::from_bytes(&mat.to_bytes(&textures)).unwrap();
        assert_eq!(file.albedo, [0.2, 0.3, 0.4, 1.0]);
        assert_eq!(file.alpha_cutoff, 0.4);
        assert_eq!(file.texture_ids(), vec!["char/albedo", "char/nrm"]);
        match &file.maps {
            MaterialFileMaps::Single {
                albedo,
                normal,
                metallic_roughness,
            } => {
                assert_eq!(albedo.as_deref(), Some("char/albedo"));
                assert_eq!(normal.as_deref(), Some("char/nrm"));
                assert!(metallic_roughness.is_none());
            }
            _ => panic!("expected single maps"),
        }
        match file.shading_model {
            ShadingModel::Hair(h) => assert!(h.soft_blend),
            _ => panic!("expected hair"),
        }
    }

    #[test]
    fn unknown_chunk_skipped() {
        let mut bytes = Material::default().to_bytes(&TextureStore::default());
        bytes.extend(b"ZZZZ");
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        MaterialFile::from_bytes(&bytes).unwrap();
    }

    #[test]
    fn quantized_rejected() {
        let mut bytes = Vec::new();
        bytes.extend(b"MAT ");
        bytes.extend(1u16.to_le_bytes());
        bytes.extend(1u16.to_le_bytes());
        assert_eq!(err(MaterialFile::from_bytes(&bytes)), MaterialBytesError::Quantized);
    }

    #[test]
    fn texg_roundtrip() {
        use crate::texgen::{NodeKind, TexGraph, TexGraphFile};

        let mut g = TexGraph::new();
        let out = g.output_id.clone();
        let n = g.add(NodeKind::Noise);
        g.connect(&n, "out", &out, "roughness");
        let mut file = MaterialFile::default();
        file.albedo = [0.2, 0.3, 0.4, 1.0];
        file.graph = Some(TexGraphFile {
            resolution: 256,
            graph: g,
        });
        let back = MaterialFile::from_bytes(&file.to_bytes()).unwrap();
        assert!(back.is_procedural());
        assert_eq!(back.albedo, [0.2, 0.3, 0.4, 1.0]);
        let g = back.graph.unwrap();
        assert_eq!(g.resolution, 256);
        assert_eq!(g.graph.links.len(), 1);
    }
}
