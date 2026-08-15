//! Locate and load `openxr_loader` without requiring it next to the exe.
//!
//! `openxr::Entry::load()` only searches the dynamic-loader path (exe dir,
//! System32, PATH). SteamVR / Quest Link / etc. ship the Khronos loader
//! inside their own install, which is almost never on PATH — so a bare
//! `load()` fails on a typical Windows VR box. We probe a few well-known
//! locations and `load_from` the first file that exists.

use super::session::XrError;
use openxr as xr;
use std::path::{Path, PathBuf};

/// Load the OpenXR loader. Tries the platform default name first, then
/// `OPENXR_LOADER` if set, then runtime-install guesses (SteamVR, …).
///
/// # Safety
///
/// Same as [`xr::Entry::load`]: the chosen library must be a spec-compliant
/// OpenXR loader.
pub unsafe fn load_entry() -> Result<xr::Entry, XrError> {
    let mut errors = Vec::new();

    match unsafe { xr::Entry::load() } {
        Ok(entry) => return Ok(entry),
        Err(e) => errors.push(format!("default name: {e}")),
    }

    for path in candidate_paths() {
        if !path.is_file() {
            continue;
        }
        match unsafe { xr::Entry::load_from(&path) } {
            Ok(entry) => {
                eprintln!("xr: loaded OpenXR loader from {}", path.display());
                return Ok(entry);
            }
            Err(e) => errors.push(format!("{}: {e}", path.display())),
        }
    }

    Err(XrError(format!(
        "failed to load OpenXR loader (set OPENXR_LOADER to the dll/so path?). tried:\n  {}",
        errors.join("\n  ")
    )))
}

fn candidate_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Some(p) = std::env::var_os("OPENXR_LOADER") {
        push_unique(&mut out, PathBuf::from(p));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            push_unique(&mut out, dir.join(loader_filename()));
        }
    }

    #[cfg(windows)]
    windows_candidates(&mut out);

    out
}

fn loader_filename() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "openxr_loader.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libopenxr_loader.dylib"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "libopenxr_loader.so"
    }
}

fn push_unique(out: &mut Vec<PathBuf>, path: PathBuf) {
    if !out.iter().any(|p| p == &path) {
        out.push(path);
    }
}

#[cfg(windows)]
fn windows_candidates(out: &mut Vec<PathBuf>) {
    if let Some(manifest) = active_runtime_manifest() {
        push_loader_near_runtime(out, &manifest);
    }

    let mut steam_roots = Vec::new();
    if let Some(steam) = steam_install_dir() {
        steam_roots.push(steam);
    }
    if let Some(pf) = std::env::var_os("ProgramFiles(x86)") {
        steam_roots.push(PathBuf::from(pf).join("Steam"));
    }
    if let Some(pf) = std::env::var_os("ProgramFiles") {
        steam_roots.push(PathBuf::from(pf).join("Steam"));
    }

    for steam in steam_roots {
        for lib in steam_library_roots(&steam) {
            push_unique(
                out,
                lib.join("steamapps/common/SteamVR/bin/win64").join(loader_filename()),
            );
        }
    }
}

/// `HKLM\SOFTWARE\Khronos\OpenXR\1\ActiveRuntime` — JSON manifest of the
/// currently selected OpenXR runtime (SteamVR, WMR, …).
#[cfg(windows)]
fn active_runtime_manifest() -> Option<PathBuf> {
    let hklm = winreg::RegKey::predef(winreg::enums::HKEY_LOCAL_MACHINE);
    let key = hklm.open_subkey(r"SOFTWARE\Khronos\OpenXR\1").ok()?;
    let value: String = key.get_value("ActiveRuntime").ok()?;
    let path = PathBuf::from(value);
    path.is_file().then_some(path)
}

/// Guess loader locations relative to a runtime JSON (the loader is *not*
/// the `runtime.library_path` inside that JSON — that's the runtime itself).
#[cfg(windows)]
fn push_loader_near_runtime(out: &mut Vec<PathBuf>, manifest: &Path) {
    let Some(dir) = manifest.parent() else { return };
    let name = loader_filename();
    push_unique(out, dir.join(name));
    push_unique(out, dir.join("bin/win64").join(name));
    push_unique(out, dir.join("bin").join(name));
}

#[cfg(windows)]
fn steam_install_dir() -> Option<PathBuf> {
    let hklm = winreg::RegKey::predef(winreg::enums::HKEY_LOCAL_MACHINE);
    for sub in [r"SOFTWARE\WOW6432Node\Valve\Steam", r"SOFTWARE\Valve\Steam"] {
        if let Ok(key) = hklm.open_subkey(sub) {
            if let Ok(path) = key.get_value::<String, _>("InstallPath") {
                let p = PathBuf::from(path);
                if p.is_dir() {
                    return Some(p);
                }
            }
        }
    }
    let hkcu = winreg::RegKey::predef(winreg::enums::HKEY_CURRENT_USER);
    if let Ok(key) = hkcu.open_subkey(r"SOFTWARE\Valve\Steam") {
        if let Ok(path) = key.get_value::<String, _>("SteamPath") {
            let p = PathBuf::from(path);
            if p.is_dir() {
                return Some(p);
            }
        }
    }
    None
}

/// Steam install dir plus extra library folders from `libraryfolders.vdf`
/// (SteamVR may live on another drive).
#[cfg(windows)]
fn steam_library_roots(steam: &Path) -> Vec<PathBuf> {
    let mut roots = vec![steam.to_path_buf()];
    let vdf = steam.join("steamapps/libraryfolders.vdf");
    let Ok(text) = std::fs::read_to_string(&vdf) else {
        return roots;
    };
    // Cheap parse: lines like `"path"		"D:\\SteamLibrary"`.
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("\"path\"") else { continue };
        let rest = rest.trim();
        let Some(q1) = rest.find('"') else { continue };
        let rest = &rest[q1 + 1..];
        let Some(q2) = rest.find('"') else { continue };
        let p = PathBuf::from(rest[..q2].replace("\\\\", "\\"));
        if p.is_dir() {
            push_unique(&mut roots, p);
        }
    }
    roots
}
