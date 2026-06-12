// Prevents an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

const SIDECAR_PORT: &str = "8756";

/// Pipe sidecar stdout/stderr into the dev console.
fn pipe_output(mut rx: tauri::async_runtime::Receiver<CommandEvent>) {
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    println!("[sidecar] {}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Stderr(line) => {
                    eprintln!("[sidecar] {}", String::from_utf8_lossy(&line));
                }
                _ => {}
            }
        }
    });
}

/// Launch the PyInstaller-built Python ML sidecar (packaged builds).
/// The binary lives in src-tauri/binaries/ named with the target triple
/// (e.g. panther-sidecar-aarch64-apple-darwin).
fn spawn_packaged_sidecar(app: &tauri::App) -> Result<(), String> {
    let cmd = app
        .shell()
        .sidecar("panther-sidecar")
        .map_err(|e| e.to_string())?;
    let (rx, _child) = cmd
        .args([SIDECAR_PORT])
        .spawn()
        .map_err(|e| e.to_string())?;
    pipe_output(rx);
    Ok(())
}

/// Dev fallback: run sidecar/server.py with the repo's venv python, so
/// `cargo tauri dev` lights up the real backend without packaging anything.
#[cfg(debug_assertions)]
fn spawn_dev_sidecar(app: &tauri::App) {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("src-tauri has no grandparent");
    let python = repo_root.join("venv/bin/python");
    let server = repo_root.join("sidecar/server.py");
    if !python.exists() || !server.exists() {
        eprintln!("[sidecar] dev venv/server not found (sample-data mode)");
        return;
    }
    match app
        .shell()
        .command(python.to_string_lossy().to_string())
        .args([server.to_string_lossy().to_string(), SIDECAR_PORT.into()])
        .spawn()
    {
        Ok((rx, _child)) => {
            println!("[sidecar] dev mode: running sidecar/server.py from venv");
            pipe_output(rx);
        }
        Err(e) => eprintln!("[sidecar] dev spawn failed (sample-data mode): {e}"),
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // The sidecar is optional: if it can't start, the window still opens
            // and the frontend falls back to sample data (and recovers via its
            // health poll once the engine appears).
            if let Err(e) = spawn_packaged_sidecar(app) {
                eprintln!("[sidecar] packaged binary not available: {e}");
                #[cfg(debug_assertions)]
                spawn_dev_sidecar(app);
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
