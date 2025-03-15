#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use app::BrowseApp;
use clap::Parser;
mod app;
mod chat_gpt2;
mod chat_llama2;
mod chat_template;
mod llm;
mod prompt;
mod sampler;

#[cfg(not(target_arch = "wasm32"))]
mod cli;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
#[async_std::main]
async fn main() -> eframe::Result<()> {
    let cli = cli::Cli::parse();
    if cli.headless {
        cli::run_headless(&cli).await.unwrap();
        Ok(())
    } else {
        tracing_subscriber::fmt::init();
        let native_options = eframe::NativeOptions::default();
        let app = BrowseApp::new().await;
        eframe::run_native("browse", native_options, Box::new(|cc| Ok(Box::new(app))))
    }
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::{wasm_bindgen::JsCast, web_sys};

    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let app = BrowseApp::new().await;
        eframe::WebRunner::new()
            .start(canvas, web_options, Box::new(|cc| Ok(Box::new(app))))
            .await
            .expect("failed to start eframe");
    });
}
