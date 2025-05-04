use crate::chat_template::ChatTemplate;
use crate::llm::ChatLlm;
use crate::prompt::Prompt;
use crate::sampler::SamplerParams;
use components::{Chat, Home};
use dioxus::prelude::*;
use std::sync::Arc;
use wgcore::gpu::GpuInstance;

mod chat_gpt2;
mod chat_llama2;
mod chat_template;
mod components;
// mod cli;
mod llm;
mod prompt;
mod sampler;

pub type GpuInstanceCtx = Arc<GpuInstance>;
pub type LoadedModelSignal = Signal<Option<LoadedModel>>;

#[derive(Clone, Debug, Default)]
enum PromptResponse {
    #[default]
    Empty,
    Thinking,
    Responding(String),
}

#[derive(Default)]
struct PromptState {
    prompt: Prompt,
    response: PromptResponse,
}

#[derive(Clone)]
pub struct GgufMetadata {
    metadata: Vec<String>,
    tensors: Vec<String>,
}

#[derive(Clone)]
pub struct LoadedModel {
    pub llm: Arc<ChatLlm>,
    pub sampler: SamplerParams,
    pub template: ChatTemplate,
    pub metadata: GgufMetadata,
}

const FAVICON: Asset = asset!("/assets/wgml logo.png");
const MAIN_CSS: Asset = asset!("/assets/styling/main.css");

fn main() {
    #[cfg(feature = "desktop")]
    {
        use dioxus::desktop::{tao, LogicalSize};
        let window =
            tao::window::WindowBuilder::default().with_inner_size(LogicalSize::new(1300.0, 900.0));
        dioxus::LaunchBuilder::new()
            .with_cfg(dioxus::desktop::Config::new().with_window(window))
            .launch(App);
    }

    #[cfg(not(feature = "desktop"))]
    {
        dioxus::launch(App);
    }
}

#[component]
fn App() -> Element {
    let gpu =
        use_resource(|| async move { Ok::<_, anyhow::Error>(Arc::new(GpuInstance::new().await?)) });

    match &*gpu.read_unchecked() {
        Some(Ok(gpu)) => {
            use_context_provider(|| gpu.clone());
            use_context_provider(|| LoadedModelSignal::new(None));
            use_context_provider(|| Signal::new(PromptState::default()));

            rsx! {
                // Global app resources
                document::Link { rel: "icon", href: FAVICON }
                document::Link { rel: "stylesheet", href: MAIN_CSS }
                document::Title { "WGML chat" }

                if use_context::<LoadedModelSignal>().read().is_none() {
                    Home {}
                } else {
                    Chat {}
                }

            }
        }
        _ => rsx! {},
    }
}
