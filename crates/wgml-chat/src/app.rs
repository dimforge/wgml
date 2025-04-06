use crate::chat_llama2::ChatLlama2;
use crate::chat_template::ChatTemplate;
use crate::llm::ChatLlm;
use crate::prompt::{ChatEvent, Prompt, PromptEntry};
use crate::sampler::SamplerParams;
use egui::{Context, DragValue, RichText, TextWrapMode, Ui};
use log::info;
use std::future::Future;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use wgcore::gpu::GpuInstance;
use wgml::gguf::Gguf;
use wgml::models::llama2::Llama2;

enum Response {
    Empty,
    Thinking,
    Responding(String),
}

pub enum GgufLoadingProgress {
    ReadingFile,
    ReadingTensors,
    PopulatingGpuResources,
}

#[derive(Default)]
pub struct ParsedPrompt {
    templated_prompt: String,
    tokens: Vec<(usize, String)>,
}

pub struct BrowseApp {
    gpu: Arc<GpuInstance>,
    llm_channel: (Sender<Gguf>, Receiver<Gguf>),
    llm: Option<Arc<ChatLlm>>,
    gguf: Option<Arc<Gguf>>,
    gguf_progress: Option<GgufLoadingProgress>,
    gguf_progress_channel: Option<(
        async_channel::Sender<GgufLoadingProgress>,
        async_channel::Receiver<GgufLoadingProgress>,
    )>,
    response_rcv: Option<async_channel::Receiver<ChatEvent>>,
    next_prompt: String,
    next_pos: usize,
    tok_per_second: f64,
    prompt: Prompt,
    parsed_prompt: ParsedPrompt,
    sampler: SamplerParams,
    response: Response,
    chat_template: ChatTemplate,
    error: String,
}

impl BrowseApp {
    /// Called once before the first frame.
    pub async fn new() -> Self {
        Self {
            gpu: Arc::new(GpuInstance::new().await.unwrap()),
            llm_channel: channel(),
            llm: None,
            gguf: None,
            gguf_progress: None,
            gguf_progress_channel: None,
            next_prompt: "Give me a chocolate cake recipe.".to_string(),
            parsed_prompt: ParsedPrompt::default(),
            next_pos: 0,
            tok_per_second: 0.0,
            prompt: Prompt::default(),
            sampler: SamplerParams::default(),
            chat_template: ChatTemplate::default(),
            response: Response::Empty,
            response_rcv: None,
            error: "".to_string(),
        }
    }

    fn gguf_ui(&self, ui: &mut Ui) {
        if let Some(llm) = &self.llm {
            let txt = format!("Detected {}", llm.model_name());
            ui.label(RichText::new(txt).color(egui::Color32::from_rgb(0, 155, 0)));
        }

        if let Some(gguf) = &self.gguf {
            ui.label("GGUF file loaded.");
            ui.style_mut().wrap_mode = Some(TextWrapMode::Extend);

            ui.collapsing("Metadata", |ui| {
                egui::ScrollArea::vertical()
                    .hscroll(true)
                    .max_height(250.0)
                    .show(ui, |ui| {
                        let mut strings: Vec<_> = gguf.metadata_debug_strings().collect();
                        strings.sort();
                        for str in strings {
                            ui.label(str);
                        }
                    });
            });

            ui.collapsing("Tensors", |ui| {
                egui::ScrollArea::vertical()
                    .hscroll(true)
                    .max_height(250.0)
                    .show(ui, |ui| {
                        for str in gguf.tensors_debug_strings() {
                            ui.label(str);
                        }
                    });
            });
        } else {
            if self.gguf_progress.is_some() {
                ui.label("Loading GGUF file.");
            } else {
                ui.label("No GGUF file loaded.");
            }

            // TODO: this doesn’t really work because the gguf loading acts synchronously right now.
            //
            // match self.gguf_progress {
            //     Some(GgufLoadingProgress::ReadingFile) => {
            //         ui.label("[1/3] Reading GGUF bytes…");
            //     }
            //     Some(GgufLoadingProgress::ReadingTensors) => {
            //         ui.label("[2/3] Reading tensors…");
            //     }
            //     Some(GgufLoadingProgress::PopulatingGpuResources) => {
            //         ui.label("[3/3] Populating VRam…");
            //     }
            //     None => {
            //         ui.label("No GGUF file loaded.");
            //     }
            // }
        }
    }

    fn user_ui(ui: &mut Ui, txt: &str) {
        ui.label(RichText::new("User:").color(egui::Color32::GRAY).italics());
        ui.label(RichText::new(txt).color(egui::Color32::DARK_BLUE));
    }

    fn assistant_ui(ui: &mut Ui, txt: &str) {
        ui.label(
            RichText::new("Assistant:")
                .color(egui::Color32::GRAY)
                .italics(),
        );
        ui.label(RichText::new(txt).color(egui::Color32::DARK_GREEN));
    }

    fn response_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                for entry in self.prompt.entries() {
                    match entry {
                        PromptEntry::User(txt) => {
                            Self::user_ui(ui, txt);
                        }
                        PromptEntry::Assistant(txt) => {
                            Self::assistant_ui(ui, txt);
                        }
                        PromptEntry::System(txt) => {
                            ui.label(
                                RichText::new(txt)
                                    .color(egui::Color32::LIGHT_GRAY)
                                    .italics(),
                            );
                        }
                    }
                }

                match &mut self.response {
                    Response::Responding(resp) => {
                        Self::assistant_ui(ui, resp);
                    }
                    Response::Thinking => {
                        Self::assistant_ui(ui, &"<Thinking…>");
                    }
                    Response::Empty => {}
                }
            });
        });
    }

    fn perf_ui(&mut self, ui: &mut Ui) {
        ui.collapsing("Performances", |ui| {
            ui.label(format!(
                "{:.2} tok/s − generated {} tokens",
                self.tok_per_second, self.next_pos
            ));
        });
    }

    fn sampler_ui(&mut self, ui: &mut Ui) {
        ui.collapsing("Sampler", |ui| {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.sampler.top_k_enabled, "Top-K");
                ui.add_enabled_ui(self.sampler.top_k_enabled, |ui| {
                    ui.label("K:");
                    ui.add(DragValue::new(&mut self.sampler.top_k.k));
                    ui.label("Min-keep:");
                    ui.add(DragValue::new(&mut self.sampler.top_k.min_keep));
                });
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.sampler.typical_enabled, "Typical");
                ui.add_enabled_ui(self.sampler.typical_enabled, |ui| {
                    ui.label("P:");
                    ui.add(
                        DragValue::new(&mut self.sampler.typical.p)
                            .range(0.0..=1.0)
                            .speed(0.01),
                    );
                    ui.label("Min-keep:");
                    ui.add(DragValue::new(&mut self.sampler.typical.min_keep));
                });
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.sampler.top_p_enabled, "Top-P");
                ui.add_enabled_ui(self.sampler.top_p_enabled, |ui| {
                    ui.label("P:");
                    ui.add(
                        DragValue::new(&mut self.sampler.top_p.p)
                            .range(0.0..=1.0)
                            .speed(0.01),
                    );
                    ui.label("Min-keep:");
                    ui.add(DragValue::new(&mut self.sampler.top_p.min_keep));
                });
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.sampler.temperature_enabled, "Temperature:");
                ui.add_enabled_ui(self.sampler.temperature_enabled, |ui| {
                    ui.add(
                        DragValue::new(&mut self.sampler.temperature.temperature)
                            .range(0.0..=1.0)
                            .speed(0.01),
                    );
                });
            });
        });
    }

    fn chat_template_ui(&mut self, ui: &mut Ui) {
        if let Some(gguf) = &self.gguf {
            ui.collapsing("Chat template", |ui| {
                ui.text_edit_multiline(&mut self.chat_template.template);
                if ui.button("Reset to default").clicked() {
                    self.chat_template = ChatTemplate::from_gguf(gguf)
                }
            });
        }
    }

    fn templated_prompt_ui(&mut self, ui: &mut Ui) {
        if !self.parsed_prompt.templated_prompt.is_empty() {
            ui.collapsing("Templated prompt & tokens", |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("prompt")
                    .max_height(100.0)
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(&self.parsed_prompt.templated_prompt)
                                .color(egui::Color32::BROWN),
                        );
                    });
                ui.separator();

                egui::ScrollArea::vertical()
                    .id_salt("tokens")
                    .max_height(200.0)
                    .show(ui, |ui| {
                        use egui_extras::{Column, TableBuilder};
                        let toks_per_line = 10;

                        let mut table = TableBuilder::new(ui);

                        for _ in 0..toks_per_line - 1 {
                            table = table.column(Column::auto().resizable(true));
                        }
                        table.column(Column::remainder()).body(|mut body| {
                            for tokens_chunk in self.parsed_prompt.tokens.chunks(toks_per_line) {
                                body.row(30.0, |mut row| {
                                    for (token, string) in tokens_chunk {
                                        let string = string.replace("\n", "\\\n");
                                        row.col(|ui| {
                                            ui.horizontal(|ui| {
                                                ui.label(
                                                    RichText::new(format!("'{}'", string))
                                                        .color(egui::Color32::BLACK),
                                                );

                                                ui.label(
                                                    RichText::new(format!("{token}"))
                                                        .color(egui::Color32::BROWN),
                                                );
                                            });
                                        });
                                    }

                                    // Complete the row if there isn’t enough token to fill it.
                                    for _ in tokens_chunk.len()..toks_per_line {
                                        row.col(|ui| {
                                            ui.label("");
                                        });
                                    }
                                });
                            }
                        });
                    });
            });
        }
    }

    fn prompt_ui(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Prompt:");
            ui.text_edit_multiline(&mut self.next_prompt);
        });

        ui.horizontal(|ui| {
            if ui.button("Submit").clicked() {
                if let Some(llm) = self.llm.clone() {
                    if let Response::Responding(resp) = &self.response {
                        self.prompt.append_assistant(resp.clone());
                    }

                    self.response = Response::Thinking;

                    let gpu = self.gpu.clone();

                    self.prompt.append_user(self.next_prompt.clone());

                    self.next_prompt.clear();

                    let prompt = self.prompt.clone();
                    let next_pos = self.next_pos;
                    let sampler = self.sampler;
                    let chat_template = self.chat_template.clone();
                    let (snd, rcv) = async_channel::unbounded();
                    self.response_rcv = Some(rcv);
                    execute(async move {
                        llm.forward(&gpu, prompt, sampler, chat_template, next_pos, snd)
                            .await
                    });
                } else {
                    self.response = Response::Empty;
                }
            };

            if ui.button("Clear").clicked() {
                self.response = Response::Empty;
                self.prompt.clear();
                self.response_rcv = None;
                self.next_pos = 0;
            }
        });
    }

    fn update_response(&mut self, ctx: &Context) {
        if let Some(rcv) = &self.response_rcv {
            while let Ok(event) = rcv.try_recv() {
                match event {
                    ChatEvent::PromptTokens(tokens) => {
                        self.parsed_prompt.tokens = tokens;
                    }
                    ChatEvent::TemplatedPrompt(prompt) => {
                        self.parsed_prompt.templated_prompt = prompt;
                    }
                    ChatEvent::Token {
                        string,
                        next_pos,
                        tok_per_second,
                    } => {
                        if let Response::Responding(ref mut curr) = &mut self.response {
                            curr.push_str(&string)
                        } else {
                            self.response = Response::Responding(string);
                        }
                        self.next_pos = next_pos;
                        self.tok_per_second = tok_per_second;
                    }
                }
            }

            if !rcv.is_closed() {
                // Keep repainting until the channel is closed.
                ctx.request_repaint();
            }
        }
    }

    fn receive_gguf_loading_events(&mut self) {
        // Load the gguf file once it comes in.
        if let Some((_, progress)) = &self.gguf_progress_channel {
            if let Ok(event) = progress.try_recv() {
                self.gguf_progress = Some(event);
            }
        }
    }

    fn complete_gguf_loading(&mut self) {
        // Load the gguf file once it comes in.
        if let Ok(gguf) = self.llm_channel.1.try_recv() {
            match ChatLlm::from_gguf(self.gpu.device(), &gguf) {
                Ok(llm) => {
                    self.llm = Some(Arc::new(llm));
                    self.error = "".to_string();
                }
                Err(err) => {
                    self.error = format!("{}", err);
                }
            }

            self.chat_template = ChatTemplate::from_gguf(&gguf);
            self.gguf = Some(Arc::new(gguf));
            self.gguf_progress = None;
        }
    }
}

impl eframe::App for BrowseApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.receive_gguf_loading_events();
        self.update_response(ctx);

        egui::SidePanel::left("File select")
            .resizable(true)
            .default_width(400.0)
            .show(ctx, |ui| {
                // a simple button opening the dialog
                if ui.button("📂 Open GGUF file").clicked() {
                    let sender = self.llm_channel.0.clone();
                    let task = rfd::AsyncFileDialog::new()
                        .add_filter("gguf model file", &["gguf"])
                        .pick_file();
                    // Context is wrapped in an Arc so it's cheap to clone as per:
                    // > Context is cheap to clone, and any clones refers to the same mutable data (Context uses refcounting internally).
                    // Taken from https://docs.rs/egui/0.24.1/egui/struct.Context.html
                    let (gguf_snd, gguf_rcv) = async_channel::unbounded();
                    self.gguf_progress_channel = Some((gguf_snd.clone(), gguf_rcv));
                    let ctx = ui.ctx().clone();
                    execute(async move {
                        let file = task.await;
                        if let Some(file) = file {
                            info!("Opening file");
                            let _ = gguf_snd.send(GgufLoadingProgress::ReadingFile).await;
                            let bytes = file.read().await;
                            info!("Bytes read");
                            let _ = gguf_snd.send(GgufLoadingProgress::ReadingTensors).await;
                            let gguf = Gguf::from_bytes(&bytes).unwrap();
                            let _ = gguf_snd
                                .send(GgufLoadingProgress::PopulatingGpuResources)
                                .await;
                            info!("gguf done");
                            let _ = sender.send(gguf);
                            ctx.request_repaint();
                        }
                    });
                }

                if !self.error.is_empty() {
                    ui.label(RichText::new(&self.error).color(egui::Color32::from_rgb(255, 0, 0)));
                }

                self.gguf_ui(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.prompt_ui(ui);

            ui.separator();

            self.perf_ui(ui);
            self.sampler_ui(ui);
            self.chat_template_ui(ui);
            self.templated_prompt_ui(ui);

            ui.separator();

            self.response_ui(ui);
        });

        self.complete_gguf_loading();
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn execute<F: Future<Output = ()> + Send + 'static>(f: F) {
    // this is stupid... use any executor of your choice instead
    std::thread::spawn(move || futures::executor::block_on(f));
}

#[cfg(target_arch = "wasm32")]
fn execute<F: Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}
