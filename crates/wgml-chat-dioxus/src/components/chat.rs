use crate::prompt::{ChatEvent, PromptEntry};
use crate::{GpuInstanceCtx, LoadedModel, LoadedModelSignal, PromptResponse, PromptState};
use dioxus::prelude::*;
use dioxus_markdown::Markdown;
use futures_util::StreamExt;
use std::future::Future;
use std::rc::Rc;

#[component]
pub fn User(mut text: String) -> Element {
    rsx! {
        div {
            class: "user",
            b {
                Markdown {
                    src: "👤 ".to_owned() + &text
                }
            }
        }
    }
}

#[component]
pub fn Assistant(text: String, bubble_index: usize, show_reasoning: Signal<Vec<bool>>) -> Element {
    rsx! {
        div {
            class: "assistant",
            if !text.contains("<think>") {
                // No reasoning mechanism for this model.
                Markdown {
                    src: "🤖 ".to_owned() + &text
                }
            } else {{
                // This model has a reasoning mechanism, render it in a collapsible
                // section.
                const THINK_OPEN_TAG: &str = "<think>";
                const THINK_CLOSE_TAG: &str = "</think>";
                const THINK_OPEN_TAG_LEN: usize = THINK_OPEN_TAG.len();
                const THINK_CLOSE_TAG_LEN: usize = THINK_CLOSE_TAG.len();

                let start_bytes = text.find(THINK_OPEN_TAG).unwrap_or(0);
                let end_bytes = text.find(THINK_CLOSE_TAG).unwrap_or(text.len());

                let thinking = &text[start_bytes + THINK_OPEN_TAG_LEN..end_bytes];

                rsx! {
                    div {
                        class: "think",
                        button {
                            class: if show_reasoning.read()[bubble_index] { "active_think_header" } else { "think_header" },
                            onclick: move |_| {
                                let show_reasoning = &mut *show_reasoning.write();
                                show_reasoning[bubble_index] = !show_reasoning[bubble_index];
                            },
                            "🧠 View reasoning."
                        }

                        if show_reasoning.read()[bubble_index] {
                            div {
                                class: "think_text",
                                Markdown {
                                    src: thinking
                                }
                            }
                        }
                    }

                    if end_bytes < text.len() {
                        Markdown {
                            src: "🤖 ".to_owned() + &text[end_bytes + THINK_CLOSE_TAG_LEN..].trim()
                        }
                    }
                }
            }}
        }
    }
}

#[component]
pub fn Chat() -> Element {
    let mut user_prompt = use_signal(|| "Give me a chocolate cake recipe.".to_string()); // "Give me a chocolate cake recipe.".to_string());
    let mut tok_per_sec = use_signal(|| "".to_string());
    let mut show_reasoning = use_signal(|| vec![true]);
    let mut show_metadata = use_signal(|| false);
    let mut show_tensors = use_signal(|| false);
    let mut tok_per_sec_element: Signal<Option<Rc<MountedData>>> = use_signal(|| None);
    let autoscroll = use_signal(|| false);

    let gpu = use_context::<GpuInstanceCtx>().clone();
    let model = use_context::<LoadedModelSignal>();
    let mut prompt_state = use_context::<Signal<PromptState>>();

    {
        let mut reasoning = show_reasoning.write();
        let reasoning_entries = prompt_state.read().prompt.entries().len() + 1;
        if reasoning.len() < reasoning_entries {
            reasoning.resize(reasoning_entries, true);
        }
    }

    let chat_event_handler =
        use_coroutine(move |mut rx: UnboundedReceiver<ChatEvent>| async move {
            while let Some(event) = rx.next().await {
                match event {
                    ChatEvent::Token {
                        string,
                        next_pos: _,
                        token_count,
                        token_time,
                    } => {
                        let tok_per_second = if token_count == 0 {
                            0.0
                        } else {
                            token_count as f64 / token_time
                        };
                        *tok_per_sec.write() = format!(
                            "Generated {} tokens in {:.2}s ({:.2} tokens/second).",
                            token_count, token_time, tok_per_second
                        );

                        let mut prompt_state = prompt_state.write();
                        if let PromptResponse::Responding(ref mut curr) = &mut prompt_state.response
                        {
                            curr.push_str(&string)
                        } else {
                            prompt_state.response = PromptResponse::Responding(string);
                        }
                    }
                    _ => {}
                }
            }
        });

    let _ = use_resource(move || async move {
        // Not used but we want to trigger this closure any time text
        // is added to the prompt.
        if *autoscroll.read() {
            let _ = prompt_state.read();
            if let Some(tok_per_sec_element) = tok_per_sec_element.cloned() {
                let _ = tok_per_sec_element.scroll_to(ScrollBehavior::Instant).await;
            }
        }
    });

    let gpu0 = gpu.clone();

    // TODO: this currently crashes in the markdown dependecy
    // let mut components = CustomComponents::new();
    // components.register("think", |props| {
    //     Ok(rsx! {
    //         "Hello world"
    //     })
    // });

    rsx! {
        div {
            id: "chat",
            div {
                class: "metadata",
                if let Some(model) = model.as_ref() {
                    div {
                        class: "think",
                        button {
                            class: if *show_metadata.read() { "active_think_header" } else { "think_header" },
                            onclick: move |_| {
                                let show_metadata = &mut *show_metadata.write();
                                *show_metadata = !*show_metadata;
                            },
                            "🏷️ View GGUF metadata."
                        }

                        if *show_metadata.read() {
                            div {
                                class: "think_text",
                                // NOTE: kind of hacky, but we are delegating to Markdown to generate
                                //       the html elements for the list.
                                Markdown {
                                    src: { let mut s = model.metadata.metadata.join("\n- "); s.insert_str(0, "- "); s }
                                }
                            }
                        }
                    }
                    br {}
                    div {
                        class: "think",
                        button {
                            class: if *show_tensors.read() { "active_think_header" } else { "think_header" },
                            onclick: move |_| {
                                let show_tensors = &mut *show_tensors.write();
                                *show_tensors = !*show_tensors;
                            },
                            "📁 View GGUF tensors."
                        }

                        if *show_tensors.read() {
                            div {
                                class: "think_text",
                                // NOTE: kind of hacky, but we are delegating to Markdown to generate
                                //       the html elements for the list.
                                Markdown {
                                    src: { let mut s = model.metadata.tensors.join("\n- "); s.insert_str(0, "- "); s }
                                }
                            }
                        }
                    }
                }
            }

            // NOTE: don’t disply the assistant bubble if there is on repsonse.
            if !matches!(prompt_state.read().response, PromptResponse::Empty) {
                for (i, entry) in prompt_state.read().prompt.entries().iter().enumerate() {
                    match entry.clone() {
                        PromptEntry::User(text) => rsx! {
                            User {
                                text
                            }
                        },
                        PromptEntry::System(text) => rsx! {
                            User {
                                text
                            }
                        },
                        PromptEntry::Assistant(text) => rsx! {
                            Assistant {
                                text,
                                bubble_index: i,
                                show_reasoning,
                            }
                        }
                    }
                }

                match &prompt_state.read().response {
                    PromptResponse::Empty => unreachable!(),
                    PromptResponse::Thinking => {
                       rsx! {
                            Assistant {
                                text: "_Parsing prompt…_".to_string(),
                                bubble_index: show_reasoning.read().len() - 1,
                                show_reasoning: show_reasoning
                            }
                       }
                    }
                    PromptResponse::Responding(resp) => {
                        rsx! {
                            Assistant {
                                text: resp.clone(),
                                bubble_index: show_reasoning.read().len() - 1,
                                show_reasoning: show_reasoning
                            }
                        }
                    }
                }
                div {
                    id: "tok_per_sec",
                    onmounted: move |cx| tok_per_sec_element.set(Some(cx.data())),
                    {tok_per_sec}
                }
            }
        }
        div {
            id: "footer",
            textarea {
                id: "user_prompt",
                // we tell the component what to render
                value: "{user_prompt}",
                // and what to do when the value changes
                oninput: move |event| user_prompt.set(event.value()),
                placeholder: "Type your message…"
            }
            button {
                id: "submit",
                onclick: move |_| {
                    if let Some(model) = (*model.read()).clone() {
                        submit_prompt(gpu0.clone(), user_prompt.read().clone(), &model, &mut prompt_state.write(), chat_event_handler.tx());
                    }
                },
                "Submit"
            }
            // input {
            //     r#type: "checkbox",
            //     onchange: move |_| {
            //         let enabled = *autoscroll.read();
            //         autoscroll.set(!enabled);
            //     },
            //     checked: *autoscroll.read(),
            //     "Auto-scroll"
            // }
        }
    }
}

fn submit_prompt(
    gpu: GpuInstanceCtx,
    user_text: String,
    model: &LoadedModel,
    state: &mut PromptState,
    out: UnboundedSender<ChatEvent>,
) {
    if let PromptResponse::Responding(resp) = &state.response {
        state.prompt.append_assistant(resp.clone());
    }

    state.response = PromptResponse::Thinking;
    state.prompt.append_user(user_text);

    // state.next_prompt.clear();

    let prompt = state.prompt.clone();
    let next_pos = 0; // FIXME: state.next_pos;
    let sampler = model.sampler;
    let chat_template = model.template.clone();
    // let (snd, rcv) = async_channel::unbounded();
    // state.response_rcv = Some(rcv);

    let gpu = gpu.clone();
    let llm = model.llm.clone();
    execute(async move {
        llm.forward(&gpu, prompt, sampler, chat_template, next_pos, out)
            .await
    });
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
