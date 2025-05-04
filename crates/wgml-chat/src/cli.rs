use crate::app::ParsedPrompt;
use crate::chat_template::ChatTemplate;
use crate::llm::ChatLlm;
use crate::prompt::{ChatEvent, Prompt};
use crate::sampler::SamplerParams;
use clap::Parser;
use colored::Colorize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use wgcore::gpu::GpuInstance;
use wgml::gguf::Gguf;

#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Cli {
    pub path: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub headless: bool,
}

pub async fn run_headless(cli: &Cli) -> anyhow::Result<()> {
    let gpu = Arc::new(GpuInstance::new().await?);
    let Some(path) = &cli.path else {
        println!("{}", "No model file provided, exiting.".red());
        return Ok(());
    };
    println!("{}", format!("Loading GGUF file: {:?}", cli.path).dimmed());
    let t_gguf = std::time::Instant::now();
    let file = File::open(&path).expect("Unable to open the GGUF model file");
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gguf = Gguf::from_bytes(&mmap[..])?;
    println!(
        "{}",
        format!(
            "GGUF model loaded in {:.2} seconds.",
            t_gguf.elapsed().as_secs_f32()
        )
        .dimmed()
    );
    let t_chat_llm = std::time::Instant::now();
    let llm = Arc::new(ChatLlm::from_gguf(gpu.device(), &gguf)?);
    let chat_template = ChatTemplate::from_gguf(&gguf);
    println!(
        "{}",
        format!(
            "Uploaded model to GPU in {:.2} seconds.",
            t_chat_llm.elapsed().as_secs_f32()
        )
        .dimmed()
    );

    println!("{}", "Starting interactive chat:".dimmed());
    let mut prompt = Prompt::default();
    let mut next_pos = 0;
    let mut tok_per_second = 0.0;
    let sampler = SamplerParams::default();

    loop {
        // Read stdin.
        println!("{}", "[User]".purple().bold());
        let mut user_prompt = String::new();
        std::io::stdout().flush()?;
        std::io::stdin().read_line(&mut user_prompt)?;
        user_prompt.truncate(user_prompt.trim_end().len());
        prompt.append_user(user_prompt);

        // Forward the transformer.
        let (snd, rcv) = async_channel::unbounded();

        {
            let gpu = gpu.clone();
            let llm = llm.clone();
            let prompt = prompt.clone();
            let sampler = sampler;
            let chat_template = chat_template.clone();
            async_std::task::spawn(async move {
                llm.forward(&gpu, prompt, sampler, chat_template, next_pos, snd)
                    .await
            });
        }

        let mut full_response = String::new();
        let mut last_tok = String::new();
        println!("{}", "[Assistant]".green().bold());

        let mut k = 0;
        while let Ok(event) = rcv.recv().await {
            match event {
                ChatEvent::Token {
                    string,
                    next_pos: next,
                    tok_per_second: tps,
                } => {
                    // Don’t print multiple newlines, takes too
                    // much room on the console.
                    if last_tok != "\n" || string != "\n" {
                        print!("{}", string);
                    }
                    next_pos = next;
                    tok_per_second = tps;
                    full_response.push_str(&string);
                    last_tok = string;
                    std::io::stdout().flush()?;
                }
                _ => {}
            }
        }
        println!();
        println!(
            "{}",
            format!(
                "({:.2} tok/s) − generated {} tokens",
                tok_per_second, next_pos
            )
            .italic()
            .dimmed()
        );

        prompt.append_assistant(full_response);
    }

    Ok(())
}
