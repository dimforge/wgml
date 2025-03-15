use crate::prompt::Prompt;
use regex::Regex;
use wgml::gguf::Gguf;
use wgml::models::llama2;

#[derive(Clone, Debug, Default)]
pub struct ChatTemplate {
    pub template: String,
}

impl ChatTemplate {
    pub fn new(mut template: String) -> Self {
        template = template
            .replace(".split(", "|split(")
            .replace("[-1]", "|last");
        Self { template }
    }

    pub fn from_gguf(gguf: &Gguf) -> Self {
        let default_chat_template =
            || ChatTemplate::new(llama2::LlamaTokenizer::CHAT_TEMPLATE.into());
        gguf.metadata
            .get("tokenizer.chat_template")
            .map(|val| ChatTemplate::new(val.as_string().clone()))
            .unwrap_or_else(default_chat_template)
    }

    pub fn apply(&self, prompt: &Prompt, bos: &str, eos: &str) -> String {
        use minijinja::{context, Environment};

        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.add_global("bos_token", bos);
        env.add_global("eos_token", eos);
        env.add_global("add_generation_prompt", true);
        env.add_template("main", &self.template).unwrap();

        let tmpl = env.get_template("main").unwrap();
        let result = tmpl.render(prompt.to_jinja_input()).unwrap();
        result
    }
}
