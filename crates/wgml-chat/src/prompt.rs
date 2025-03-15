use minijinja::{context, Value};
use std::fmt::{Display, Formatter};

pub enum ChatEvent {
    Token {
        string: String,
        next_pos: usize,
        tok_per_second: f64,
    },
    TemplatedPrompt(String),
    PromptTokens(Vec<(usize, String)>),
}

#[derive(Debug, Clone)]
pub enum PromptEntry {
    User(String),
    Assistant(String),
    System(String),
}

impl PromptEntry {
    pub fn as_str(&self) -> &str {
        match self {
            Self::User(ref s) => s.as_ref(),
            Self::Assistant(ref s) => s.as_ref(),
            Self::System(ref s) => s.as_ref(),
        }
    }
}

impl Display for PromptEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User(name) => write!(f, "[User] {}", name),
            Self::Assistant(name) => write!(f, "[Assistant] {}", name),
            Self::System(name) => write!(f, "[System] {}", name),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Prompt {
    entries: Vec<PromptEntry>,
}

impl Display for Prompt {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for entry in &self.entries {
            writeln!(f, "{}", entry)?;
        }

        Ok(())
    }
}

impl Prompt {
    pub fn entries(&self) -> &[PromptEntry] {
        &self.entries
    }

    /// Append a text typed by the user.
    pub fn append_user(&mut self, text: String) {
        self.entries.push(PromptEntry::User(text));
    }

    /// Append a response from the chatbot.
    pub fn append_assistant(&mut self, text: String) {
        self.entries.push(PromptEntry::Assistant(text));
    }

    /// Append system configuration for the chatbot.
    pub fn append_system(&mut self, text: String) {
        self.entries.push(PromptEntry::System(text));
    }

    /// Converts to a jinja-compatible value for inputting to
    /// a chat template.
    pub fn to_jinja_input(&self) -> Value {
        let messages = self.to_jinja_messages();
        context!(messages => messages)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn to_jinja_messages(&self) -> Vec<Value> {
        let mut ctx = vec![];

        for entry in &self.entries {
            match entry {
                PromptEntry::User(text) => {
                    ctx.push(context!(role => "user", content => text));
                }
                PromptEntry::Assistant(text) => {
                    ctx.push(context!(role => "assistant", content => text));
                }
                PromptEntry::System(text) => {
                    ctx.push(context!(role => "system", content => text));
                }
            }
        }

        ctx
    }
}
