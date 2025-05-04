# wgml-chat

This is a basic chat interface for testing **wgml**. Currently only supports models of the Llama family and Qwen 2.

To run natively:
```bash
cargo run --features desktop
```
To run of the browser:
```sh
dx serve --release
```

Note that due to the WebAssembly 4GB memory limitation, the size of the models that can be loaded when running the web
version is fairly limited. Opening GGUF files that are too big wrt. that memory limit will crash with an out-of-memory
error (that gits printed on the console only).
