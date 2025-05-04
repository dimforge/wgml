# wgml − GPU local inference every platform

<p align="center">
  <img src="./crates/wgml-chat/assets/wgml logo.png" alt="crates.io" height="200px">
</p>
<p align="center">
    <a href="https://discord.gg/vt9DJSW">
        <img src="https://img.shields.io/discord/507548572338880513.svg?logo=discord&colorB=7289DA">
    </a>
</p>

-----

**wgml** is a set of [Rust](https://www.rust-lang.org/) libraries exposing [WebGPU](https://www.w3.org/TR/WGSL/) shaders
and kernels for local Large Language Models (LLMs) inference on the GPU. It is cross-platform and runs on the web.
**wgml** can be used as a rust library to assemble your own transformer from the provided operators (and write your
owns on top of it).

Aside from the library, two binary crates are provided:
- **wgml-bench** is a basic benchmarking utility for measuring calculation times for matrix multiplication with various
  quantization formats.
- **wgml-chat** is a basic chat GUI application for loading GGUF files and chat with the model. It can be run natively
  or on the browser. Check out its [README](./crates/wgml-chat/README.md) for details on how to run it. You can run
  it from your browser with the [online demo](https://wgmath.rs/demos/wgml/index.html).

⚠️ **wgml** is still under heavy development and might be lacking some important features. Contributions  are welcome!

----