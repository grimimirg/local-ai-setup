# Local AI Development Environment Setup Guide

Complete step-by-step deployment guide for running a fully local AI coding assistant setup using **llama.cpp** as the backend server and **Aider CLI** as the coding frontend.

---

## Target OS & Environment
* **Primary OS:** Arch Linux / CachyOS & Debian / Ubuntu
* **Hardware Profile:** High-Performance CPU Setup (e.g., 64 GB System RAM)
* **Privacy:** 100% Local (No external API keys or cloud dependencies)

---

## 1. System Architecture Overview

This setup utilizes a decoupled client-server architecture designed for high efficiency, complete privacy, and modularity:

* **Backend (`llama-server`):** Compiled natively from `llama.cpp` C/C++ source code. It loads quantized GGUF weights into system RAM/VRAM and exposes an OpenAI-compatible HTTP REST API endpoint on `http://localhost:8080/v1`.
* **Frontend (`aider`):** A Git-integrated AI pair programming CLI tool. It inspects your project repository, reads file contexts, constructs structured prompts, communicates with `llama-server`, and automatically applies code modifications via Search & Replace blocks.

```
+-----------------------------------------------------------+
|                        Aider CLI                          |
|  (Interacts with Git, formats code context & edit blocks) |
+-----------------------------+-----------------------------+
                              | OpenAI-compatible API
                              v (HTTP / REST on Port 8080)
+-----------------------------------------------------------+
|                       llama-server                        |
|  (Inference Engine running quantized GGUF in System RAM)  |
+-----------------------------------------------------------+
```

---

## 2. Installing System Dependencies

To compile `llama.cpp` natively with maximum CPU optimizations and avoid Python package conflicts (particularly on newer Python releases like 3.14+), install the build toolchain and `uv` package manager:

### Arch Linux / CachyOS
```bash
sudo pacman -S base-devel cmake git python-uv
```

### Debian / Ubuntu
```bash
sudo apt update
sudo apt install build-essential cmake git python3-pip
# Install astral-sh uv for Debian/Ubuntu
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

---

## 3. Compiling `llama.cpp` from Source

Clone the official repository and build the release binaries utilizing all available CPU cores:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Configure build system and compile natively
cmake -B build
cmake --build build --config Release -j$(nproc)
```

> **Note:** Upon successful compilation, binaries including `llama-server` and `llama-cli` will be available in the `build/bin/` directory.

---

## 4. Isolated Installation of `aider`

To prevent system-level Python environment issues and avoid compilation errors with C/Fortran extension libraries (like `scipy` or `numpy`), enforce an isolated Python 3.12 environment using `uv`:

```bash
# Install Aider inside an isolated Python 3.12 tool environment
uv tool install --python 3.12 aider-chat

# Ensure user binaries are present in your shell PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc   # For Zsh (default on CachyOS)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # For Bash (default on Debian)
source ~/.zshrc || source ~/.bashrc
```

---

## 5. Model Selection, Hugging Face Download & Hardware Scaling

### Finding GGUF Models on Hugging Face
The `llama.cpp` backend operates exclusively on **GGUF** formatted models.
1. Visit [Hugging Face](https://huggingface.co).
2. Search for the model name followed by `GGUF` (e.g., `Qwen2.5-Coder-14B GGUF`).
3. Prefer high-quality quantization repos maintained by creators such as **bartowski** or **TheBloke**.

### Downloading the Model

#### Terminal / Automation (Recommended):
```bash
mkdir -p ~/models && cd ~/models
curl -L -O https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf
```

#### Manual Browser Download:
Open the repository page on Hugging Face, click on **Files and versions**, select your desired `.gguf` file, download it, and place it in `~/models/`.

### Hardware Scaling & Quantization Matrix

| System RAM | Recommended Model Size | Quantization | File Size | Context Window (`-c`) & RAM Usage |
| :--- | :--- | :--- | :--- | :--- |
| **8 GB RAM** | 3B – 7B Parameters | `Q4_K_M` | ~2.5 - 4.0 GB | Limit to `-c 4096` to avoid RAM swapping |
| **16 GB RAM** | 7B – 8B Parameters | `Q5_K_M` / `Q6_K` | ~5.5 - 7.0 GB | Up to `-c 16384` (~1.5 GB allocated for KV Cache) |
| **32 GB RAM** | 14B Parameters | `Q4_K_M` / `Q5_K_M` | ~9.0 - 11.0 GB | Up to `-c 32768` (~3.0 GB allocated for KV Cache) |
| **64 GB RAM (Optimal Setup)** | **14B – 32B Parameters** | **`Q6_K` / `Q8_0`** | **~12.0 - 20.0 GB** | **Optimal: `-c 32768` or `-c 65536` with total stability** |

---

## 6. Executable Commands & Quick Reference

### Command A: Launching `llama-server` (Backend Service)
Run this command from your terminal to start the OpenAI-compatible server:

```bash
/path/to/llama.cpp/build/bin/llama-server \
  -m ~/models/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf \
  -t 8 \
  -c 32768 \
  --port 8080
```

* `-m`: Absolute path to the local `.gguf` model file.
* `-t 8`: Number of physical CPU threads allocated to inference.
* `-c 32768`: Context window size in tokens (pre-allocates ~2-3 GB RAM for KV Cache).
* `--port 8080`: Network port serving the REST API.

---

### Command B: Launching `aider` (CLI Assistant)
In a separate terminal tab or inside your project repository root:

```bash
~/.local/bin/aider \
  --openai-api-base http://localhost:8080/v1 \
  --openai-api-key not-needed \
  --model openai/qwen2.5-coder
```

* `--openai-api-base`: Directs Aider to your local `llama-server`.
* `--openai-api-key`: Placeholder string (ignored by local server).
* `--model`: LiteLLM architecture identifier used by Aider to optimize edit formats.

---

### Command C: Native Interactive CLI Chat (Direct Terminal Q&A)
If you want to chat directly with the model without starting an HTTP server or running Aider:

```bash
/path/to/llama.cpp/build/bin/llama-cli \
  -m ~/models/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf \
  -c 32768 \
  -t 8 \
  -cnv
```

---

## 7. Recommended Shell Aliases

For seamless daily execution, add these convenience aliases to your shell configuration (`~/.zshrc` or `~/.bashrc`):

```bash
# Add to ~/.zshrc or ~/.bashrc
alias llama-coder="/path/to/llama.cpp/build/bin/llama-server -m ~/models/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf -t 8 -c 32768 --port 8080"
alias aider-local="~/.local/bin/aider --openai-api-base http://localhost:8080/v1 --openai-api-key not-needed --model openai/qwen2.5-coder"
```

Reload your config:
```bash
source ~/.zshrc # or source ~/.bashrc
```

Now you can start the backend with `llama-coder` and open Aider in any project directory with `aider-local`.

---

## 8. Technical FAQs & Deep Dive Concepts

### Q1: What is the `-c` (Context Window Size) parameter?
It defines the total token limit (1 token ≈ 3–4 English characters / code elements) the model can hold in memory simultaneously. This encompasses:
$$	ext{Total Context} = 	ext{System Prompt} + 	ext{Chat History} + 	ext{Project Source Files} + 	ext{Generated Output}$$

> **KV Cache Allocation:** `llama.cpp` pre-allocates RAM for the entire context window at launch to ensure memory stability. On a 64 GB system, setting `-c 32768` reserves ~2.5 GB of RAM for the KV Cache. A code project with ~3,000 lines fits cleanly inside a 32k context.

---

### Q2: Where is the `.gguf` file path defined?
The model file is loaded **exclusively by `llama-server`** via the `-m` parameter. Aider connects over HTTP and never touches the `.gguf` file on disk directly.

---

### Q3: Why specify `openai/qwen2.5-coder` as the model in Aider?
This is a LiteLLM identifier string required by Aider to structure its behavior:
1. `openai/`: Configures Aider to send standard OpenAI JSON payloads via HTTP POST.
2. `qwen2.5-coder`: Tells Aider what system prompt strategies and diff-formats (e.g., Search/Replace blocks) are optimal for the Qwen architecture.
