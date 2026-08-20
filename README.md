# Local AI Development Environment Setup

**Complete Step-by-Step Deployment Guide: llama.cpp Backend & Aider CLI Frontend**  
**Target OS:** Arch Linux / CachyOS & Debian/Ubuntu | **Hardware Spec:** High-Performance CPU Setup

---

## 1. System Architecture Overview

This setup utilizes a decoupled client-server architecture designed for high efficiency and total local privacy:

* **Backend (`llama-server`):** Compiled natively from `llama.cpp` source code. It loads quantized LLM weights into system RAM and exposes an OpenAI-compatible HTTP REST API on port `8080`.
* **Frontend (`aider`):** A Git-integrated AI coding CLI tool. It inspects project files, formats edit prompts, communicates with the backend server, and directly applies modifications to your codebase.

---

## 2. Installing System Dependencies

To compile `llama.cpp` natively and prevent Python package conflicts (particularly on newer releases like Python 3.14), install the required build tools and package managers:

### CachyOS / Arch Linux
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

## 3. Compiling llama.cpp from Source

Clone the official repository and build the release binaries leveraging full CPU parallel processing:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Configure build system and compile natively
cmake -B build
cmake --build build --config Release -j$(nproc)
```

Upon completion, the server binary `llama-server` and CLI binary `llama-cli` will be available inside `build/bin/`.

---

## 4. Isolated Installation of Aider

To prevent system-level Python conflicts and avoid compilation failures of underlying C/Fortran libraries (such as `scipy` or `numpy` under newer Python runtimes), force an isolated Python 3.12 environment using `uv`:

```bash
# Install Aider inside an isolated Python 3.12 environment
uv tool install --python 3.12 aider-chat

# Ensure user binaries are present in your shell PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc   # For Zsh (default on CachyOS)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc  # For Bash (default on Debian)

source ~/.zshrc || source ~/.bashrc
```

---

## 5. Model Selection, Hugging Face Download & Hardware Scaling

### Finding GGUF Models on Hugging Face
The `llama.cpp` backend operates exclusively on GGUF formatted models. To locate them on Hugging Face:
* Search for the model family name appended with GGUF (e.g., `Qwen2.5-Coder-14B GGUF`).
* Rely on well-maintained quantization repositories from community creators such as `bartowski` or `TheBloke`.

### Download Methods
* **Terminal / Automation (Recommended):** Fetch using `curl` or `wget`:
  ```bash
  mkdir -p ~/models && cd ~/models
  curl -L -O https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf
  ```
* **Manual Browser Download:** Download directly from the Hugging Face *Files and versions* tab into `~/models/`.

### Hardware Scaling & Quantization Matrix
Model selection depends heavily on available System RAM or GPU VRAM. Quantization formats (e.g., `Q4_K_M`, `Q6_K`) compress weights while retaining accuracy.

| System RAM | Recommended Model Size | Quantization | File Size | Context Window (`-c`) & RAM Usage |
| :--- | :--- | :--- | :--- | :--- |
| **8 GB RAM** | 3B–7B Parameters | `Q4_K_M` | ~2.5–4.0 GB | Limit to `-c 4096` to avoid memory swapping |
| **16 GB RAM** | 7B–8B Parameters | `Q5_K_M` / `Q6_K` | ~5.5–7.0 GB | Up to `-c 16384` (~1.5 GB allocated for KV Cache) |
| **32 GB RAM** | 14B Parameters | `Q4_K_M` / `Q5_K_M` | ~9.0–11.0 GB | Up to `-c 32768` (~3.0 GB allocated for KV Cache) |
| **64 GB RAM** *(Your Setup)* | 14B–32B Parameters | `Q6_K` / `Q8_0` | ~12.0–20.0 GB | Optimal: `-c 32768` or `-c 65536` with total stability |

---

## 6. Executable Commands & Shell Automation

### Option A: Launching `llama-server` (Backend Service)
```bash
/percorso/di/llama.cpp/build/bin/llama-server   -m ~/models/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf   -t 8 -c 32768 --port 8080
```

### Option B: Launching `aider` (CLI Client)
```bash
~/.local/bin/aider   --openai-api-base http://localhost:8080/v1   --openai-api-key not-needed   --model openai/qwen2.5-coder
```

### Option C: Native Interactive CLI Chat (Without Aider or HTTP Server)
```bash
/percorso/di/llama.cpp/build/bin/llama-cli   -m ~/models/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf   -c 32768 -t 8 -cnv
```

### Recommended Shell Aliases
Add these aliases to your `~/.bashrc` or `~/.zshrc`:

```bash
alias llama-coder="./build/bin/llama-server -m ~/models/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf -t 8 -c 32768 --port 8080"
alias aider-local="~/.local/bin/aider --openai-api-base http://localhost:8080/v1 --openai-api-key not-needed --model openai/qwen2.5-coder"
```

---

## 7. Technical Concept Reference & FAQs

### What is the `-c` (Context Window Size) parameter?
It specifies the maximum number of tokens (1 token ≈ 3–4 characters) the model can hold in memory at once (`System Prompt` + `Chat History` + `Source Code` + `Model Output`).

**KV Cache Pre-allocation:** `llama.cpp` pre-allocates memory for the full context window upon startup to guarantee stability. On a 64 GB RAM setup, `-c 32768` reserves ~2.5 GB RAM. An application of 3,000 lines of code equals roughly 30,000–35,000 tokens, fitting comfortably inside a 32k context window.

### Where is the `.gguf` file specified?
The `.gguf` path is passed exclusively to `llama-server` via the `-m` flag. Aider never accesses the model file directly.

### Why use the model identifier `openai/qwen2.5-coder` in Aider?
This string is a LiteLLM identifier used by Aider to configure its internal logic:
* `openai/` tells Aider to send requests via the OpenAI API protocol over HTTP.
* `qwen2.5-coder` informs Aider of the model family capabilities, optimizing its system prompts and code-editing strategy.

---

## 8. Performance Factors & Latency Troubleshooting (Why is it slow?)

When asking Aider to read large files (e.g., `README.md` or source files), you may experience long delays before receiving the first token (*Time To First Token* / TTFT). The key performance variables are:

* **Prompt Processing Overhead (TTFT):** Before generating text, `llama-server` must process and compute the attention matrix for all context tokens (System Prompt + Aider Repo Map + Target File). Reading a large file forces a massive initial evaluation phase.
* **CPU Thread Tuning (`-t` Parameter):** Setting `-t` incorrectly causes severe performance degradation.
  * **Rule:** Set `-t` equal to the number of **physical CPU cores** (e.g., `-t 8` on an 8-core CPU).
  * **Avoid:** Setting `-t` to logical threads (Hyper-Threading / SMT) leads to CPU cache thrashing and slower prompt evaluation.
* **System Memory Bandwidth (RAM Bottleneck):** LLM CPU inference is constrained by RAM speed (GB/s), not CPU clock rate. Dual-channel DDR5 provides significantly higher token generation speeds than DDR4. Full GPU offloading (via VRAM) is orders of magnitude faster than CPU RAM.
* **Quantization Level Impact:** Heavy quantizations (`Q8_0` or `Q6_K`) require moving more data through system RAM per token compared to `Q4_K_M`.
* **Aider's Auto Repo Map:** Aider uses `tree-sitter` to parse your Git repository and automatically sends a symbol map. In large projects, this inflates the token count sent to `llama-server` on the very first prompt.

### Quick Diagnostic Checklist for Slow Responses
1. Check the terminal running `llama-server` for metrics:
   * `prompt eval time`: Indicates initial context processing speed.
   * `eval time`: Indicates token generation speed (tokens/sec).
2. Confirm `-t` matches your physical CPU cores (not logical threads).
3. Note that subsequent questions on the same file/context are much faster because `llama-server` reuses the prompt cache.

# Setup Guide: Crush (Charm) as an Alternative to Aider

This guide covers how to install, configure, and use **[Crush](https://github.com/charmbracelet/crush)** by Charmbracelet as a modern, terminal-based AI coding assistant and alternative to Aider, integrated with a local `llama.cpp` backend.

---

## 1. Overview & Comparison

| Feature | Aider | Crush |
| :--- | :--- | :--- |
| **Developer / Ecosystem** | Independent (`aider-chat`) | Charmbracelet (`charm.sh`) |
| **UI / TUI Design** | Classic CLI text-stream | Rich TUI powered by Bubble Tea & Lip Gloss |
| **Backend Integration** | LiteLLM / OpenAI API Protocol | Native / OpenAI-compatible provider schema |
| **Configuration** | Command-line flags, `.aider.conf.yml` | Declarative JSON configuration (`crush.json`) |

---

## 2. Installation

### Package Managers

#### macOS & Linux (Homebrew)
```bash
brew install charmbracelet/tap/crush
```

#### Arch Linux / CachyOS (AUR)
```bash
yay -S crush-bin
# or
paru -S crush-bin
```

#### Debian / Ubuntu
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.charm.sh/apt/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/charm.gpg
echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" | sudo tee /etc/apt/sources.list.d/charm.list
sudo apt update && sudo apt install crush
```

### Direct Download / Go Install
If you have Go (1.22+) installed, build directly from source:
```bash
go install github.com/charmbracelet/crush@latest
```

---

## 3. Configuration Setup

Crush uses a JSON schema located in `~/.config/crush/crush.json` to define custom models and local backends.

1. Create the configuration directory:
   ```bash
   mkdir -p ~/.config/crush
   ```

2. Create and edit the configuration file `~/.config/crush/crush.json`:
   ```bash
   nano ~/.config/crush/crush.json
   ```

3. Insert the following configuration structure:

```json
{
  "$schema": "https://charm.land/crush.json",
  "providers": {
    "llama-local": {
      "id": "local-qwen3.8",
      "name": "local-qwen3.8",
      "type": "openai-compat",
      "base_url": "http://localhost:8080/v1",
      "api_key": "not-needed",
      "models": [
        {
          "id": "local-qwen3.8",
          "name": "Qwen 3.8 Coder",
          "context_window": 32768,
          "default_max_tokens": 4096,
          "can_reason": true,
          "supports_tools": true
        }
      ]
    }
  }
}
```

---

## 4. Running the Complete Stack

### Step 1: Launch the Local `llama.cpp` Backend
Ensure your `llama-server` is up and running on port `8080`:

```bash
/percorso/di/llama.cpp/build/bin/llama-server   -m ~/models/Qwen2.5-Coder-14B-Instruct-Q6_K.gguf   -t 8 -c 32768 --port 8080
```

### Step 2: Start Crush
Launch Crush inside your project repository:

```bash
crush --provider llama-local --model local-qwen3.8
```

---

## 5. Shell Aliases & Automation

Add an alias to your `~/.bashrc` or `~/.zshrc` for quick access:

```bash
alias crush-local="crush --provider llama-local --model local-qwen3.8"
```

After updating your configuration (`source ~/.zshrc`), you can start coding with Crush simply by typing:

```bash
crush-local
```
