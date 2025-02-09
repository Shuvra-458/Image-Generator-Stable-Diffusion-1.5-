# GPT-2 Prompt Optimization and Image Generation

This project integrates GPT-2 for prompt optimization and uses ComfyUI with Stable Diffusion to generate images based on the optimized prompts.

## Features

- **Prompt Optimization**: Uses GPT-2 to enhance user-provided prompts.
- **Image Generation**: Leverages ComfyUI and Stable Diffusion to generate images.

## Installation

### Prerequisites
- Python 3.10 or higher
- `pip` package manager
- Git
- PyTorch
- NumPy
- Flask
- Hugging Face CLI
- ComfyUI (with Stable Diffusion 1.5)

### Setup

# Install and authenticate Hugging Face
pip install huggingface_hub
huggingface-cli login
```

## Usage
- Run the ComfyUI server first in the respective directory.
- After running the server, run the program prompter.py through cmd and provide the desired prompt for the required image.


## Troubleshooting

- If you encounter a missing node error in ComfyUI, ensure that node names match exactly in your configuration.
- If `huggingface-cli` is not found, try reinstalling: `pip install --upgrade huggingface_hub`.
- For permission issues, run commands with administrator privileges.


