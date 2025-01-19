# PicForge: Telegram Image Generation Bot

## Overview

PicForge is an experimental Telegram bot that leverages AI to generate images based on user prompts. This project was developed as a fun exploration into AI-assisted image generation and bot development, primarily using v0.dev and ChatGPT.

🎨 Create stunning images with simple text prompts
🤖 Interact through an intuitive Telegram interface
⚙️ Customize various generation parameters for fine-tuned results

## Disclaimer

This bot is in an early, experimental stage. The code may contain bugs, inefficiencies, or suboptimal solutions. It's a project born out of curiosity and the desire to learn, rather than a production-ready application. Use at your own risk and ensure compliance with all relevant laws and regulations.

## Features

- Text-to-image generation using state-of-the-art AI models
- Customizable parameters: model selection, VAE, LoRA, sampler, CFG scale, steps, and image size
- Interactive settings panel for easy parameter adjustments
- Queue system for handling multiple generation requests
- Basic error handling

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)

### Required Libraries

Install the following libraries using pip:

```bash
pip install python-telegram-bot torch diffusers transformers Pillow asyncio PyYAML
```

### Setup

1. Clone the repository:

```shellscript
git clone https://github.com/your-username/picforge.git
cd picforge
```


2. Create a `config.yaml` file in the project root directory:

```yaml
bot_token: "YOUR_TELEGRAM_BOT_TOKEN"
allowed_user_id: YOUR_TELEGRAM_USER_ID
models_path: "./models/checkpoints"
lora_path: "./models/lora"
vae_path: "./models/vae"
output_path: "./output"
log_path: "./logs"
default_model: "sd1#HUGGINGFACE/LINK"
default_model_type: "sd1"
default_precision: "fp16"
use_xformers: true
```


3. Create the necessary directories:

```shellscript
mkdir -p models/checkpoints models/lora models/vae output logs
```




## Obtaining Models

1. Stable Diffusion models:

- [Civitai](https://civitai.com/)
- [Hugging Face](https://huggingface.co/models)



2. LoRA models:

- [Civitai LoRAs](https://civitai.com/models?type=LORA)



3. VAE models:

- [Stable Diffusion VAEs](https://huggingface.co/stabilityai)





Download the models and place them in the corresponding directories specified in `config.yaml`.

## Usage

1. Start the bot:

```shellscript
python main.py
```


2. Open your Telegram client and start a chat with your bot.
3. Use the `/start` or `/s` command to display the interactive settings panel.
4. Adjust generation parameters using the panel buttons or quick setup commands.
5. Send a text message with your prompt or use the `/generate` command to start image generation.
6. Wait for the generation process to complete and receive the result.


## Bot Commands

- `/start` or `/s` - Display the interactive settings panel
- `/generate` or `/g` - Start generation with current settings
- `/help` or `/h` - Show command help


Quick setup commands:

- `/set_model` or `/sm <model>` - Set the model
- `/set_vae` or `/sv <vae>` - Set the VAE
- `/set_lora` or `/sl <lora>` - Set the LoRA
- `/set_sampler` or `/ss <sampler>` - Set the sampler
- `/set_cfg_scale` or `/sc <value>` - Set the CFG Scale
- `/set_steps` or `/st <number>` - Set the number of steps
- `/set_size` or `/sz <width>x<height>` - Set the image size
- `/set_prompt` or `/sp <prompt>` - Set the prompt
- `/set_negative_prompt` or `/sn <negative prompt>` - Set the negative prompt


## Tips

- Experiment with different models and settings to achieve desired results.
- Use LoRA for fine-tuning the generation style.
- Utilize negative prompts to exclude unwanted elements from the generation.


## Troubleshooting

- If the bot is unresponsive, check the logs in the `logs` directory.
- Ensure you have sufficient disk space for saving generated images.
- For model loading issues, verify the paths in `config.yaml`.
- Make sure your Telegram user ID is correctly set in `config.yaml`.


## Contributing

While this project was primarily a personal endeavor, contributions, suggestions, and feedback are welcome. Please keep in mind the experimental nature of the code when proposing changes or improvements.

## License

[GPL-3.0 license](https://github.com/s-andrianov/picforge/?tab=GPL-3.0-1-ov-file)

## Acknowledgements

This project wouldn't have been possible without the incredible AI tools and communities that inspire and assist developers every day. Special thanks to:

- v0.dev for AI-assisted development
- ChatGPT for code suggestions and problem-solving assistance
- The open-source community for providing invaluable resources and inspiration


Happy forging with PicForge! 🎨🤖
