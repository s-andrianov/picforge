import asyncio
import time
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, DPMSolverMultistepScheduler, AutoencoderKL
from typing import Dict, Any
import os

class ImageGenerator:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.scheduler = None
        self.vae = None
        self.compiled_model = None

    def parse_model_data(self, model_data):
        model_type, model_name = model_data.split("#", 1)
        return model_type, model_name
    
    async def load_model(self, model_data: str, vae_name: str = None):
        model_type, model_name = self.parse_model_data(model_data)
        model_path = os.path.normpath(os.path.join(self.config.models_path, model_data))
        if not os.path.exists(model_path):
            model_path = model_name  # HuggingFace модель

        # Выбор пайплайна в зависимости от типа модели
        pipeline_class = StableDiffusionXLPipeline if model_type in ['pony', 'xl'] else StableDiffusionPipeline

        # Загружаем модель
        if model_path.endswith('.safetensors'):
            print(f"Загрузка модели (файл): {model_path}")
            self.model = pipeline_class.from_single_file(
                model_path,
                use_safetensors=True, 
                torch_dtype=torch.float16 if self.config.default_precision == 'fp16' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            print(f"Загрузка модели (кэш/онлайн): {model_path}")
            self.model = AutoPipelineForText2Image.from_pretrained(
                model_path,
                use_safetensors=True, 
                torch_dtype=torch.float16 if self.config.default_precision == 'fp16' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir="models/cache/",
            )

        self.model.enable_attention_slicing()

        # Загружаем пользовательский VAE, если указан
        vae = None
        if vae_name:
            vae_path = os.path.join(self.config.vae_path, vae_name)
            vae_path = os.path.normpath(vae_path)
            if os.path.exists(vae_path):
                torch_dtype = torch.float16 if self.config.default_precision == 'fp16' else torch.float32
                if vae_path.endswith('.safetensors'):
                    print(f"Загрузка VAE (файл): {vae_path}")
                    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch_dtype)
                else:
                    print(f"Загрузка VAE (онлайн): {vae_path}")
                    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch_dtype)

                self.model.vae = vae
                print(f"Пользовательский VAE загружен: {vae_name}")
        else:
            print("Используется встроенный VAE.")

        self.model.to(self.device)

        if self.config.use_xformers:
            self.model.enable_xformers_memory_efficient_attention()

        # Настройка планировщика
        self.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config)
        self.model.scheduler = self.scheduler

        if torch.cuda.is_available():
            self.compiled_model = torch.compile(self.model)
        else:
            self.compiled_model = self.model

        print(f"Модель {model_name} успешно загружена (Тип: {model_type}).")

    async def load_lora(self, lora_name: str, alpha: float = 0.75):
        lora_path = os.path.join(self.config.lora_path, lora_name)
        lora_path = os.path.normpath(lora_path)
        if os.path.exists(lora_path):
            self.model.load_lora_weights(lora_path)
            self.model.fuse_lora(alpha=alpha)

    async def generate_image(self, params: Dict[str, Any], progress_callback=None):
        if not self.model:
            raise ValueError("Model not loaded")

        width, height = map(int, params.get('size', '512x768').split('x'))
        start_time = time.time()

        # Ensure prompt and negative_prompt are strings
        prompt = str(params.get('prompt', 'masterpiece, best quality, 1girl, beautiful, dynamic pose'))
        negative_prompt = str(params.get('negative_prompt', '(worst quality:1.2), (low quality:1.2), (lowres:1.1), (monochrome:1.1), (greyscale), multiple views, comic, sketch, missing fingers'))

        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=True):
            result = self.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(params.get('steps', 24)),
                guidance_scale=float(params.get('cfg_scale', 7.0)),
                width=width,
                height=height,
                generator=torch.manual_seed(params['seed']) if params.get('seed') is not None else None,
            )

        torch.cuda.empty_cache()
        return result.images[0]

    def unload_model(self):
        if self.model:
            del self.model
            del self.scheduler
            del self.vae
            torch.cuda.empty_cache()
            self.model = None
            self.scheduler = None
            self.vae = None

