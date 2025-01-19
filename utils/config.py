import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    @property
    def allowed_user_id(self) -> int:
        return self.config['allowed_user_id']

    @property
    def models_path(self) -> str:
        return self.config['models_path']

    @property
    def lora_path(self) -> str:
        return self.config['lora_path']

    @property
    def vae_path(self) -> str:
        return self.config['vae_path']

    @property
    def output_path(self) -> str:
        return self.config['output_path']

    @property
    def log_path(self) -> str:
        return self.config['log_path']

    @property
    def default_model(self) -> str:
        return self.config['default_model']

    @property
    def default_model_type(self) -> str:
        return self.config['default_model_type']
    
    @property
    def default_settings(self) -> str:
        return {
            'model': self.config['default_model'],
            'vae': 'default',
            'lora': 'None',
            'sampler': 'Euler a',
            'cfg_scale': 7.0,
            'steps': 24,
            'size': '512x768',
            'prompt': 'masterpiece, best quality, 1girl',
            'negative_prompt': 'lowres, text, jpeg artifacts, ugly, (worst quality, low quality, bad quality), (blurry), missing fingers, extra fingers, extra legs, extra hands'
        }

    @property
    def default_precision(self) -> str:
        return self.config['default_precision']

    @property
    def use_xformers(self) -> bool:
        return self.config['use_xformers']