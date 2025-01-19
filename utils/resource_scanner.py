import os
from typing import List, Dict

class ResourceScanner:
    def __init__(self, config):
        self.config = config

    def scan_models(self) -> List[str]:
        return self._scan_directory(self.config.models_path)

    def scan_loras(self) -> List[str]:
        return self._scan_directory(self.config.lora_path)

    def scan_vaes(self) -> List[str]:
        return self._scan_directory(self.config.vae_path)
    
    def scan_samplers(self):
        samplers = ['Euler a', 'DPM++ SDE', 'DPM++ 2S a Karras', 'DPM++ 2M Karras', 'DPM++ SDE Karras', 'UniPC']
        return samplers

    def _scan_directory(self, directory: str) -> List[str]:
        if not os.path.exists(directory):
            return []
        return [f for f in os.listdir(directory) if f.endswith('.safetensors') or f.endswith('.ckpt')]

    def get_available_resources(self) -> Dict[str, List[str]]:
        return {
            'models': self.scan_models(),
            'lora': self.scan_loras(),
            'vae': self.scan_vaes()
        }

