#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import hydra

from baseline.abstract.factory import ModelRegistry
from common.log import setup_log
from common.path import get_conf_file_path
from common.utils import setup_yaml


logger = logging.getLogger('baseline')


@hydra.main(config_path="hydra_configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main training function that can handle any registered baseline model."""
    setup_yaml()
    setup_log()
    
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Get model type from config
    model_type: str = cfg.get('model_type', None)

    # Validate model type
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
    # Create base config for the specified model type
    config_class = ModelRegistry.get_config_class(model_type)
    
    # Convert OmegaConf to dict and validate with config class
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config = config_class.model_validate(cfg_dict)
    
    # Validate configuration
    if not config.validate_config():
        raise ValueError(f"Invalid configuration for model type: {model_type}")
    
    # Setup output directory
    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and run trainer
    trainer = ModelRegistry.create_trainer(config)
    trainer.run()


def list_available_models():
    """List all available model types."""
    print("Available baseline models:")
    for model_type in ModelRegistry.list_models():
        print(f"  - {model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        main()