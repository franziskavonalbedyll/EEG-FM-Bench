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


def run_training(cfg: DictConfig | dict, exp_name: str = None, preproc_exp_config: dict = None):
    setup_yaml()
    setup_log()

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    model_type = cfg.get("model_type")
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")

    config_class = ModelRegistry.get_config_class(model_type)
    config = config_class.model_validate(cfg)

    if not config.validate_config():
        raise ValueError(f"Invalid configuration for model type: {model_type}")

    output_dir = Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = ModelRegistry.create_trainer(config)
    # Pass experiment info for dataloader to find correct preprocessed dataset
    trainer.exp_name = exp_name
    trainer.preproc_exp_config = preproc_exp_config
    trainer.run()


def list_available_models():
    """List all available model types."""
    print("Available baseline models:")
    for model_type in ModelRegistry.list_models():
        print(f"  - {model_type}")


@hydra.main(config_path="hydra_configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        list_available_models()
    else:
        main()