#!/usr/bin/env python3
"""
Unified pipeline for EEG data curation and model training.

This script orchestrates:
1. Data preprocessing with dropout augmentation
2. Model finetuning
3. Evaluation

Supports Hydra multi-run for sweeping over different dropout rates.
"""

import logging


from omegaconf import DictConfig, OmegaConf
import hydra

from common.config import PreprocArgs
from common.log import setup_log
from preproc import preproc

from baseline_main import run_training


logger = logging.getLogger('pipeline')


@hydra.main(config_path="hydra_configs", version_base=None)
def main(cfg: DictConfig):
    """Main pipeline entry point."""
    setup_log()
    
    logger.info("="*80)
    logger.info("EEG DATA CURATION AND FINETUNING PIPELINE")
    logger.info("="*80)
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create PreprocArgs for preprocessing step
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    preproc_conf = PreprocArgs.model_validate(cfg_dict)
    
    # Extract experiment config as plain dicts (not OmegaConf) for consistency downstream
    experiment_config = cfg_dict.get('experiment', None)
    exp_name = experiment_config.get("name", "") if experiment_config else None
    preproc_exp_conf = experiment_config.get('preproc', {}) if experiment_config else {}
    training_exp_conf = experiment_config.get('training', {}) if experiment_config else {}
    
    # Run preprocessing
    preproc(preproc_conf, exp_name=exp_name, exp_config=preproc_exp_conf)
    
    # Run training (pass preproc_exp_conf so dataloader can find the correct preprocessed dataset)
    run_training(cfg, 
                 exp_name=exp_name, 
                 preproc_exp_config=preproc_exp_conf
                 )
    
    logger.info("="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
