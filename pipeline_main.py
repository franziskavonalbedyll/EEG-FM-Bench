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
from pathlib import Path
from typing import Type

from omegaconf import DictConfig, OmegaConf
import hydra

from common.config import PreprocArgs
from common.log import setup_log
from data.processor.builder import EEGDatasetBuilder
from data.processor.wrapper import DATASET_SELECTOR
from baseline.abstract.factory import ModelRegistry


logger = logging.getLogger('pipeline')


def run_preprocessing(
        conf: PreprocArgs,
        builder_cls: Type[EEGDatasetBuilder],
        dataset_name: str,
        config_name: str
):
    """Run preprocessing step for a single dataset."""
    logger.info(f"[PREPROC] Preparing dataset {dataset_name} {config_name}...")
    logger.info(f"[PREPROC] Dropout config: random_dropout={conf.random_dropout}, "
                f"dropout_rate={conf.dropout_rate}, dropout_seed={conf.dropout_seed}")
    
    builder = builder_cls(config_name, preproc_args=conf)
    
    if conf.clean_middle_cache:
        logger.info(f"[PREPROC] Cleaning middle cache...")
        builder.clean_disk_cache()
    
    builder.preproc(n_proc=conf.num_preproc_mid_workers)
    builder.download_and_prepare(num_proc=conf.num_preproc_arrow_writers)
    dataset = builder.as_dataset()
    
    logger.info(f"[PREPROC] Dataset {dataset_name} {config_name} is prepared.")
    logger.info(f"[PREPROC] {dataset}")


def run_preprocessing_all(preproc_conf: PreprocArgs):
    """Run preprocessing for all configured datasets."""
    logger.info("="*80)
    logger.info("STEP 1: PREPROCESSING FINETUNING DATASETS")
    logger.info("="*80)
    
    # Only process finetuning datasets (no pretraining)
    dataset_names = list(preproc_conf.finetune_datasets.keys())
    dataset_configs = list(preproc_conf.finetune_datasets.values())

    for dataset, config in zip(dataset_names, dataset_configs):
        if dataset not in DATASET_SELECTOR.keys():
            raise ValueError(f"Dataset {dataset} is not supported.")

        builder_cls = DATASET_SELECTOR[dataset]
        if config not in builder_cls.builder_configs.keys():
            raise ValueError(f"Config {config} is not supported for dataset {dataset}.")

        run_preprocessing(preproc_conf, builder_cls, dataset, config)
    
    logger.info("[PREPROC] All datasets preprocessed successfully!")


def run_training(cfg: DictConfig):
    """Run model finetuning step."""
    logger.info("="*80)
    logger.info("STEP 2: MODEL FINETUNING")
    logger.info("="*80)
    
    # Get model type from config
    model_type: str = cfg.get('model_type', None)
    logger.info(f"[TRAINING] Retrieved model_type: {model_type}")
    if model_type is None:
        raise ValueError("model_type must be specified in configuration")

    # Validate model type
    available_models = ModelRegistry.list_models()
    if model_type not in available_models:
        logger.error(f"[TRAINING] Retrieved model {model_type} not in available models. Available: {available_models}")
        logger.info(f"[TRAINING] Retrieved model {model_type} not in available models. Available: {available_models}")
        raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
    
    logger.info(f"[TRAINING] Model type: {model_type}")
    logger.info(f"[TRAINING] Dropout config: random_dropout={cfg.data.random_dropout}, "
                f"dropout_rate={cfg.data.dropout_rate}, dropout_seed={cfg.data.dropout_seed}")
    
    # Create config for the specified model type
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
    logger.info(f"[TRAINING] Starting training...")
    trainer = ModelRegistry.create_trainer(config)
    logger.info(f"[TRAINING] Successfully created trainer, beginning run...")
    trainer.run()
    
    logger.info("[TRAINING] Training completed successfully!")


@hydra.main(config_path="hydra_configs", config_name="config", version_base=None)
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
    
    # Run preprocessing
    run_preprocessing_all(preproc_conf)
    
    # Run training
    run_training(cfg)
    
    logger.info("="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
