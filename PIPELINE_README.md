# EEG Data Curation Pipeline with Dropout Sweeps

This pipeline orchestrates preprocessing, training, and evaluation of EEG models with different dropout rates for data curation experiments.

## Key Features

- **Unified Pipeline**: Single script that handles preprocessing and training
- **Hydra Multi-Run**: Automatic sweeping over different dropout rates
- **Separate Caching**: Each dropout configuration gets its own cache directory
- **Consistent Configuration**: Dropout parameters flow from preprocessing through training

## Quick Start

### Option 1: Run Complete Pipeline with Sweep

Run preprocessing and training for multiple dropout rates automatically:

```bash
# Local testing with 3 dropout rates (0.3, 0.5, 0.7)
python pipeline_main.py --config-name=exp_1_eegpt_workload_dropout_sweep_local

# Full sweep with 5 dropout rates (0.1, 0.3, 0.5, 0.7, 0.9)
python pipeline_main.py --config-name=exp_1_eegpt_workload_dropout_sweep
```

### Option 2: Run Pipeline with Single Dropout Rate

Override the dropout rate from command line:

```bash
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'
```

### Option 3: Run Preprocessing and Training Separately

If you want more control, run each step independently:

```bash
# Step 1: Preprocessing only
python preproc.py \
  --config-name=exp_1_eegpt_workload_random_dropout_local \
  dropout_rate=0.5

# Step 2: Training only (uses cached preprocessed data)
python baseline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'
```

## Configuration Files

### Pipeline Configs

- `exp_1_eegpt_workload_dropout_sweep.yaml`: Full sweep (5 dropout rates) for cluster
- `exp_1_eegpt_workload_dropout_sweep_local.yaml`: Local testing (3 dropout rates)

### Key Parameters

```yaml
# Dropout configuration
random_dropout: true
dropout_rate: 0.5      # Data dropout rate (0.0 to 1.0)
dropout_seed: 12       # Random seed for reproducibility

# Preprocessing
clean_middle_cache: false  # Set true to force reprocessing

# Model training
training:
  max_epochs: 100
  freeze_encoder: true
```

## How It Works

### 1. Preprocessing Step

For each dropout rate, the pipeline:
- Creates a `PreprocArgs` object with dropout parameters
- Passes it to the dataset builder
- Builder creates cache with unique name: `finetune_dropout_0.5_seed_12`
- Each dropout configuration gets separate cache directories

### 2. Training Step

During training:
- Dropout parameters are passed through `BaseDataArgs` config
- DataLoader factory extracts dropout params from config
- `load_concat_eeg_datasets` creates builder with matching dropout parameters
- Loads from correct cached dataset (e.g., `finetune_dropout_0.5_seed_12`)

### 3. Cache Directory Structure

```
data/
  cache/
    workload/
      finetune/              # No dropout
      finetune_dropout_0.3_seed_12/
      finetune_dropout_0.5_seed_12/
      finetune_dropout_0.7_seed_12/
      ...
```

## Output Structure

```
outputs/
  exp_1_eegpt_workload_dropout_sweep_local/
    dropout_0.3/
      logs/
      checkpoints/
    dropout_0.5/
      logs/
      checkpoints/
    dropout_0.7/
      logs/
      checkpoints/
```

## Customization

### Add More Dropout Rates

Edit the sweep config:

```yaml
hydra:
  sweeper:
    params:
      dropout_rate: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

### Change Other Parameters

Sweep multiple parameters simultaneously:

```yaml
hydra:
  sweeper:
    params:
      dropout_rate: 0.3,0.5,0.7
      dropout_seed: 12,42,123
```

### Disable Sweeping

Run with a single configuration:

```bash
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'
```

## Troubleshooting

### Cache Conflicts

If you see `NonMatchingSplitsSizesError`, it means the cache has stale data:

```bash
# Clean all caches
rm -rf data/cache/workload/finetune_dropout_*

# Or enable auto-clean in config
clean_middle_cache: true
```

### Memory Issues

Reduce batch size or workers:

```bash
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  data.batch_size=16 \
  data.num_workers=0
```

### Check Available Models

```bash
python baseline_main.py list-models
```

## Advanced Usage

### SLURM Submission

```bash
sbatch --export=CONFIG=exp_1_eegpt_workload_dropout_sweep slurm/pipeline_submit.slurm
```

### Custom Dropout Implementation

The dropout is applied during preprocessing in the dataset builder. To modify the dropout logic, edit:
- `data/processor/builder.py`: Builder initialization and config name modification
- `data/dataset/<dataset_name>.py`: Dataset-specific preprocessing

## Notes

- Evaluation happens automatically at the end of each epoch during training
- WandB logging is enabled by default (can disable with `logging.use_cloud=false`)
- Each dropout rate creates completely independent cached datasets
- Preprocessing is idempotent - safe to run multiple times with same parameters
