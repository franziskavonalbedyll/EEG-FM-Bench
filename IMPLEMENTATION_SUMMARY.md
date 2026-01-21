# Implementation Summary: Dropout-Aware Pipeline

## Overview

Successfully implemented a complete Hydra-based pipeline that preprocesses EEG datasets with different dropout rates and then trains/evaluates models on each version. The key achievement is **automatic cache separation** based on dropout parameters.

## Files Modified

### 1. `common/config.py`
**Change**: Added dropout parameters to `BaseDataArgs`
```python
class BaseDataArgs(BaseModel):
    # ... existing fields ...
    random_dropout: bool = False
    dropout_rate: float = 0.0
    dropout_seed: int = 12
```
**Why**: Allows dropout parameters to flow through the entire config chain from top-level config → trainer → dataloader → dataset builder.

### 2. `data/processor/wrapper.py`
**Change**: Updated `load_concat_eeg_datasets()` to accept and use dropout parameters
```python
def load_concat_eeg_datasets(
    # ... existing params ...
    random_dropout: bool = False,
    dropout_rate: float = 0.0,
    dropout_seed: int = 12,
):
    preproc_args = PreprocArgs(
        random_dropout=random_dropout,
        dropout_rate=dropout_rate,
        dropout_seed=dropout_seed
    )
    builder = builder_cls(config_name=ds_config, preproc_args=preproc_args)
```
**Why**: Creates builders with correct dropout config, enabling them to generate unique cache names.

### 3. `baseline/abstract/adapter.py`
**Changes**:
- `loading_dataset()`: Added dropout params
- `create_dataloader()`: Added dropout params and passes them through

**Why**: Bridges the gap between config and dataset loading, extracting dropout params from config and passing to `load_concat_eeg_datasets()`.

### 4. `baseline/abstract/trainer.py`
**Changes**:
- `create_dataloader()`: Extracts dropout params from `self.cfg.data` and passes to factory
- `create_single_dataloader()`: Same as above

**Why**: Ensures trainer uses dropout config when creating dataloaders.

### 5. `baseline_main.py`
**Change**: Converted from manual config loading to Hydra
```python
@hydra.main(config_path="hydra_configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Now uses Hydra like preproc.py
```
**Why**: Enables Hydra multi-run sweeps for training, consistent with preprocessing.

## Files Created

### 6. `pipeline_main.py` (NEW)
**Purpose**: Unified script that runs both preprocessing and training
**Features**:
- Single entry point for complete pipeline
- Supports Hydra multi-run
- Automatically passes dropout config to both stages
- Clear logging for each pipeline stage

### 7. `hydra_configs/exp_1_eegpt_workload_dropout_sweep.yaml` (NEW)
**Purpose**: Production sweep configuration
**Features**:
- Sweeps 5 dropout rates: 0.1, 0.3, 0.5, 0.7, 0.9
- Full training (100 epochs)
- WandB logging enabled
- Separate output dirs per dropout rate

### 8. `hydra_configs/exp_1_eegpt_workload_dropout_sweep_local.yaml` (NEW)
**Purpose**: Local testing sweep configuration
**Features**:
- Sweeps 3 dropout rates: 0.3, 0.5, 0.7
- Reduced training (50 epochs)
- WandB logging disabled
- Optimized for quick local testing

### 9. `PIPELINE_README.md` (NEW)
**Purpose**: Comprehensive documentation
**Includes**:
- Quick start commands
- Configuration options
- Troubleshooting guide
- Advanced usage examples

## How It Works End-to-End

### Configuration Flow

```
Hydra Config (dropout_rate=0.5)
    ↓
pipeline_main.py receives cfg
    ↓
┌─────────────────────────────────────┐
│ PREPROCESSING STAGE                  │
│ PreprocArgs(dropout_rate=0.5) →     │
│ builder.__init__(preproc_args) →    │
│ conf.name = "finetune_dropout_0.5"  │
│ Cache: data/cache/.../finetune_dropout_0.5_seed_12/ │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ TRAINING STAGE                       │
│ trainer.cfg.data.dropout_rate=0.5 → │
│ create_dataloader(dropout_rate=0.5)→ │
│ load_concat_eeg_datasets(dropout_rate=0.5) → │
│ builder = cls(config="finetune", preproc_args) → │
│ builder.__init__ modifies conf.name → │
│ Loads: data/cache/.../finetune_dropout_0.5_seed_12/ │
└─────────────────────────────────────┘
```

### Cache Naming Logic

In `data/processor/builder.py.__init__()`:
```python
if preproc_args is not None:
    conf.random_dropout = preproc_args.random_dropout
    conf.dropout_rate = preproc_args.dropout_rate
    conf.dropout_seed = preproc_args.dropout_seed
    
    if conf.random_dropout and conf.dropout_rate > 0.0:
        conf.name = f"{config_name}_dropout_{conf.dropout_rate}_seed_{conf.dropout_seed}"
```

Result: Each dropout configuration gets isolated cache:
- `finetune` (no dropout)
- `finetune_dropout_0.3_seed_12`
- `finetune_dropout_0.5_seed_12`
- etc.

## Usage Examples

### Run Full Pipeline with Sweep
```bash
python pipeline_main.py --config-name=exp_1_eegpt_workload_dropout_sweep_local
```

### Run Single Dropout Rate
```bash
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'
```

### Run Only Preprocessing
```bash
python preproc.py \
  --config-name=exp_1_eegpt_workload_random_dropout_local \
  dropout_rate=0.5
```

### Run Only Training (using cached data)
```bash
python baseline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'
```

## Key Benefits

1. **Automatic Cache Separation**: No manual cache management needed
2. **Consistency**: Same dropout config used in preprocessing and training
3. **Reproducibility**: Dropout seed ensures consistent results
4. **Flexibility**: Run full pipeline or individual stages
5. **Scalability**: Easy to add more dropout rates or sweep other parameters
6. **Clean Hydra Integration**: Works seamlessly with Hydra's multi-run

## Testing Checklist

- [ ] Run preprocessing with dropout_rate=0.5
- [ ] Verify cache created at `data/cache/workload/finetune_dropout_0.5_seed_12/`
- [ ] Run training with same dropout_rate
- [ ] Verify training loads correct cached dataset
- [ ] Run pipeline_main.py with single dropout rate
- [ ] Run pipeline_main.py with multi-run sweep
- [ ] Verify separate output directories created per dropout rate

## Next Steps

1. Test the pipeline with a small dataset first
2. Verify cache directories are created correctly
3. Run a local sweep with 2-3 dropout rates
4. Monitor WandB to ensure runs are properly tagged
5. Submit to cluster for full sweep
