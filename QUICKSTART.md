# Quick Start Guide: Dropout Sweep Pipeline

## Prerequisites

### 1. Install Dependencies

If `hydra-core` is not installed:

```bash
pip install hydra-core
# or
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python validate_config.py 'hydra.mode=RUN' dropout_rate=0.5
```

Expected output:
```
================================================================================
CONFIGURATION VALIDATION
================================================================================
...
✓ ALL TESTS PASSED
================================================================================
```

## Running Experiments

### Option 1: Full Pipeline with Sweep (Recommended)

Runs preprocessing + training for multiple dropout rates:

```bash
# Local test (3 dropout rates: 0.3, 0.5, 0.7)
python pipeline_main.py --config-name=exp_1_eegpt_workload_dropout_sweep_local

# Full experiment (5 dropout rates: 0.1, 0.3, 0.5, 0.7, 0.9)
python pipeline_main.py --config-name=exp_1_eegpt_workload_dropout_sweep
```

### Option 2: Single Dropout Rate

```bash
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'
```

### Option 3: Separate Steps

If you need to run preprocessing and training separately:

```bash
# Step 1: Preprocessing
python preproc.py \
  --config-name=exp_1_eegpt_workload_random_dropout_local \
  dropout_rate=0.5

# Step 2: Training
python baseline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'
```

## What Gets Created

### Cache Directories

Each dropout rate gets its own cache:

```
data/cache/workload/
├── finetune/                      # No dropout (baseline)
├── finetune_dropout_0.3_seed_12/  # 30% dropout
├── finetune_dropout_0.5_seed_12/  # 50% dropout
└── finetune_dropout_0.7_seed_12/  # 70% dropout
```

### Output Directories

```
outputs/exp_1_eegpt_workload_dropout_sweep_local/
├── dropout_0.3/
│   ├── .hydra/
│   └── logs/
├── dropout_0.5/
│   ├── .hydra/
│   └── logs/
└── dropout_0.7/
    ├── .hydra/
    └── logs/
```

## Configuration Files

- `exp_1_eegpt_workload_dropout_sweep.yaml` - Full sweep for cluster (5 rates)
- `exp_1_eegpt_workload_dropout_sweep_local.yaml` - Local testing (3 rates)
- `exp_1_eegpt_workload_random_dropout_local.yaml` - Single rate (for preproc only)

## Troubleshooting

### Cache Conflict Error

```
datasets.exceptions.NonMatchingSplitsSizesError
```

**Solution**: Clean stale caches
```bash
rm -rf data/cache/workload/finetune_dropout_*
# or set in config:
clean_middle_cache: true
```

### Module Not Found: hydra

**Solution**: Install hydra-core
```bash
pip install hydra-core
```

### Wrong Dataset Loaded

Check that dropout parameters match:
```bash
python validate_config.py 'hydra.mode=RUN' dropout_rate=0.5
```

## Advanced Usage

### Add More Dropout Rates

Edit `hydra_configs/exp_1_eegpt_workload_dropout_sweep_local.yaml`:

```yaml
hydra:
  sweeper:
    params:
      dropout_rate: 0.1,0.2,0.3,0.4,0.5
```

### Override Any Parameter

```bash
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.6 \
  training.max_epochs=20 \
  data.batch_size=16
```

### Check Available Models

```bash
python baseline_main.py list-models
```

## Monitoring

### WandB

Local configs have WandB disabled. To enable:

```bash
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  logging.use_cloud=true \
  logging.offline=false
```

### Logs

Real-time logs are printed to console. Each run also saves logs to:
```
outputs/<experiment_name>/dropout_<rate>/.hydra/
```

## Files Overview

| File | Purpose |
|------|---------|
| `pipeline_main.py` | Unified preprocessing + training pipeline |
| `preproc.py` | Preprocessing only |
| `baseline_main.py` | Training only |
| `validate_config.py` | Configuration validation |
| `PIPELINE_README.md` | Detailed documentation |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation details |

## Example Workflow

```bash
# 1. Validate configuration
python validate_config.py 'hydra.mode=RUN' dropout_rate=0.5

# 2. Test with single rate
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local \
  dropout_rate=0.5 \
  'hydra.mode=RUN'

# 3. Run full local sweep
python pipeline_main.py \
  --config-name=exp_1_eegpt_workload_dropout_sweep_local

# 4. Submit to cluster
sbatch slurm/pipeline_submit.slurm
```

## Getting Help

1. Read `PIPELINE_README.md` for comprehensive documentation
2. Read `IMPLEMENTATION_SUMMARY.md` for technical details
3. Run validation: `python validate_config.py`
4. Check logs in `outputs/<experiment_name>/dropout_<rate>/`

## Key Points to Remember

✓ Each dropout rate gets **separate cache directories**  
✓ Cache names include dropout rate: `finetune_dropout_0.5_seed_12`  
✓ Preprocessing and training must use **same dropout parameters**  
✓ Use `validate_config.py` to verify configuration  
✓ Local configs have reduced epochs and disabled WandB  
✓ Use `'hydra.mode=RUN'` to disable sweeping for single runs  
