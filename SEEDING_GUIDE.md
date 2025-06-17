# GraFIQs Seeding Options

## Overview

The enhanced GraFIQs system now supports sophisticated seeding control for reproducible dataset subset selection. You can control seeds separately for:

1. **Identity Selection**: Which identities to include in the subset
2. **Sample Selection**: Which samples to select within each identity
3. **Model Operations**: PyTorch/NumPy random operations

## Command Line Arguments

### Core Seeding Arguments

- `--random-seed`: Controls sample selection within identities and general random operations (default: 42)
- `--identity-selection-seed`: Controls which identities are selected (default: uses `--random-seed`)
- `--random-identity-selection`: Enable random identity selection instead of deterministic first-N selection

### Subset Control Arguments

- `--num-identities`: Maximum number of identities to process
- `--samples-per-identity`: Maximum number of samples per identity
- `--allow-insufficient-samples`: Include identities with fewer samples than specified

## Usage Examples

### 1. Fully Reproducible with Same Seed
```bash
python extract_grafiqs.py \
    --image-path /path/to/dataset \
    --num-identities 100 \
    --samples-per-identity 50 \
    --random-seed 42 \
    --backbone iresnet50 \
    --weights weights.pth
```
- Uses seed 42 for both identity and sample selection
- Deterministic: takes first 100 identities alphabetically
- Results: Fully reproducible across runs

### 2. Random Identity Selection with Fixed Sample Selection
```bash
python extract_grafiqs.py \
    --image-path /path/to/dataset \
    --num-identities 100 \
    --samples-per-identity 50 \
    --random-identity-selection \
    --identity-selection-seed 123 \
    --random-seed 456 \
    --backbone iresnet50 \
    --weights weights.pth
```
- Uses seed 123 to randomly select 100 identities
- Uses seed 456 to select 50 samples within each identity
- Results: Different identity subset but consistent sample selection

### 3. Multiple Experiments with Different Seeds
```bash
# Experiment 1
python extract_grafiqs.py \
    --image-path /path/to/dataset \
    --output-dir ./exp1_gradients \
    --num-identities 100 \
    --samples-per-identity 50 \
    --random-identity-selection \
    --identity-selection-seed 111 \
    --random-seed 222 \
    --backbone iresnet50 \
    --weights weights.pth

# Experiment 2  
python extract_grafiqs.py \
    --image-path /path/to/dataset \
    --output-dir ./exp2_gradients \
    --num-identities 100 \
    --samples-per-identity 50 \
    --random-identity-selection \
    --identity-selection-seed 333 \
    --random-seed 444 \
    --backbone iresnet50 \
    --weights weights.pth
```
- Creates different but reproducible subsets for comparison
- Each experiment uses different identities and different samples

### 4. Quality Control with Deterministic Selection
```bash
python extract_grafiqs.py \
    --image-path /path/to/dataset \
    --num-identities 200 \
    --samples-per-identity 100 \
    --random-seed 42 \
    --backbone iresnet50 \
    --weights weights.pth
```
- Takes first 200 identities (deterministic)
- Excludes identities with <100 samples
- Uses seed 42 for sample selection within valid identities

## Dataset API Usage

```python
from dataset import create_dataset_with_subset

# Create dataset with separate seeds
dataset = create_dataset_with_subset(
    dataset_type='CasiaWebFace',
    root_dir='/path/to/dataset',
    local_rank=0,
    num_identities=100,
    samples_per_identity=50,
    selective=True,
    random_seed=456,  # For sample selection
    identity_selection_seed=123  # For identity selection
)
```

## Seeding Strategies

### Strategy 1: Complete Reproducibility
- Use same seed for all operations
- Best for: Exact reproduction of results

### Strategy 2: Identity Variation with Consistent Samples  
- Different seed for identity selection
- Same seed for sample selection
- Best for: Testing robustness across different identity sets

### Strategy 3: Sample Variation with Fixed Identities
- Deterministic identity selection (no random flag)
- Different seeds for sample selection
- Best for: Testing sample diversity within fixed identities

### Strategy 4: Full Randomization
- Different seeds for both identity and sample selection
- Best for: Creating multiple independent dataset variants

## Output Information

The system provides detailed logging:
```
Using random seed: 456 for reproducible results
Using identity selection seed: 123
Randomly selected 100 identities from 500 available (seed: 123)
Found 15 identities with insufficient samples:
  - person_001: 23 samples
  - person_045: 38 samples
Processing 85 identities that meet the sample requirement.
Dataset loaded: 85 identities, 4250 total images
```

## Best Practices

1. **Document Seeds**: Always record the seeds used for reproducibility
2. **Separate Concerns**: Use different seeds for identity vs sample selection when testing different aspects
3. **Validate Quality**: Check the insufficient samples report to ensure dataset quality
4. **Multiple Runs**: Use different seeds to create multiple independent subsets for robust evaluation

## Folder Structure Maintained

Regardless of seeding strategy, the gradient output maintains the original folder structure:
```
output_folder/
├── image/
│   ├── identity_001/
│   │   ├── sample_001_grad.npy
│   │   └── sample_002_grad.npy
│   └── identity_002/
│       ├── sample_001_grad.npy
│       └── sample_002_grad.npy
├── block1/
│   └── ... (same structure)
└── ... (block2, block3, block4)
```

This ensures that gradients can always be traced back to their source images, regardless of the subset selection strategy used.
