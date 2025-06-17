# GraFIQs Strict Mode Implementation Summary

## Overview
The GraFIQs pipeline has been enhanced to ensure exactly the specified number of gradients are produced when both `--num-identities` and `--samples-per-identity` are specified.

## Key Features

### 1. Strict Sample Count Validation
- **Requirement**: If 100 identities × 100 samples is chosen, exactly 10,000 gradients must be produced
- **Implementation**: Only identities with at least the required number of samples are considered
- **Behavior**: If insufficient identities are available, the script aborts before gradient calculation

### 2. Identity Filtering Logic
```python
# Example: Need 100 identities with ≥50 samples each
valid_candidates = []
insufficient_candidates = []

for identity in all_identities:
    sample_count = count_samples(identity)
    if sample_count >= required_samples_per_identity:
        valid_candidates.append(identity)
    else:
        insufficient_candidates.append(identity)

# Only proceed if enough valid identities exist
if len(valid_candidates) < required_identities:
    print("ERROR: Not enough identities with sufficient samples!")
    exit()
```

### 3. Exact Sample Collection
- From each selected identity, exactly `samples_per_identity` samples are randomly selected
- Total samples collected = `num_identities × samples_per_identity`
- Final validation ensures exact count before gradient calculation

### 4. Error Handling and Reporting
When insufficient identities are found:
- Clear error message explaining the problem
- List of identities with insufficient samples
- Suggestions for resolution (reduce requirements or add more samples)
- Script aborts without wasting computation time

### 5. Reproducible Selection
- `--random-seed`: Controls sample selection within identities
- `--identity-selection-seed`: Controls which identities are chosen
- `--random-identity-selection`: Enables random vs. deterministic identity selection

## Usage Examples

### Strict Mode (Guaranteed Exact Count)
```bash
python extract_grafiqs.py \
    --image-path /path/to/dataset \
    --output-dir ./gradients \
    --backbone iresnet50 \
    --weights /path/to/weights.pth \
    --num-identities 100 \
    --samples-per-identity 50 \
    --random-seed 42
```
**Result**: Exactly 5,000 gradients or script aborts if impossible

### Large Scale Processing
```bash
python extract_grafiqs.py \
    --image-path /path/to/large_dataset \
    --output-dir ./gradients_large \
    --backbone iresnet100 \
    --weights /path/to/weights.pth \
    --num-identities 500 \
    --samples-per-identity 100 \
    --random-identity-selection \
    --identity-selection-seed 123 \
    --random-seed 456
```
**Result**: Exactly 50,000 gradients or abort if insufficient identities

## Validation Flow

1. **Dataset Scan**: Identify all identities and count their samples
2. **Filtering**: Only keep identities with ≥ required samples
3. **Availability Check**: Ensure enough valid identities exist
4. **Selection**: Choose required number of identities (random or deterministic)
5. **Sample Collection**: Extract exactly the specified number of samples from each
6. **Final Validation**: Verify exact total count before processing
7. **Gradient Extraction**: Process exactly the required number of samples

## Benefits

- **Predictable Output**: Always know exactly how many gradients will be produced
- **Quality Assurance**: Only use identities with sufficient samples
- **Resource Planning**: No wasted computation on incomplete datasets
- **Reproducibility**: Consistent results with seed control
- **Early Detection**: Identify data quality issues before expensive processing

## Backward Compatibility

- If only one of `--num-identities` or `--samples-per-identity` is specified, flexible mode is used
- If neither is specified, all available samples are processed
- Existing workflows continue to work unchanged
