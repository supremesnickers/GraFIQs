#!/usr/bin/env python3
"""
Example usage of GraFIQs with strict subset selection and sample validation.

This script demonstrates how to use the enhanced GraFIQs functionality 
to process an exact subset of a dataset for gradient extraction with strict 
sample count requirements.

KEY FEATURES:
- Ensures exactly (num_identities × samples_per_identity) gradients are produced
- Skips identities that don't have enough samples
- Aborts before gradient calculation if insufficient identities are available
- Provides reproducible subset selection with configurable seeds
"""

# Example command line usage with strict sample count requirements:

# 1. Process exactly 100 identities × 50 samples = 5000 gradients (strict mode)
# python extract_grafiqs.py \
#     --image-path /path/to/dataset \
#     --output-dir ./gradients_output \
#     --backbone iresnet50 \
#     --weights /path/to/weights.pth \
#     --num-identities 100 \
#     --samples-per-identity 50 \
#     --random-seed 42 \
#     --dump-gradients
#     # IMPORTANT: Only identities with ≥50 samples will be considered
#     # If fewer than 100 such identities exist, the script will abort

# 2. Randomly select identities with separate seeds for reproducible results
# python extract_grafiqs.py \
#     --image-path /path/to/dataset \
#     --output-dir ./gradients_output \
#     --backbone iresnet50 \
#     --weights /path/to/weights.pth \
#     --num-identities 100 \
#     --samples-per-identity 50 \
#     --random-identity-selection \
#     --identity-selection-seed 123 \
#     --random-seed 456 \
#     --dump-gradients
#     # Result: Exactly 5000 gradients from randomly selected identities

# 3. Large scale processing: 500 identities × 100 samples = 50,000 gradients
# python extract_grafiqs.py \
#     --image-path /path/to/large_dataset \
#     --output-dir ./gradients_large_scale \
#     --backbone iresnet100 \
#     --weights /path/to/weights.pth \
#     --num-identities 500 \
#     --samples-per-identity 100 \
#     --random-identity-selection \
#     --identity-selection-seed 789 \
#     --random-seed 789 \
#     --dump-gradients
#     # Will abort if fewer than 500 identities have ≥100 samples

# 4. Small scale testing: 10 identities × 20 samples = 200 gradients
# python extract_grafiqs.py \
#     --image-path /path/to/dataset \
#     --output-dir ./gradients_output \
#     --backbone iresnet50 \
#     --weights /path/to/weights.pth \
#     --num-identities 200 \
#     --samples-per-identity 100 \
#     --random-seed 42 \
#     --dump-gradients
#     # This will take the first 200 identities and exclude those with insufficient samples

# 4. Use with dataset loader
from dataset import create_dataset_with_subset, DataLoaderX

def example_dataset_usage():
    """Example of how to use the dataset with subset selection and custom seeds."""
    
    # Create a dataset with 100 identities, 50 samples per identity
    # Using separate seeds for identity selection and sample selection
    dataset = create_dataset_with_subset(
        dataset_type='CasiaWebFace',
        root_dir='/path/to/casia_webface',
        local_rank=0,
        num_identities=100,
        samples_per_identity=50,
        selective=True,  # Use selective scanning (sorted by file count)
        random_seed=456,  # Seed for sample selection within identities
        identity_selection_seed=123,  # Seed for identity selection
        strict_sample_count=False  # Allow identities with fewer samples
    )
    
    # Create dataloader
    dataloader = DataLoaderX(
        local_rank=0,
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Process batches
    identity_sample_counts = {}
    for batch_idx, (data, labels, indices, paths) in enumerate(dataloader):
        # data: batch of images/embeddings
        # labels: corresponding class labels
        # indices: dataset indices  
        # paths: full file paths for gradient dumping
        
        # Track samples per identity for validation
        for label in labels:
            label_int = label.item()
            identity_sample_counts[label_int] = identity_sample_counts.get(label_int, 0) + 1
        
        print(f"Batch {batch_idx}: {data.shape}, {len(paths)} paths")
        if batch_idx >= 2:  # Just show first few batches
            break
    
    # Report sample distribution
    print("\nSample distribution:")
    for identity, count in sorted(identity_sample_counts.items())[:10]:  # Show first 10
        print(f"  Identity {identity}: {count} samples")

def validate_dataset_quality(dataset_path, min_samples_per_identity=50):
    """
    Validate dataset quality by checking sample counts per identity.
    
    Args:
        dataset_path (str): Path to dataset directory
        min_samples_per_identity (int): Minimum required samples per identity
    
    Returns:
        dict: Statistics about the dataset
    """
    import os
    
    stats = {
        'total_identities': 0,
        'sufficient_identities': 0,
        'insufficient_identities': [],
        'total_samples': 0
    }
    
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    stats['total_identities'] = len(subdirs)
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        images = [f for f in os.listdir(subdir_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        sample_count = len(images)
        stats['total_samples'] += sample_count
        
        if sample_count >= min_samples_per_identity:
            stats['sufficient_identities'] += 1
        else:
            stats['insufficient_identities'].append((subdir, sample_count))
    
    # Print report
    print(f"\nDataset Quality Report for: {dataset_path}")
    print(f"Total identities: {stats['total_identities']}")
    print(f"Identities with ≥{min_samples_per_identity} samples: {stats['sufficient_identities']}")
    print(f"Identities with <{min_samples_per_identity} samples: {len(stats['insufficient_identities'])}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Average samples per identity: {stats['total_samples'] / stats['total_identities']:.1f}")
    
    if stats['insufficient_identities']:
        print(f"\nIdentities with insufficient samples:")
        for identity, count in stats['insufficient_identities'][:10]:  # Show first 10
            print(f"  {identity}: {count} samples")
        if len(stats['insufficient_identities']) > 10:
            print(f"  ... and {len(stats['insufficient_identities']) - 10} more")
    
    return stats

def demonstrate_seeding_strategies():
    """
    Demonstrate different seeding strategies for reproducible dataset subsets.
    """
    print("\n=== Seeding Strategies Demo ===")
    
    # Strategy 1: Same seed for both identity and sample selection
    print("\n1. Using same seed (42) for both identity and sample selection:")
    print("   --random-seed 42 --identity-selection-seed 42")
    print("   Result: Fully reproducible subset")
    
    # Strategy 2: Different seeds for identity vs sample selection
    print("\n2. Using different seeds for identity (123) vs sample (456) selection:")
    print("   --random-seed 456 --identity-selection-seed 123")
    print("   Result: Different identity subset, but same samples within each identity")
    
    # Strategy 3: Random identity selection with fixed sample selection
    print("\n3. Random identity selection with fixed sample selection:")
    print("   --random-identity-selection --identity-selection-seed 789 --random-seed 42")
    print("   Result: Random identities (seed 789), but deterministic samples (seed 42)")
    
    # Strategy 4: Deterministic identity selection with random samples
    print("\n4. Deterministic identity selection (first N) with random samples:")
    print("   --random-seed 999 (no --random-identity-selection)")
    print("   Result: First N identities, but random samples within each (seed 999)")
    
    # Example of creating datasets with different strategies
    strategies = [
        {
            'name': 'Same seed strategy',
            'random_seed': 42,
            'identity_selection_seed': 42,
            'random_identity': True
        },
        {
            'name': 'Different seeds strategy', 
            'random_seed': 456,
            'identity_selection_seed': 123,
            'random_identity': True
        },
        {
            'name': 'Deterministic identity, random samples',
            'random_seed': 999,
            'identity_selection_seed': None,  # Will use random_seed
            'random_identity': False
        }
    ]
    
    print(f"\n=== Creating datasets with different strategies ===")
    for strategy in strategies:
        print(f"\nStrategy: {strategy['name']}")
        print(f"  Random seed: {strategy['random_seed']}")
        print(f"  Identity seed: {strategy['identity_selection_seed']}")
        print(f"  Random identity selection: {strategy['random_identity']}")
        
        # This would create the actual dataset (commented out for demo)
        # dataset = create_dataset_with_subset(
        #     dataset_type='CasiaWebFace',
        #     root_dir='/path/to/dataset',
        #     local_rank=0,
        #     num_identities=50,
        #     samples_per_identity=25,
        #     random_seed=strategy['random_seed'],
        #     identity_selection_seed=strategy['identity_selection_seed']
        # )


if __name__ == "__main__":
    # Example usage
    try:
        example_dataset_usage()
    except Exception as e:
        print(f"Dataset example failed: {e}")
    
    # Demonstrate seeding strategies
    demonstrate_seeding_strategies()
    
    # Example dataset validation
    # validate_dataset_quality('/path/to/your/dataset', min_samples_per_identity=100)
    
    # Demonstrate seeding strategies
    demonstrate_seeding_strategies()
