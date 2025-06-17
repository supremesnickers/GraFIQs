from pathlib import Path
import argparse
import random
import os

import torch
import torch.autograd as autograd
import torch.nn.functional as F

from torch.nn import MSELoss
import torchvision.transforms.v2 as transforms

import numpy as np
import cv2

from tqdm import tqdm

from backbones.iresnet import iresnet100, iresnet50, iresnet18
from backbones.bn import BN_Model


def get_model(
    nn_architecture,
    rank,
    nn_weights_path,
    embedding_size = 512
):
    
    if nn_architecture == "iresnet100":
        backbone = iresnet100(num_features=embedding_size, use_se=False).to(rank)
    elif nn_architecture == "iresnet50":
        backbone = iresnet50(num_features=embedding_size, dropout=0.4, use_se=False).to(rank)
    else:
        raise ValueError("Unknown model architecture given.")
    
    backbone.load_state_dict(torch.load(nn_weights_path, map_location=torch.device(rank)))
    backbone.return_intermediate = True
    backbone.eval()
    
    backbone = BN_Model(backbone, rank)

    return backbone


def dump_gradients(grad_tensor, image_path, output_dir, layer_name, epoch=None):
    """
    Dump gradient tensors in the same folder structure as the original images.
    
    Args:
        grad_tensor: Tensor containing gradients
        image_path: Original image file path
        output_dir: Base directory to save gradients
        layer_name: Name of the layer (e.g., 'image', 'block1', etc.)
        epoch: Optional epoch number to include in filename
    """
    output_dir = Path(output_dir)
    image_path = Path(image_path)
    
    # Extract label (parent folder name) and filename
    if image_path.parent.name != output_dir.name:
        # If image is in a subfolder, use that as label
        label = image_path.parent.name
    else:
        # If images are directly in the folder, use 'default' as label
        label = 'default'
    
    filename = image_path.stem  # filename without extension
    
    # Create label directory if it doesn't exist
    label_dir = output_dir / layer_name / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Create gradient filename
    if epoch is not None:
        grad_filename = f"{filename}_grad_epoch_{epoch}.npy"
    else:
        grad_filename = f"{filename}_grad.npy"
    
    # Save gradient for this sample
    grad_path = label_dir / grad_filename
    gradient = grad_tensor.detach().cpu().numpy()
    np.save(grad_path, gradient)


def dump_gradients_with_metadata(grad_tensor, image_path, output_dir, layer_name, 
                                metadata=None, epoch=None):
    """
    Enhanced version that also saves metadata alongside gradients.
    
    Args:
        grad_tensor: Tensor containing gradients
        image_path: Original image file path
        output_dir: Base directory to save gradients
        layer_name: Name of the layer
        metadata: Dict with additional info (loss, accuracy, etc.)
        epoch: Optional epoch number
    """
    output_dir = Path(output_dir)
    image_path = Path(image_path)
    
    # Extract label (parent folder name) and filename
    if image_path.parent.name != output_dir.name:
        label = image_path.parent.name
    else:
        label = 'default'
    
    filename = image_path.stem
    
    # Create label directory
    label_dir = output_dir / layer_name / label
    label_dir.mkdir(parents=True, exist_ok=True)
    
    # Save gradient
    grad_filename = f"{filename}_grad.npz"

    grad_path = label_dir / grad_filename
    gradient = grad_tensor.detach().cpu().numpy()
    
    # Save with metadata
    save_dict = {'gradient': gradient}
    if metadata:
        save_dict.update(metadata)
        
    np.savez(grad_path, **save_dict)


def main(args):
    print(args)
      # Use the specified seed for all random operations
    seed = args.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Set identity selection seed (defaults to main seed if not specified)
    if args.identity_selection_seed is None:
        args.identity_selection_seed = seed
    
    print(f"Using random seed: {seed} for reproducible results")
    print(f"Using identity selection seed: {args.identity_selection_seed}")
    
    rank = torch.device(f"cuda:{args.gpu}")
      # Handle both single directory and dataset-style folder structure
    if os.path.isdir(args.image_path):
        # Check if it's a dataset-style structure (contains subdirectories)
        subdirs = [d for d in os.listdir(args.image_path) if os.path.isdir(os.path.join(args.image_path, d))]
        if subdirs and not args.flat_structure:
            # Dataset-style structure: collect images
            images = []
            
            # If both num_identities and samples_per_identity are specified, enforce strict requirements
            if args.num_identities and args.samples_per_identity:
                required_total_samples = args.num_identities * args.samples_per_identity
                print(f"Strict mode: Requiring exactly {args.num_identities} identities × {args.samples_per_identity} samples = {required_total_samples} total samples")
                
                # Find all identities that have at least the required number of samples
                valid_candidates = []
                insufficient_candidates = []
                
                for subdir in subdirs:
                    subdir_path = os.path.join(args.image_path, subdir)
                    subdir_images = [f for f in os.listdir(subdir_path) 
                                   if f.lower().endswith(args.image_extension.lower())]
                    
                    if len(subdir_images) >= args.samples_per_identity:
                        valid_candidates.append((subdir, len(subdir_images)))
                    else:
                        insufficient_candidates.append((subdir, len(subdir_images)))
                
                print(f"Found {len(valid_candidates)} identities with ≥{args.samples_per_identity} samples")
                print(f"Found {len(insufficient_candidates)} identities with <{args.samples_per_identity} samples")
                
                # Check if we have enough valid identities
                if len(valid_candidates) < args.num_identities:
                    print(f"\nERROR: Not enough identities with sufficient samples!")
                    print(f"Required: {args.num_identities} identities with ≥{args.samples_per_identity} samples each")
                    print(f"Available: {len(valid_candidates)} identities with sufficient samples")
                    print(f"Missing: {args.num_identities - len(valid_candidates)} identities")
                    
                    if insufficient_candidates:
                        print(f"\nIdentities with insufficient samples:")
                        for identity, count in insufficient_candidates[:10]:  # Show first 10
                            print(f"  - {identity}: {count} samples (need {args.samples_per_identity - count} more)")
                        if len(insufficient_candidates) > 10:
                            print(f"  ... and {len(insufficient_candidates) - 10} more")
                    
                    print(f"\nCannot proceed with gradient calculations.")
                    print(f"Please either:")
                    print(f"  1. Reduce --num-identities to {len(valid_candidates)} or less")
                    print(f"  2. Reduce --samples-per-identity to allow more identities")
                    print(f"  3. Add more samples to existing identities")
                    print(f"Exiting.")
                    return
                
                # Select the required number of valid identities
                if args.random_identity_selection:
                    # Randomly select from valid candidates
                    random.seed(args.identity_selection_seed)
                    selected_candidates = random.sample(valid_candidates, args.num_identities)
                    print(f"Randomly selected {args.num_identities} identities from {len(valid_candidates)} valid candidates (seed: {args.identity_selection_seed})")
                else:
                    # Take first N valid candidates (deterministic)
                    selected_candidates = valid_candidates[:args.num_identities]
                    print(f"Selected first {args.num_identities} identities from {len(valid_candidates)} valid candidates")
                
                # Sort selected candidates by identity name for consistent processing
                selected_candidates.sort(key=lambda x: x[0])
                
                # Collect exactly the specified number of samples from each selected identity
                total_collected = 0
                for subdir, available_count in selected_candidates:
                    subdir_path = os.path.join(args.image_path, subdir)
                    subdir_images = [f for f in os.listdir(subdir_path) 
                                   if f.lower().endswith(args.image_extension.lower())]
                    subdir_images.sort()  # Ensure consistent ordering
                    
                    # Sample exactly the required number
                    random.seed(args.random_seed)
                    selected_images = random.sample(subdir_images, args.samples_per_identity)
                    selected_images.sort()
                    
                    # Add full paths
                    full_paths = [Path(os.path.join(subdir_path, img)) for img in selected_images]
                    images.extend(full_paths)
                    total_collected += len(selected_images)
                
                print(f"SUCCESS: Collected exactly {total_collected} samples from {len(selected_candidates)} identities")
                print(f"Target achieved: {total_collected} samples = {args.num_identities} identities × {args.samples_per_identity} samples/identity")
                
                # Verify the exact count
                assert total_collected == required_total_samples, f"Internal error: collected {total_collected} != required {required_total_samples}"
                
            else:
                # Flexible mode: when either num_identities or samples_per_identity is not specified
                print("Flexible mode: Processing without strict count requirements")
                
                # Select identities (with optional random selection)
                if args.num_identities and len(subdirs) > args.num_identities:
                    if args.random_identity_selection:
                        random.seed(args.identity_selection_seed)
                        selected_subdirs = random.sample(subdirs, args.num_identities)
                        selected_subdirs.sort()
                        print(f"Randomly selected {args.num_identities} identities from {len(subdirs)} available")
                    else:
                        selected_subdirs = subdirs[:args.num_identities]
                        print(f"Selected first {args.num_identities} identities from {len(subdirs)} available")
                else:
                    selected_subdirs = subdirs
                    print(f"Processing all {len(subdirs)} identities")
                
                # Collect images with flexible sample counts
                for subdir in selected_subdirs:
                    subdir_path = os.path.join(args.image_path, subdir)
                    subdir_images = [f for f in os.listdir(subdir_path) 
                                   if f.lower().endswith(args.image_extension.lower())]
                    subdir_images.sort()
                    
                    # Apply samples per identity limit if specified
                    if args.samples_per_identity and len(subdir_images) > args.samples_per_identity:
                        random.seed(args.random_seed)
                        subdir_images = random.sample(subdir_images, args.samples_per_identity)
                        subdir_images.sort()
                    
                    # Add full paths
                    full_paths = [Path(os.path.join(subdir_path, img)) for img in subdir_images]
                    images.extend(full_paths)
                
                print(f"Collected {len(images)} total samples using flexible mode")
        else:
            # Flat structure: all images in one directory
            images = sorted(Path(args.image_path).glob(f"*.{args.image_extension}"))
            if args.samples_per_identity and len(images) > args.samples_per_identity:
                random.seed(args.random_seed)
                images = random.sample(images, args.samples_per_identity)
                images.sort()
    else:
        # Single file
        images = [Path(args.image_path)]
    
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Final validation: ensure we have exactly the expected number of samples
    if args.num_identities and args.samples_per_identity:
        expected_total = args.num_identities * args.samples_per_identity
        actual_total = len(images)
        
        print(f"\n=== Final Sample Count Validation ===")
        print(f"Expected: {expected_total} samples ({args.num_identities} identities × {args.samples_per_identity} samples/identity)")
        print(f"Actual: {actual_total} samples")
        
        if actual_total != expected_total:
            print(f"ERROR: Sample count mismatch!")
            print(f"Expected exactly {expected_total} samples but got {actual_total}")
            print(f"Cannot proceed with gradient calculations. Exiting.")
            return
        else:
            print(f"SUCCESS: Sample count validation passed!")
            print(f"Proceeding with gradient extraction for exactly {actual_total} samples.")
    else:
        print(f"\nFlexible mode: Processing {len(images)} samples without strict count requirements.")
    
    print(f"\n=== Starting Gradient Extraction ===")
    
    model_backbone = get_model(
                                nn_architecture=args.backbone,
                                rank=rank,
                                nn_weights_path=args.weights
                              )

    image_transforms = transforms.Compose(
                            [                            transforms.ToImage(),
                            transforms.Resize(size=(112,112),
                                              interpolation=transforms.InterpolationMode.BILINEAR,
                                              antialias=True),
                            transforms.ToDtype(torch.float32, scale=True),
                            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   
                            ]
                        )

    gradients = {k:[] for k in ["image", "block1", "block2", "block3", "block4"]}
    
    for path in tqdm(images):
        image = cv2.imread(str(path))
        if args.bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_transforms(image).unsqueeze(0).to(rank).requires_grad_(True)
        
        bn_score, (emb, block1, block2, block3, block4, bn) = model_backbone.get_BN(image)
        grads = autograd.grad(
                    outputs=bn_score,
                    inputs=[image, block1, block2, block3, block4]
                )
        print("embedding shape: ", emb.shape)
        print("block1 shape: ", block1.shape)
        print("block2 shape: ", block2.shape)
        print("block3 shape: ", block3.shape)
        print("block4 shape: ", block4.shape)
        print("grads: ", len(grads), grads[0].shape, grads[1].shape, grads[2].shape, grads[3].shape, grads[4].shape)
          # Store gradients and dump them with the same folder structure as images
        for idx, key in enumerate(["image", "block1", "block2", "block3", "block4"]):
            grad_tensor = grads[idx][0]
            
            # Dump gradient maintaining folder structure (if enabled)
            if args.dump_gradients:
                dump_gradients(
                    grad_tensor=grad_tensor,
                    image_path=path,
                    output_dir=output_folder,
                    layer_name=key
                )
              # Also store the sum for backward compatibility (optional)
            gradients[key].append(float(torch.abs(grad_tensor).sum()))
    
    if args.dump_gradients:
        print(f"Gradients saved to {output_folder}")
        print("Gradient folder structure mirrors the image folder structure.")
    
    # Optionally also save summary scores as text files
    if not args.dump_gradients:
        for key in ["image", "block1", "block2", "block3", "block4"]:
            output_file = output_folder / f"GraFIQs_{key}.txt"
            with open(output_file, "w") as f:
                for idx, score in enumerate(gradients[key]):
                    image_path = str(images[idx])
                    if args.path_replace is not None and args.path_replace_with is not None:
                        image_path = image_path.replace(args.path_replace, args.path_replace_with)
                    f.write(f"{image_path} {score}\n")
    
    # Final completion report
    print(f"\n=== Gradient Extraction Complete ===")
    if args.num_identities and args.samples_per_identity:
        expected_total = args.num_identities * args.samples_per_identity
        print(f"Successfully processed exactly {len(images)} samples as required")
        print(f"Target: {args.num_identities} identities × {args.samples_per_identity} samples = {expected_total} gradients")
        print(f"Result: {expected_total} gradient files generated for each layer (image, block1-4)")
        print(f"Total gradient files: {expected_total * 5} files")
    else:
        print(f"Successfully processed {len(images)} samples in flexible mode")
    print(f"Output directory: {output_folder}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='GraFIQs')
    
    parser.add_argument('--image-path', type=str, help='Path to images.')
    parser.add_argument('--image-extension', type=str, default="jpg", help='Extension/File type of images (e.g. jpg, png).')
    parser.add_argument('--output-dir', type=str, help='Directory to write score files to (will be created if it does not exist).')
    parser.add_argument('--backbone', type=str, choices=["iresnet50", "iresnet100"], help='Backbone architecture to use.')
    parser.add_argument('--weights', type=str, help='Path to backbone architecture weights.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use.')
    parser.add_argument('--path-replace', type=str, default=None, help='Prefix of image path which shall be replaced.')
    parser.add_argument('--path-replace-with', type=str, default=None, help='String that replaces prefix given in --path-replace.')
    parser.add_argument('--bgr2rgb', action='store_true', help='If specified, changes color space of CV2 image from BGR to RGB.')
    parser.add_argument('--flat-structure', action='store_true', help='If specified, treats the input as a flat directory structure instead of label-based subdirectories.')
    parser.add_argument('--dump-gradients', action='store_true', default=True, help='If specified, dumps gradient tensors maintaining folder structure.')
    parser.add_argument('--num-identities', type=int, default=None, help='Maximum number of identities to process. If None, process all identities.')
    parser.add_argument('--samples-per-identity', type=int, default=None, help='Maximum number of samples per identity. If None, use all samples.')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducible sample selection within identities.')
    parser.add_argument('--allow-insufficient-samples', action='store_true', help='If specified, include identities that have fewer samples than specified in --samples-per-identity.')
    parser.add_argument('--random-identity-selection', action='store_true', help='If specified, randomly select identities instead of taking the first N identities.')
    parser.add_argument('--identity-selection-seed', type=int, default=None, help='Random seed for identity selection. If None, uses the same seed as --random-seed.')

    main(parser.parse_args())
    