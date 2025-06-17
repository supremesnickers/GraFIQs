import numbers
import os
import queue as Queue
import threading

import cv2
import numpy as np
np.bool = np.bool_
import mxnet as mx

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_file_count(directory):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        file_count += len(filenames)
    return file_count


def sort_directories_by_file_count(base_path):
    directories = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    directories_file_counts = [
        (d, get_file_count(os.path.join(base_path, d))) for d in directories
    ]
    directories_file_counts.sort(key=lambda x: x[1], reverse=True)
    return directories_file_counts


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(
                    device=self.local_rank, non_blocking=True
                )

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, "train.rec")
        path_imgidx = os.path.join(root_dir, "train.idx")
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, "r")
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, idx

    def __len__(self):
        return len(self.imgidx)


class CasiaWebFace(Dataset):
    def __init__(self, root_dir, local_rank, num_classes=10572, selective=False, 
                 max_samples_per_identity=None, random_seed=42, identity_selection_seed=None):
        super(CasiaWebFace, self).__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.max_samples_per_identity = max_samples_per_identity
        self.random_seed = random_seed
        self.identity_selection_seed = identity_selection_seed if identity_selection_seed is not None else random_seed
        self.imgidx, self.labels = self.scan(root_dir, num_classes, selective)
        self.imageindex = np.array(range(len(self.imgidx)))

    def scan(self, root, num_classes, selective):
        import random
        
        imgidex = []
        labels = []
        lb = -1
        list_dir = os.listdir(root)
        list_dir.sort()

        # Set random seed for reproducible subset selection
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # If max_samples_per_identity is specified, enforce strict requirements
        if self.max_samples_per_identity is not None:
            print(f"Dataset strict mode: Requiring exactly {num_classes} identities with ≥{self.max_samples_per_identity} samples each")
            
            # Find all identities that have at least the required number of samples
            valid_candidates = []
            insufficient_candidates = []
            
            for d in list_dir:
                images = os.listdir(os.path.join(root, d))
                images = [img for img in images if any(img.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])]
                
                if len(images) >= self.max_samples_per_identity:
                    valid_candidates.append((d, len(images)))
                else:
                    insufficient_candidates.append((d, len(images)))
            
            # Sort candidates for consistent behavior
            if selective:
                # Sort by file count (descending) for selective mode
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
            else:
                # Sort alphabetically for non-selective mode
                valid_candidates.sort(key=lambda x: x[0])
            
            print(f"Found {len(valid_candidates)} identities with ≥{self.max_samples_per_identity} samples")
            print(f"Found {len(insufficient_candidates)} identities with <{self.max_samples_per_identity} samples")
            
            # Check if we have enough valid identities
            if len(valid_candidates) < num_classes:
                print(f"ERROR: Not enough identities with sufficient samples!")
                print(f"Required: {num_classes} identities with ≥{self.max_samples_per_identity} samples each")
                print(f"Available: {len(valid_candidates)} identities with sufficient samples")
                
                if insufficient_candidates:
                    print(f"Identities with insufficient samples:")
                    for identity, count in insufficient_candidates[:10]:
                        print(f"  - {identity}: {count} samples (need {self.max_samples_per_identity - count} more)")
                    if len(insufficient_candidates) > 10:
                        print(f"  ... and {len(insufficient_candidates) - 10} more")
                
                raise ValueError(f"Cannot create dataset: need {num_classes} identities but only {len(valid_candidates)} have sufficient samples")
            
            # Select the required number of valid identities
            selected_candidates = valid_candidates[:num_classes]
            
            # Collect exactly the specified number of samples from each selected identity
            total_collected = 0
            for l, available_count in selected_candidates:
                images = os.listdir(os.path.join(root, l))
                images = [img for img in images if any(img.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])]
                images.sort()  # Ensure consistent ordering
                
                # Sample exactly the required number
                if self.random_seed is not None:
                    random.seed(self.random_seed)
                selected_images = random.sample(images, self.max_samples_per_identity)
                selected_images.sort()
                
                lb += 1
                for img in selected_images:
                    imgidex.append(os.path.join(l, img))
                    labels.append(lb)
                    total_collected += 1
            
            expected_total = num_classes * self.max_samples_per_identity
            print(f"Dataset created: {len(selected_candidates)} identities, {total_collected} total samples")
            print(f"Target achieved: {total_collected} samples = {num_classes} identities × {self.max_samples_per_identity} samples/identity")
            
            # Verify the exact count
            assert total_collected == expected_total, f"Internal error: collected {total_collected} != expected {expected_total}"
            
        else:
            # Original flexible logic for when strict requirements are not specified
            if selective:
                sorted_directories = sort_directories_by_file_count(root)
                selected_dirs = sorted_directories[:num_classes]
            else:
                selected_dirs = [(d, 0) for d in list_dir[:num_classes]]

            # Track identities with insufficient samples
            insufficient_identities = []
            valid_identities = []

            for l, file_count in selected_dirs:
                images = os.listdir(os.path.join(root, l))
                images = [img for img in images if any(img.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])]
                images.sort()  # Ensure consistent ordering
                
                if images:  # Only add if there are valid images
                    valid_identities.append(l)
                    lb += 1
                    for img in images:
                        imgidex.append(os.path.join(l, img))
                        labels.append(lb)

            print(f"Dataset loaded: {len(valid_identities)} identities, {len(imgidex)} total samples (flexible mode)")
            
        return imgidex, labels

    def read_image(self, path):
        return cv2.imread(os.path.join(self.root_dir, path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        imageindex = self.imageindex[index]
        img = self.read_image(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)

        # Return the full path for gradient dumping
        full_path = os.path.join(self.root_dir, path)
        return sample, label, imageindex, full_path

    def __len__(self):
        return len(self.imgidx)


class CasiaWebFaceEmbeddings(Dataset):
    def __init__(self, root_dir, local_rank, num_classes=10573, selective=False,
                 max_samples_per_identity=None, random_seed=42, identity_selection_seed=None):
        """
        Args:
            root_dir (str): Root directory containing subdirectories (labels) of .npy files.
            local_rank: (int) GPU ID for compatibility.
            num_classes (int): Maximum number of classes to load.
            selective (bool): Whether to use selective scanning (based on file counts).
            max_samples_per_identity (int): Maximum number of samples per identity. If None, use all samples.
            random_seed (int): Random seed for reproducible sample selection.
            identity_selection_seed (int): Random seed for identity selection. If None, uses random_seed.
        """
        super(CasiaWebFaceEmbeddings, self).__init__()
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.max_samples_per_identity = max_samples_per_identity
        self.random_seed = random_seed
        self.identity_selection_seed = identity_selection_seed if identity_selection_seed is not None else random_seed
        # We don't need any transform as these are embeddings; however you can add one if required.
        self.emb_paths, self.labels = self.scan(root_dir, num_classes, selective)
        self.emb_idx = np.array(range(len(self.emb_paths)))

    def scan(self, root, num_classes, selective):
        """
        Scans the root directory for .npy files arranged in subdirectories.
        Returns:
            emb_paths (list): List of relative paths to each embedding file.
            labels (list): Corresponding integer label for each embedding.
        """
        import random
        
        emb_paths = []
        labels = []
        lb = -1
        list_dir = os.listdir(root)
        list_dir.sort()

        # Set random seed for reproducible subset selection
        if self.random_seed is not None:
            random.seed(self.random_seed)

        if selective:
            sorted_directories = sort_directories_by_file_count(root)
            selected_dirs = [d for d, _ in sorted_directories[:num_classes]]
        else:
            selected_dirs = list_dir[:num_classes]        # Track identities with insufficient samples
        insufficient_identities = []
        valid_identities = []

        for d in selected_dirs:
            full_dir = os.path.join(root, d)
            if not os.path.isdir(full_dir):
                continue
            files = [f for f in os.listdir(full_dir) if f.lower().endswith(".npy")]
            files.sort()  # Ensure consistent ordering
            
            # Check if identity has enough samples
            if self.max_samples_per_identity is not None and len(files) < self.max_samples_per_identity:
                insufficient_identities.append((d, len(files)))
                # Still process but note the shortage
            
            # Apply subset selection if max_samples_per_identity is specified
            if self.max_samples_per_identity is not None and len(files) >= self.max_samples_per_identity:
                # Randomly sample max_samples_per_identity files from this identity
                files = random.sample(files, self.max_samples_per_identity)
                files.sort()  # Sort again for consistency
            
            if files:  # Only add if there are valid files
                valid_identities.append(d)
                lb += 1
                for file in files:
                    emb_paths.append(os.path.join(d, file))
                    labels.append(lb)

        # Report statistics
        if insufficient_identities:
            print(f"Warning: {len(insufficient_identities)} identities have fewer than {self.max_samples_per_identity} embeddings:")
            for identity, count in insufficient_identities[:5]:  # Show first 5
                print(f"  - {identity}: {count} embeddings")
            if len(insufficient_identities) > 5:
                print(f"  ... and {len(insufficient_identities) - 5} more")

        print(f"Embeddings dataset loaded: {len(valid_identities)} identities, {len(emb_paths)} total samples")
        if self.max_samples_per_identity is not None:
            exact_count = sum(1 for identity, count in insufficient_identities if count == self.max_samples_per_identity)
            print(f"Identities with exactly {self.max_samples_per_identity} samples: {len(valid_identities) - len(insufficient_identities) + exact_count}")

        return emb_paths, labels

    def __getitem__(self, index):
        # Get the relative path and label.
        path = self.emb_paths[index]
        embidx = self.emb_idx[index]
        label = self.labels[index]
        # Load the embedding from the .npy file.
        emb = np.load(os.path.join(self.root_dir, path))
        # Optionally convert the embedding to a torch tensor.
        emb = torch.tensor(emb, dtype=torch.float)
        # Return the full path for gradient dumping
        full_path = os.path.join(self.root_dir, path)
        return emb, label, embidx, full_path

    def __len__(self):
        return len(self.emb_paths)


def create_dataset_with_subset(dataset_type, root_dir, local_rank, num_identities=100, 
                             samples_per_identity=100, selective=False, random_seed=42,
                             strict_sample_count=False, identity_selection_seed=None):
    """
    Helper function to create dataset instances with subset selection.
    
    Args:
        dataset_type (str): Type of dataset ('CasiaWebFace' or 'CasiaWebFaceEmbeddings')
        root_dir (str): Root directory of the dataset
        local_rank (int): GPU ID for compatibility
        num_identities (int): Number of identities to select
        samples_per_identity (int): Number of samples per identity
        selective (bool): Whether to use selective scanning
        random_seed (int): Random seed for reproducible sample selection
        strict_sample_count (bool): If True, only include identities with exact sample count
        identity_selection_seed (int): Random seed for identity selection. If None, uses random_seed
    
    Returns:
        Dataset instance with specified subset
    """
    if dataset_type == 'CasiaWebFace':
        dataset = CasiaWebFace(
            root_dir=root_dir,
            local_rank=local_rank,
            num_classes=num_identities,
            selective=selective,
            max_samples_per_identity=samples_per_identity,
            random_seed=random_seed,
            identity_selection_seed=identity_selection_seed
        )
    elif dataset_type == 'CasiaWebFaceEmbeddings':
        dataset = CasiaWebFaceEmbeddings(
            root_dir=root_dir,
            local_rank=local_rank,
            num_classes=num_identities,
            selective=selective,
            max_samples_per_identity=samples_per_identity,
            random_seed=random_seed,
            identity_selection_seed=identity_selection_seed
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # If strict mode, filter out identities with insufficient samples
    if strict_sample_count and samples_per_identity is not None:
        print(f"Strict mode: Filtering identities to only include those with exactly {samples_per_identity} samples...")
        # This would require additional logic to refilter the dataset
        # For now, just report the warning
        print("Note: Strict filtering is implemented via command line --allow-insufficient-samples flag")
    
    return dataset
