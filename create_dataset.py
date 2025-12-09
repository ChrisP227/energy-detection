import os
import yaml
import time
import argparse

from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.dataset_utils import to_dataset_metadata
from torchsig.datasets.datasets import TorchSigIterableDataset
from torchsig.utils.writer import DatasetCreator, default_collate_fn
from torchsig.utils.data_loading import WorkerSeedingDataLoader

# Transforms for generating data as spectrogram images with bounding box labels
from torchsig.transforms.transforms import Spectrogram
from torchsig.transforms.metadata_transforms import YOLOLabel

# Custom filehandler to save spectrograms as PNGs and labels as YAML files
from filehandler import SpectrogramWriter

import numpy as np
from shutil import disk_usage

### ------------------------------------------------------------------ ###

def read_yaml(filepath: str) -> DatasetMetadata:
    """
    Reads in dataset metadata from YAML file.

    Args:
        filepath: path to dataset yaml
    
    Returns:
        metadata: metadata loaded from yaml
    """
    # yaml loads in as a dict
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
    
    # Convert the dictionary to a DatasetMetadata object
    metadata = to_dataset_metadata(config)
    return metadata

def get_directory_size_gigabytes(start_path: str) -> float:
    """
    Returns total size of a directory (including subdirs) in gigabytes.

    Args:
        start_path:     The top level directory to check the size of

    Returns:
        total_size_gb:  The total size of the directory in GB
    """
    total_size = 0
    for path, _, files in os.walk(start_path):
        for f in files:
            fp = os.path.join(path, f)
            #total_size += os.path.getsize(fp)
            try:
                total_size += os.path.getsize(fp)
            except (OSError, FileNotFoundError):
                # file might have been deleted/moved by another thread
                # skip it and continue
                continue
    
    total_size_gb = total_size/(1000**3)
    return total_size_gb

def main():
    # Instantiate CLI arg parser
    parser = argparse.ArgumentParser(description="Quick script for generating spectrograms with labels.")
    parser.add_argument("root", type=str, help="Directory where data will be written to disk")
    parser.add_argument("-y", "--yaml", type=str, required=True, help="YAML file containing metadata for dataset")
    parser.add_argument("-n", "--numspec", type=int, default=10, help="Number of spectrograms to generate")
    parser.add_argument("-b", "--batchsize", type=int, default=10, help="How many samples per batch to load when generating data")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count()//3, help="Number of workers to use to generate data")
    parser.add_argument("-s", "--seed", type=int, default=13, help="Seed for generating dataset")
    parser.add_argument("-m", "--multithread", action="store_true", help="Flag indicating if multithreading should be used to speed up generation")

    # Parse arguments
    args = parser.parse_args()
    root = args.root
    yaml_filepath = args.yaml
    dataset_length = args.numspec
    batch_size = args.batchsize
    num_workers = args.workers
    seed = args.seed
    multithreading = args.multithread
    
    # Check the available storage
    disk_size_available_bytes = disk_usage(os.getcwd())[2]
    disk_size_available_gigabytes = np.round(disk_size_available_bytes/(1000**3),2)
    print(f"Available disk storage: {disk_size_available_gigabytes} GB")

    # Estimated storage is based on the dataset YAML used for this project (~350kB per spectrogram)
    print(f"Estimated storage required: {np.round((350e3*dataset_length)/(1000**3),2)} GB")

    # Load in the dataset metadata
    dataset_md = read_yaml(yaml_filepath)
    fft_size = dataset_md.fft_size
    transforms = [Spectrogram(fft_size=fft_size), YOLOLabel()]

    # This object iterably generates the dataset, one spectrogram at a time (essentially an infinite dataset)
    dataset = TorchSigIterableDataset(
        dataset_metadata=dataset_md,
        transforms=transforms,
        target_labels=["yolo_label"]
    )
    
    # A dataloader combines a dataset and a sampler, allowing us to iterate through generated data
    dataloader = WorkerSeedingDataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=default_collate_fn)
    dataloader.seed(seed)

    # The creator will iterate through the infinite dataset for the specified number of iterations and write the data to disk for us 
    dc = DatasetCreator(
        dataloader=dataloader,
        dataset_length=dataset_length,
        root=root,
        overwrite=True,
        multithreading=multithreading,
        file_handler=SpectrogramWriter
    )

    # Time the dataset generation
    t_start = time.time()
    dc.create()
    t_end = time.time()

    # Print some stats
    print(f"\nGenerated dataset in {t_end - t_start} seconds")
    print(f"Dataset directory size: {get_directory_size_gigabytes(start_path=root)} GB\n")


if __name__ == "__main__":
    main()
