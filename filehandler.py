import yaml

import cv2
import numpy as np

from torchsig.signals.signal_types import Signal
from torchsig.datasets.dataset_metadata import load_dataset_metadata
from torchsig.utils.file_handlers.base_handler import FileWriter, FileReader

### ------------------------------------------------------------------------------------ ###

class SpectrogramWriter(FileWriter):
    """Custom filehandler class for saving spectrogram images and labels to disk."""

    def _setup(self) -> None:
        """Setup directory for storing spectrogram images and labels."""
        self.spectrogram_dir = self.root.joinpath("spectrograms")
        self.label_dir = self.root.joinpath("labels")
        self.spectrogram_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)

    def write(self, batch_idx: int, batch: list[Signal]) -> None:
        """
        Write a single batch of Signals as spectrograms and YOLO labels to disk.

        Args:
            batch_idx:  Index of the batch being written.
            batch:      List of Signal objects in batch
        """

        for idx, sig in enumerate(batch):
            
            spectrogram = sig.data
            metadatas = sig.get_full_metadata()
            labels = {i: None for i in range(len(metadatas))}
            
            for i, m in enumerate(metadatas):
                for k, v in m.to_dict().items():
                    if v is not None:
                        labels[i] = [int(l) if l.is_integer() else float(l) for l in v]

            # First normalize data from (0-255)
            mi = spectrogram.min()
            ma = spectrogram.max()
            spectrogram: np.ndarray = ((spectrogram - mi) / (ma - mi)) * 255
            spectrogram = spectrogram.astype(np.uint8)

            # Apply colormap
            spectrogram = cv2.applyColorMap(spectrogram, cv2.COLORMAP_HOT)

            # Save spectrogram as PNG
            spectrogram_path = self.spectrogram_dir.joinpath(f"spectrogram_{batch_idx * len(batch) + idx}.png")
            cv2.imwrite(str(spectrogram_path), spectrogram, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            # Save labels as YAML file
            labels_path = self.label_dir.joinpath(f"labels_{batch_idx * len(batch) + idx}.yaml")
            with open(labels_path, "w") as f:
                yaml.dump(labels, f, default_flow_style=False)
    
    def __len__(self) -> int:
        """Return the number of saved spectrograms."""
        return len(list(self.spectrogram_dir.glob("spectrogram_*.png")))

class SpectrogramReader(FileReader):
    """Custom filehandler class for reading spectrograms and labels saved to disk."""

    def __init__(self, root):
        super().__init__(root=root)
        self.spectrogram_dir = self.root.joinpath("spectrograms")
        self.label_dir = self.root.joinpath("labels") 
        self.dataset_metadata = load_dataset_metadata(self.dataset_info_filepath)

    def read(self, idx: int) -> tuple[np.ndarray, dict]:
        """
        Read a spectrogram and labels by index.

        Args:
            idx:            Index of the data to read
        
        Returns:
            spectrogram:    Spectrogram as np.ndarray
            labels:         Labels for loaded spectrogram as dictionary
        """
        # Read the spectrogram as a numpy array
        spectrogram_path = self.root.joinpath("spectrograms", f"spectrogram_{idx}.png")
        spectrogram = cv2.imread(str(spectrogram_path), cv2.IMREAD_GRAYSCALE)

        # Read the labels as a dictionary
        labels_path = self.root.joinpath("labels", f"labels_{idx}.yaml")
        labels = []
        with open(labels_path, "r") as f:
            raw_labels = yaml.load(f, Loader=yaml.FullLoader)
            for item in raw_labels.keys():
                l = raw_labels[item]
                labels.append(l)
        
        return (spectrogram, labels)

    def size(self) -> int:
        """Return the total number of spectrograms in the dataset."""
        spectrograms_path = self.root.joinpath("spectrograms")
        return len(list(spectrograms_path.glob("spectrogram_*.png")))

    def __len__(self) -> int:
        """Return the total number of spectrograms in the dataset."""
        return self.size()
