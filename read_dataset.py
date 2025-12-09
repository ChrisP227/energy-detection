import yaml

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes

from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.datasets.dataset_metadata import DatasetMetadata
from torchsig.datasets.dataset_utils import to_dataset_metadata
from filehandler import SpectrogramReader

from torchsig.signals.signal_lists import TorchSigSignalLists

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

def main():
    filepath = "dataset.yaml"
    dataset_metadata = read_yaml(filepath)
    root = "/home/paulik/school/ee-5561/project/dataset"
    s = StaticTorchSigDataset(
        root=root,
        file_handler_class=SpectrogramReader
    )
    class_list = TorchSigSignalLists.all_signals
    for i in range(len(s)):
        data, labels = s[i]
        height, width = data.shape
        fig = plt.figure(figsize=(12,6))
        fig.tight_layout()

        ax = fig.add_subplot(1,1,1)
        pos = ax.imshow(data,aspect='auto',cmap='Wistia',vmin=dataset_metadata.noise_power_db)

        fig.colorbar(pos, ax=ax)


        for t in labels:
            classindex, xcenter, ycenter, normwidth, normheight = t


            actualwidth = width * normwidth
            actualheight = height * normheight

            actualxcenter = xcenter * width
            actualycenter = ycenter * height

            x_lowerleft = actualxcenter - (actualwidth / 2)
            y_lowerleft = actualycenter + (actualheight / 2)

            ax.add_patch(Rectangle(
                (x_lowerleft, y_lowerleft), 
                actualwidth, 
                -actualheight,
                linewidth=1, 
                edgecolor='blue', 
                facecolor='none'
            ))

            textDisplay = str(class_list[classindex])
            ax.text(x_lowerleft,y_lowerleft,textDisplay, bbox=dict(facecolor='w', alpha=0.5, linewidth=0))
        
        plt.show()

if __name__ == "__main__":
    main()
