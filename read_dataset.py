import yaml
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from torchsig.datasets.datasets import StaticTorchSigDataset
from filehandler import SpectrogramReader

from torchsig.signals.signal_lists import TorchSigSignalLists

def main():
    parser = argparse.ArgumentParser(description="Quick script for visualizing a dataset that was generated and written to disk with torchsig.")
    parser.add_argument("root", type=str, help="Directory containing generated spectrograms and labels")
    args = parser.parse_args()
    root = args.root

    s = StaticTorchSigDataset(
        root=root,
        file_handler_class=SpectrogramReader
    )

    class_list = TorchSigSignalLists.all_signals
    for i in range(5):
        data, labels = s[i]
        height, width = data.shape
        fig = plt.figure(figsize=(12,6))
        fig.tight_layout()

        ax = fig.add_subplot(1,1,1)
        pos = ax.imshow(data,aspect='auto',cmap='Wistia')

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
