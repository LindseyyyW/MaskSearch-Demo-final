import sys
from pathlib import Path

# cd Demo-GUI
main = Path("./backend").resolve()
sys.path.append(str(main))

from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import shelve

from s2_masksearch import *


class ImagenettePath(datasets.Imagenette):
    def __getitem__(self, idx):
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label, path


def compute_dispersion(cam, threshold=(0.3, 0.45)):
    if isinstance(threshold, tuple):
        assert len(threshold) == 2
        return ((cam > threshold[0]) & (cam <= threshold[1])).sum()
    else:
        return (cam > threshold).sum()


def setup():
    # transform=  transforms.Compose([transforms.ToTensor(), transforms.Resize((400, 600))])
    # dataset = ImagenettePath(main/"data", size='full',
    #                          split='val', transform=transform, download=False)

    image_total = 7768  # len(dataset)
    dataset_examples = []
    for i in range(image_total):
        dataset_examples.append(f"{i}")

    image_access_order = range(len(dataset_examples))

    hist_size = 16
    hist_edges = []
    bin_width = 256 // hist_size
    for i in range(1, hist_size):
        hist_edges.append(bin_width * i)
    hist_edges.append(256)
    cam_size_y = 400
    cam_size_x = 600

    available_coords = 20

    in_memory_index_suffix = np.load(
        main/f"npy2/imagenet_cam_hist_prefix_{hist_size}_available_coords_{available_coords}_np_suffix.npy"
    )

    cam_map = shelve.open(main/"shelve/cam_map")
    image_map = shelve.open(main/"shelve/image_map")
    correctness_map = shelve.open(main/"shelve/correctness_map")
    attack_map = shelve.open(main/"shelve/attack_map")

    region_area_threshold = 5000
    region = (0, 0, cam_size_x, cam_size_y)

    return image_total, dataset_examples, image_access_order, \
           hist_size, hist_edges, bin_width, cam_size_y, cam_size_x, available_coords, \
           in_memory_index_suffix, cam_map, image_map, correctness_map, attack_map, \
           region_area_threshold, region
