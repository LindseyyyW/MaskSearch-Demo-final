import sys
from pathlib import Path

# cd Demo-GUI
main = Path("./backend").resolve()
sys.path.append(str(main))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import heapq
from operator import itemgetter
import time
# import pickle
import shelve

from s2_masksearch import *

app = Flask(__name__)
CORS(app)


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


@app.route('/api/scenario2/topk_search', methods=['POST'])
def topk_search():
    data = request.json
    k = int(data.get('k'))
    lb, ub = data.get('pixelLowerBound'), data.get('pixelUpperBound')
    order = data.get('order')
    lv, uv = float(lb), float(ub)
    reverse = False if data.get('order') == 'DESC' else True
    fn = topk_search_ms if data.get('ms') else topk_search_np

    query_command = f"""
                     SELECT mask_id
                     FROM MasksDatabaseView
                     ORDER BY CP(mask, full_image, ({lb}, {ub})) / area(roi) {order}
                     LIMIT {k};
                     """
    return fn(query_command, k, lv, uv, reverse)


def topk_search_ms(query_command, k, lv, uv, reverse):
    start = time.time()
    count, images = get_max_area_in_subregion_in_memory_version(
        "imagenet",
        image_map_2,
        correctness_map_2,
        attack_map_2,
        cam_map_2,
        None,
        bin_width_2,
        cam_size_y_2,
        cam_size_x_2,
        hist_size_2,
        dataset_examples_2,
        lv,
        uv,
        region_2,
        in_memory_index_suffix_2,
        image_access_order_2,
        early_stoppable=False,
        k=k,
        region_area_threshold=region_area_threshold_2,
        ignore_zero_area_region=True,
        reverse=reverse,
        visualize=False,
        available_coords=available_coords_2,
        compression=None,
    )
    image_ids = [image_idx for (metric, area, image_idx) in images]
    end = time.time()

    total = len(cam_map_2)
    execution_time = end - start
    print("Skipped images:", count)
    print("(MaskSearch vanilla) Query time (cold cache):", execution_time)
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'count': total - count, 'total': total, 'execution_time': execution_time})


def topk_search_np(query_command, k, lv, uv, reverse):
    vanilla_sort = heapq.nlargest if not reverse else heapq.nsmallest
    start = time.time()

    dispersion_data = []
    for idx in cam_map_2:
        cam = cam_map_2[idx]
        dispersion_data.append(compute_dispersion(cam, threshold=(lv, uv)))

    top_k = vanilla_sort(k, enumerate(dispersion_data), key=itemgetter(1))

    image_ids = [image_idx for (image_idx, dispersion) in top_k]
    end = time.time()

    execution_time = end - start
    print("(Numpy naive) Query time:", end - start)
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'skipped_images_count': 0, 'execution_time': execution_time})


@app.route('/topk_cams/<filename>')
def topk_cam(filename):
    return send_from_directory(str(main/'cam_images'), filename)


@app.route('/topk_images/<filename>')
def topk_image(filename):
    return send_from_directory(str(main/'pure_images'), filename)


@app.route('/topk_labels/<image_id>')
def topk_labels(image_id):
    return jsonify({'correctness': correctness_map_2[image_id], 'attack': attack_map_2[image_id]})


if __name__ == '__main__':
    image_total_2, dataset_examples_2, image_access_order_2, \
    hist_size_2, hist_edges_2, bin_width_2, cam_size_y_2, cam_size_x_2, available_coords_2, \
    in_memory_index_suffix_2, cam_map_2, image_map_2, correctness_map_2, attack_map_2, \
    region_area_threshold_2, region_2 = setup()
    app.run(port=9000)
