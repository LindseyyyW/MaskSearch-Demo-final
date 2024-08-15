from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin

import sys
from pathlib import Path

# cd Demo-GUI
main = Path("./backend").resolve()
sys.path.append(str(main))

from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import heapq
from operator import itemgetter
import time
import shelve

from topk import *
from s1_data_process import data_process
from s2_masksearch import *
from s2_utils import compute_dispersion, setup

app = Flask(__name__)
CORS(app)


@app.route('/api/scenario1/topk_search/pairs', methods=['POST'])
def pairs():
    data = {}

    for key in sorted_class_pairs.keys():
        cell = "({},{})".format(key[0], key[1])
        data[cell] = sorted_class_pairs[key]
    print(names)

    return jsonify({'dict':data, 'names': names})


@app.route('/api/scenario1/topk_search', methods=['POST'])
def topk_search():
    data = request.json
    k = data.get('k')
    enable = data.get('ms')
    k = int(k)
    roi = 'True' if data.get('roi') == 'object bounding box' else 'False'
    region = 'object' if roi == 'True' else 'custom'
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    lv = float(pixel_lower_bound)
    uv = float(pixel_upper_bound)
    order = data.get('order')
    reverse = False if order == 'DESC' else True

    query_command = f"""
    SELECT mask_id
    FROM MasksDatabaseView
    ORDER BY CP(mask, roi, ({pixel_lower_bound}, {pixel_upper_bound})) / area(roi) {order}
    LIMIT {k};
    """
    start = time.time()
    # Dummy implementation to return the query command and some mock image IDs
    hist_size = 16
    bin_width = 256 // hist_size
    cam_size_y = 448
    cam_size_x = 448
    region_area_threshold = 5000
    available_coords = 64

    #get_max_area_in_subregion_in_memory_version
    if enable:
        count, images = get_max_area_in_subregion_in_memory_version(
        "wilds",
        (id_val_data, ood_val_data),
        label_map,
        pred_map,
        cam_map,
        object_detection_map,
        bin_width,
        cam_size_y,
        cam_size_x,
        hist_size,
        dataset_examples,
        lv,
        uv,
        region,
        in_memory_index_suffix,
        image_access_order,
        early_stoppable=False,
        k=k,
        region_area_threshold=region_area_threshold,
        ignore_zero_area_region=True,
        reverse=reverse,
        visualize=True,
        available_coords=available_coords,
        compression=None,
        )
    else:
        count, images = vanilla_topk(
            "wilds",
            (id_val_data, ood_val_data),
            label_map,
            pred_map,
            cam_map,
            object_detection_map,
            bin_width,
            cam_size_y,
            cam_size_x,
            hist_size,
            dataset_examples,
            lv,
            uv,
            region,
            in_memory_index_suffix,
            image_access_order,
            early_stoppable=False,
            k=k,
            region_area_threshold=region_area_threshold,
            ignore_zero_area_region=True,
            reverse=reverse,
            visualize=False,
            available_coords=None,
            compression=None,
        )
        count = 0

    image_ids = [image_idx for (metric, area, image_idx) in images]
    end = time.time()
    time_used = end - start
    total = len(cam_map)
    count = total - count
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time': time_used, 'count': count, 'total': total})


@app.route('/api/scenario1/augment', methods=['POST'])
def augment():
    data = request.json
    img_ids = data.get('image_ids')
    # Now you can use img_ids for augmentation
    print(img_ids)
    return jsonify({'image_ids': img_ids})


@app.route('/api/scenario1/get_pairs', methods=['GET'])
def get_pairs():
    # Generate a list of 100 pairs (1, 2)
    pairs = [{'x': 3, 'y': 4} for _ in range(100)]
    return jsonify(pairs)


@app.route('/api/receive_selected', methods=['POST'])
def receive_selected():
    selected_lines = request.json.get('selected_lines', [])
    print('Received selected lines:', selected_lines)  # For debugging purposes
    return jsonify({'message': 'Selection received successfully'})


@app.route('/api/scenario1/filter_search', methods=['POST'])
def filter_search():
    data = request.json
    threshold = data.get('threshold')
    enable = data.get('ms')
    v = float(threshold)
    roi = 'True' if data.get('roi') == 'object bounding box' else 'False'
    region = 'object' if roi == 'True' else 'custom'
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    comparison = data.get('thresholdDirection')
    reverse = True if comparison == '<' else False
    lv = float(pixel_lower_bound)
    uv = float(pixel_upper_bound)

    query_command = f"""
    SELECT mask_id
    FROM MasksDatabaseView
    WHERE CP(mask, roi, ({pixel_lower_bound}, {pixel_upper_bound})) / area(roi) {comparison} {threshold};
    """

    hist_size = 16
    bin_width = 256 // hist_size
    cam_size_y = 448
    cam_size_x = 448
    region_area_threshold = 5000
    available_coords = 64
    start = time.time()
    if enable:
        count, images = get_images_satisfying_filter("wilds",
            cam_map,
            object_detection_map,
            in_memory_index_suffix,
            bin_width,
            hist_size,
            cam_size_y,
            cam_size_x,
            dataset_examples,
            lv,
            uv,
            region,
            v,
            region_area_threshold,
            True,
            available_coords,
        None,
        reverse=reverse)
    else:
        images = naive_get_images_satisfying_filter(
            cam_map,
            object_detection_map,
            cam_size_y,
            cam_size_x,
            dataset_examples,
            lv,
            uv,
            region,
            v,
            region_area_threshold,
            ignore_zero_area_region=True,
            compression=None,
            reverse=reverse,
            visualize=False,
        )
        count = 0
    num = 0
    end = time.time()
    time_used = end - start
    total = len(cam_map)
    num_count = np.sum(count)
    num_count = total - num_count
    num_count = int(num_count)
    print(total, num_count)
    image_ids = [image_idx for (metric,image_idx) in images]
    # Dummy implementation to return the query command and some mock image IDs
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time': time_used, 'count': num_count, 'total': total})


@app.route('/topk_results/<filename>')
def topk_image(filename):
    return send_from_directory('topk_results', filename)


@app.route('/filter_results/<filename>')
def filter_image(filename):
    return send_from_directory('filter_results', filename)


@app.route('/augment_results/<filename>')
def augment_image(filename):
    return send_from_directory('augment_results', filename)


@app.route('/orig_image/<filename>')
def orig_image(filename):
    return send_from_directory('orig_image', filename)


###############################################################################

###############################################################################


@app.route('/api/scenario2/topk_search', methods=['POST'])
def topk_search_2():
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
def topk_cam_2(filename):
    return send_from_directory(str(main/'cam_images'), filename)


@app.route('/topk_images/<filename>')
def topk_image_2(filename):
    return send_from_directory(str(main/'pure_images'), filename)


@app.route('/topk_labels/<image_id>')
def topk_labels_2(image_id):
    return jsonify({'correctness': correctness_map_2[image_id], 'attack': attack_map_2[image_id]})


###############################################################################

###############################################################################

# for scenario3

@app.route('/api/scenario3/topk_search', methods=['POST'])
def topk_search_s3():
    data = request.json
    k = data.get('k')
    enable = data.get('ms')
    k = int(k)
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    order = data.get('order')
    reverse = False if order == 'DESC' else True

    query_command = f"""
    SELECT mask_id,
    CP(intersect(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound}))
    / CP(union(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound})) as iou
    FROM MasksDatabaseView WHERE mask_type IN (1, 2)
    GROUP BY image_id ORDER BY iou {order} LIMIT {k};
    """
    start = time.time()
    cam_size_x = 384
    cam_size_y = 384
    hist_size = 2
    bin_width = 256 // hist_size
    total_images = 11788
    examples = np.arange(1, 11788)
    available_coords = 16
    lv = 0.0
    uv = 1.0
    region = (0, 0, 384, 384)
    print(enable)
    if not enable:
        count = 0
        images = naive_topk_IOU(
            cam_size_y,
            cam_size_x,
            bin_width,
            hist_size,
            examples,
            lv,
            uv,
            intersection_mask,
            union_mask,
            region,
            k,
            reverse,
        )
    else:
        count, images = get_max_IoU_across_masks_in_memory(
            cam_size_y=384,
            cam_size_x=384,
            bin_width=bin_width,
            hist_size=2,
            examples=examples,
            lv=lv,
            uv=uv,
            in_memory_index_suffix_in=in_memory_index_suffix_in,
            in_memory_index_suffix_un=in_memory_index_suffix_un,
            region=region,
            k=k,
            region_area_threshold=0,
            ignore_zero_area_region=True,
            reverse=reverse,
            available_coords=available_coords,
            compression=None,
        )
    image_ids = [int(image_idx) for (metric, image_idx) in images]
    image_ids = sorted(image_ids)
    print(image_ids)
    end = time.time()
    time_used = end - start
    execution_time = round(time_used, 3)
    total = 11788
    count = total - count
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time': execution_time, 'images_count': len(image_ids), 'count': count, 'total': total})


@app.route('/api/scenario3/filter_search', methods=['POST'])
def filter_search_s3():
    data = request.json
    threshold = data.get('threshold')
    enable = data.get('ms')
    v = float(threshold)
    pixel_upper_bound = data.get('pixelUpperBound')
    pixel_lower_bound = data.get('pixelLowerBound')
    comparison = data.get('thresholdDirection')
    reverse = True if comparison == '<' else False

    query_command = f"""
    SELECT mask_id,
    CP(intersect(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound}))
    / CP(union(mask), roi, ({pixel_lower_bound}, {pixel_upper_bound})) as iou
    FROM MasksDatabaseView WHERE iou {comparison} {threshold}, mask_type IN (1, 2)
    GROUP BY image_id;
    """

    # query_command = f"""
    # SELECT mask_id
    # FROM MasksDatabaseView
    # WHERE CP(mask, roi, ({pixel_lower_bound}, {pixel_upper_bound})) / area(roi) {comparison} {threshold};
    # """

    cam_size_x = 384
    cam_size_y = 384
    hist_size = 2
    bin_width = 256 // hist_size
    total_images = 11788
    examples = np.arange(1, 11788)
    available_coords = 16
    lv = 0.0
    uv = 1.0
    region = (0, 0, 384, 384)

    start = time.time()
    if not enable:
        count = 0
        images = naive_Filter_IoU(
            cam_size_y,
            cam_size_x,
            bin_width,
            hist_size,
            examples,
            lv,
            uv,
            intersection_mask,
            union_mask,
            region,
            v,
            reverse,
            available_coords,
        )
    else:
        count, images = get_Filter_IoU_across_masks_in_memory(
            cam_size_y=384,
            cam_size_x=384,
            bin_width=bin_width,
            hist_size=2,
            examples=examples,
            lv=lv,
            uv=uv,
            in_memory_index_suffix_in=in_memory_index_suffix_in,
            in_memory_index_suffix_un=in_memory_index_suffix_un,
            region=region,
            v=v,
            region_area_threshold=0,
            ignore_zero_area_region=True,
            reverse=reverse,
            available_coords=available_coords,
            compression=None,
        )
    num = 0
    images_count = len(images)
    num = len(images)
    print(images)
    image_ids = [int(image_idx) for (metric,image_idx) in images[:num]]
    image_ids = sorted(image_ids)
    end = time.time()
    time_used = end - start
    execution_time = round(time_used, 3)
    total = 11788
    count = total - count
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time' : execution_time, 'images_count': images_count, 'count': count, 'total': total})


@app.route('/saliency_images/<filename>')
def topk_image_s3(filename):
    return send_from_directory('saliency_images', filename)


@app.route('/human_att_images/<filename>')
def filter_image_s3(filename):
    return send_from_directory('human_att_images', filename)


@app.route('/intersect_visualization/<filename>')
def intersect_image_s3(filename):
    return send_from_directory('intersect_visualization', filename)


@app.route('/union_visualization/<filename>')
def union_image_s3(filename):
    return send_from_directory('union_visualization', filename)


if __name__ == '__main__':
    #app.run(debug=True)
    id_val_data, ood_val_data, label_map, pred_map, cam_map, object_detection_map, \
    dataset_examples, in_memory_index_suffix, image_access_order, \
    sorted_class_pairs, names, union_mask, intersection_mask = data_process()

    image_total_2, dataset_examples_2, image_access_order_2, \
    hist_size_2, hist_edges_2, bin_width_2, cam_size_y_2, cam_size_x_2, available_coords_2, \
    in_memory_index_suffix_2, cam_map_2, image_map_2, correctness_map_2, attack_map_2, \
    region_area_threshold_2, region_2 = setup()

    in_memory_index_suffix_in = np.load(main/"npy/intersect_index.npy")
    in_memory_index_suffix_un = np.load(main/"npy/union_index.npy")

    app.run(port=9000)

