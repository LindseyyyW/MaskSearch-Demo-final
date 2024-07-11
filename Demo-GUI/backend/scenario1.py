from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import sys
from topk import *
from s1_data_process import data_process
app = Flask(__name__)
CORS(app)

@app.route('/api/scenario1/topk_search/pairs', methods=['POST'])
def pairs():
    data = {}

    for key in sorted_class_pairs.keys():
        cell = "({},{})".format(key[0], key[1])
        data[cell] = sorted_class_pairs[key]
    #print(data.keys())
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
    
    image_ids = [image_idx for (metric, area, image_idx) in images]
    end = time.time()
    time_used = end - start
    print("time: ", end - start)
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time': time_used})


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
    num = 0
    end = time.time()
    time_used = end - start
    

    image_ids = [image_idx for (metric,image_idx) in images]
    # Dummy implementation to return the query command and some mock image IDs
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time': time_used})

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
        v = 0.1 if reverse else 0.9
        region_area_threshold = 5000
        imag = naive_get_images_satisfying_filter(
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
    print(images)
    image_ids = [int(image_idx) for (metric, image_idx) in images]
    end = time.time()
    time_used = end - start
    execution_time = round(time_used, 3)
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time': execution_time, 'images_count': len(image_ids)})



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
        region_area_threshold = 5000
        imag = naive_get_images_satisfying_filter(
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
    if(len(images)>50): 
        num = 50
    else: 
        num = len(images)
    
 
    images = sorted(
        [(item[0], item[1]) for item in images], reverse=not reverse
    )
    print(images)
    image_ids = [int(image_idx) for (metric,image_idx) in images[:num]]
    end = time.time()
    time_used = end - start
    execution_time = round(time_used, 3)
    
    return jsonify({'query_command': query_command, 'image_ids': image_ids, 'execution_time' : execution_time, 'images_count': images_count})
   

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
    id_val_data, ood_val_data, label_map, pred_map, cam_map, object_detection_map, dataset_examples, in_memory_index_suffix, image_access_order, sorted_class_pairs, names= data_process()
     
    in_memory_index_suffix_in = np.load(
        f"./intersect_index.npy"
    )
    in_memory_index_suffix_un = np.load(
        f"./union_index.npy"
    )

    
    app.run(port=8000)

