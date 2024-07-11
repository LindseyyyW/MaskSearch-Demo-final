import sys

sys.argv = [""]
sys.path.append("/Users/linxiwei/Documents/MaskSearch/Archive/masksearch")
                
from topk import *
import argparse
import json
import pickle
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from utils import *
from pytorch_grad_cam import (
    AblationCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    RandomCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
import shelve
import time

def data_process():

    dir="/Users/linxiwei/Documents/MaskSearch/Archive/wilds/"
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(
        dataset="iwildcam",
        download=True,
        root_dir="/Users/linxiwei/Documents/MaskSearch/Archive/wilds/"
    )

    # Get the ID validation set
    id_val_data = dataset.get_subset(
        "id_val",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )

    ood_val_data = dataset.get_subset(
        "val",
        transform=transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor()]
        ),
    )

    

    # Load from disk
    cam_map = shelve.open(dir + "id_ood_val_cam_map.shelve")
    with open(dir + "id_ood_val_pred.pkl", "rb") as f:
        pred_map = pickle.load(f)
    with open(dir + "id_ood_val_label.pkl", "rb") as f:
        label_map = pickle.load(f)

    

    id_total = 7314
    ood_total = 14961
    dataset_examples = []
    for distribution, image_total in zip(["id_val", "ood_val"], [id_total, ood_total]):
        for image_idx in range(1, 1 + image_total):
            dataset_examples.append(f"{distribution}_{image_idx}")
        
    hist_size = 16
    hist_edges = []
    bin_width = 256 // hist_size
    for i in range(1, hist_size):
        hist_edges.append(bin_width * i)
    hist_edges.append(256)

    available_coords = 64

    object_detection_map = load_object_region_index_in_memory(
        dataset_examples,
        dir + "id_ood_val_object_detection_map.shelve",
    )

    in_memory_index_suffix = np.load(
        f"{dir}id_ood_val_cam_hist_prefix_{hist_size}_in_memory_available_coords_{available_coords}_suffix.npy"
    )

    image_access_order = range(len(dataset_examples))

    #compute confusion matrix



    cam_size_y = 448
    cam_size_x = 448

    class_pairs = {}

    for i in range(len(dataset_examples)):
            img = dataset_examples[i]
            x = label_map[img]
            y = pred_map[img]
            a, b, w, h = get_object_region(
            object_detection_map, cam_size_y, cam_size_x, img
            )
            if w==0 or h==0:
                continue
            
            if ((x,y) in class_pairs):
                class_pairs.get((x,y)).append(img)
            else:
                class_pairs[(x,y)] = [img]

    sorted_class_pairs = dict(sorted(class_pairs.items(), key=lambda item: len(item[1]), reverse=True))
    # print(sorted_class_pairs[(147,17)])

    # for key, value in sorted_class_pairs.items():
    #     print(key, ':', len(value))

    keys_to_delete = []
    for key, value in sorted_class_pairs.items():
        a, b = key
        if a == b or a == 0 or b == 0:
            keys_to_delete.append(key)
        elif a==1 and b==146:
            keys_to_delete.append(key)
        

    # Deleting the items
    for key in keys_to_delete:
        del sorted_class_pairs[key]

    names = {
        1: "tayassu pecari",
        2: "dasyprocta punctata",
        3: "cuniculus paca",
        4: "puma concolor",
        5: "tapirus terrestris",
        6: "pecari tajacu",
        7: "mazama americana",
        8: "leopardus pardalis",
        9: "geotrygon montana",
        10: "nasua nasua",
        11: "dasypus novemcinctus",
        12: "eira barbara",
        13: "didelphis marsupialis",
        14: "procyon cancrivorus",
        15: "panthera onca",
        16: "myrmecophaga tridactyla",
        17: "tinamus major",
        18: "sylvilagus brasiliensis",
        19: "puma yagouaroundi",
        20: "leopardus wiedii",
        21: "mazama gouazoubira",
        22: "philander opossum",
        23: "capra aegagrus",
        24: "bos taurus",
        25: "ovis aries",
        26: "canis lupus",
        27: "lepus saxatilis",
        28: "papio anubis",
        29: "genetta genetta",
        30: "tragelaphus scriptus",
        31: "herpestes sanguineus",
        32: "loxodonta africana",
        33: "cricetomys gambianus",
        34: "raphicerus campestris",
        35: "hyaena hyaena",
        36: "aepyceros melampus",
        37: "crocuta crocuta",
        38: "caracal caracal",
        39: "equus ferus",
        40: "panthera leo",
        41: "tragelaphus oryx",
        42: "kobus ellipsiprymnus",
        43: "phacochoerus africanus",
        44: "panthera pardus",
        45: "ichneumia albicauda",
        46: "canis mesomelas",
        47: "syncerus caffer",
        48: "equus quagga",
        49: "giraffa camelopardalis",
        50: "alcelaphus buselaphus",
        51: "chlorocebus pygerythrus",
        52: "madoqua guentheri",
        53: "potamochoerus larvatus",
        54: "nanger granti",
        55: "eudorcas thomsonii",
        56: "struthio camelus",
        57: "orycteropus afer",
        58: "acinonyx jubatus",
        59: "eupodotis senegalensis",
        60: "felis silvestris",
        61: "oryx beisa",
        62: "lophotis gindiana",
        63: "ardeotis kori",
        64: "lissotis melanogaster",
        65: "argusianus argus",
        66: "prionailurus bengalensis",
        67: "hemigalus derbyanus",
        68: "muntiacus muntjak",
        69: "sus scrofa",
        70: "helarctos malayanus",
        71: "rusa unicolor",
        72: "hystrix brachyura",
        73: "pardofelis temminckii",
        74: "panthera tigris",
        75: "lariscus insignis",
        76: "chalcophaps indica",
        77: "genetta tigrina",
        78: "hystrix cristata",
        79: "lycaon pictus",
        80: "procavia capensis",
        81: "canis familiaris",
        82: "unknown bird",
        83: "unknown bat",
        84: "momotus momota",
        85: "dasyprocta fuliginosa",
        86: "geotrygon sp",
        87: "nasua narica",
        88: "tamandua mexicana",
        89: "didelphis sp",
        90: "penelope purpurascens",
        91: "phaetornis sp",
        92: "brotogeris sp",
        93: "camelus dromedarius",
        94: "otocyon megalotis",
        95: "acryllium vulturinum",
        96: "equus grevyi",
        97: "proteles cristata",
        98: "leptailurus serval",
        99: "tragelaphus strepsiceros",
        100: "hippopotamus amphibius",
        101: "burhinus capensis",
        102: "paguma larvata",
        103: "pardofelis marmorata",
        104: "cuon alpinus",
        105: "varanus salvator",
        106: "martes flavigula",
        107: "prionodon linsang",
        108: "rollulus rouloul",
        109: "lophura inornata",
        110: "polyplectron chalcurum",
        111: "manis javanica",
        112: "capricornis sumatraensis",
        113: "macaca sp",
        114: "francolinus nobilis",
        115: "cercopithecus lhoesti",
        116: "cephalophus nigrifrons",
        117: "atherurus africanus",
        118: "pan troglodytes",
        119: "cercopithecus mitis",
        120: "funisciurus carruthersi",
        121: "motacilla flava",
        122: "andropadus latirostris",
        123: "andropadus virens",
        124: "thamnomys venustus",
        125: "protoxerus stangeri",
        126: "paraxerus boehmi",
        127: "cephalophus silvicultor",
        128: "oenomys hypoxanthus",
        129: "melocichla mentalis",
        130: "hybomys univittatus",
        131: "colomys goslingi",
        132: "hylomyscus stella",
        133: "genetta servalina",
        134: "canis adustus",
        135: "mus minutoides",
        136: "musophaga rossae",
        137: "turtur tympanistria",
        138: "praomys tullbergi",
        139: "malacomys longipes",
        140: "alopochen aegyptiaca",
        141: "deomys ferrugineus",
        142: "francolinus africanus",
        143: "turdus olivaceus",
        144: "mazama sp",
        145: "urocyon cinereoargenteus",
        146: "meleagris ocellata",
        147: "crax rubra",
        148: "agouti paca",
        149: "tapirus bairdii",
        150: "procyon lotor",
        151: "odocoileus virginianus",
        152: "leptotila plumbeiceps",
        153: "mazama temama",
        154: "conepatus semistriatus",
        155: "mazama pandora",
        156: "ortalis vetula",
        157: "presbytis thomasi",
        158: "neofelis diardi",
        159: "arctonyx hoevenii",
        160: "tragulus sp",
        161: "dendrocitta occipitalis",
        162: "niltava sumatrana",
        163: "leiothrix argentauris",
        164: "arborophila rubrirostris",
        165: "lophura sp",
        166: "myiophoneus glaucinus",
        167: "lophura erythrophthalma",
        168: "spilornis cheela",
        169: "myiophoneus caeruleus",
        170: "herpestes semitorquatus",
        171: "cerdocyon thous",
        172: "motorcycle",
        173: "peromyscus sp",
        174: "puma yagoroundi",
        175: "tigrisoma mexicanum",
        176: "claravis pretiosa",
        177: "sciurus sp",
        178: "ave desconocida",
        179: "aramides cajanea",
        180: "unknown dove",
        181: "mazama  temama"
    }

    

    return id_val_data, ood_val_data, label_map, pred_map, cam_map, object_detection_map, dataset_examples, in_memory_index_suffix, image_access_order, sorted_class_pairs, names