import contextlib
import logging
import io
import os

from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

logger = logging.getLogger(__name__)


def load_coco_json(json_file,
                   image_root,
                   cat_ids=None,
                   dataset_name=None,
                   extra_annotation_keys=None):
    """
    modified from detectron2.data.dtasets.coco.py load_coco_json
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(
            json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        if cat_ids is None:
            cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [
            c["name"] for c in sorted(cats, key=lambda x: x["id"])
        ]
        meta.thing_classes = thing_classes

        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning("""
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
""")
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    total_img_ids = sorted(coco_api.imgs.keys())
    img_ids = []
    anns = []
    for img_id in total_img_ids:
        anno_ids = coco_api.getAnnIds(img_id, cat_ids)
        if len(anno_ids) > 0:
            anns.append(coco_api.loadAnns(anno_ids))
            img_ids.append(img_id)
    imgs = coco_api.loadImgs(img_ids)

    if "minival" not in json_file:
        ann_ids = [
            ann["id"] for anns_per_image in anns for ann in anns_per_image
        ]
        assert len(set(ann_ids)) == len(
            ann_ids), "Annotation ids in '{}' are not unique!".format(
                json_file)

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(
        len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"
                ] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert anno.get(
                "ignore",
                0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm
                        if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation))
    return dataset_dicts


def register_coco_instances(name, metadata, json_file, image_root, cat_ids=1):
    """
    modified from detectron2.data.datasets.register_coco.py
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_coco_json(json_file, image_root, cat_ids, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadata)


_COCO_PERSON_INSTANCES = {
    "coco_person_2017_train":
    ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_person_2017_val":
    ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_person_2017_test":
    ("coco/test2017", "coco/annotations/image_info_test2017.json"),
}


def _get_meta(cat_ids):
    return {
        "thing_dataset_id_to_contiguous_id":
        {k: i
         for i, k in enumerate(cat_ids)},
        "thing_classes": [COCO_CATEGORIES[i - 1]['name'] for i in cat_ids],
        "thing_colors": [COCO_CATEGORIES[i - 1]['color'] for i in cat_ids]
    }


def register_coco_person(root, cat_ids=[1]):
    for key, (image_root, json_file) in _COCO_PERSON_INSTANCES.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_meta(cat_ids),
            os.path.join(root, json_file)
            if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cat_ids=cat_ids,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_person(_root)
