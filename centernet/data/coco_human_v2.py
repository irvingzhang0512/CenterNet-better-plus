import os
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data.datasets.register_coco import register_coco_instances

CAT_IDS = (1,)

_COCO_PERSON_INSTANCES = {
    "coco_person_2017_train": (
        "coco/train2017",
        "coco/annotations/instances_human_train2017.json",
    ),
    "coco_person_2017_val": (
        "coco/val2017",
        "coco/annotations/instances_human_val2017.json",
    ),
}


def _get_coco_instances_person_only_meta():
    thing_ids = CAT_IDS
    thing_colors = [COCO_CATEGORIES[id - 1]["color"] for id in CAT_IDS]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [COCO_CATEGORIES[id - 1]["name"] for id in CAT_IDS]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_coco_person(root):
    for key, (image_root, json_file) in _COCO_PERSON_INSTANCES.items():
        register_coco_instances(
            key,
            _get_coco_instances_person_only_meta(),
            os.path.join(root, json_file)
            if "://" not in json_file
            else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_coco_person(_root)
