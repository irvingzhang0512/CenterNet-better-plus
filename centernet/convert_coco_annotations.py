import json


def _filter_by_cat_ids(src_anno, target_anno, cat_ids=(1,)):
    d = json.load(open(src_anno, "r"))
    human_annos = []
    img_ids = set()
    for anno in d["annotations"]:
        if anno["category_id"] in cat_ids:
            human_annos.append(anno)
            img_ids.add(anno["image_id"])

    images = []
    for img in d["images"]:
        if img["id"] in img_ids:
            images.append(img)

    d["images"] = images
    d["annotations"] = human_annos
    d["categories"] = d["categories"][:1]

    json.dump(d, open(target_anno, "w"))


src_anno = "/hdd02/zhangyiyang/CenterNet-better-plus/datasets/coco/annotations/instances_val2017.json"
target_anno = "/hdd02/zhangyiyang/CenterNet-better-plus/datasets/coco/annotations/instances_human_val2017.json"
_filter_by_cat_ids(src_anno, target_anno)


src_anno = "/hdd02/zhangyiyang/CenterNet-better-plus/datasets/coco/annotations/instances_train2017.json"
target_anno = "/hdd02/zhangyiyang/CenterNet-better-plus/datasets/coco/annotations/instances_human_train2017.json"
_filter_by_cat_ids(src_anno, target_anno)
