import torch
from centernet.centernet import build_model
from centernet.config import add_centernet_config
from detectron2.config import get_cfg
from torch import nn
from detectron2.structures import ImageList


class CenterNetInferenceModel(nn.Module):
    def __init__(self, center_model):
        super().__init__()
        self._model = center_model.cuda().eval()

    def preprocess_image(self, images):
        images = [self._model.normalizer(img / 255.) for img in images]
        images = ImageList.from_tensors(images,
                                        self._model.backbone.size_divisibility)
        return images

    def forward(self, imgs):
        imgs = self.preprocess_image(imgs)
        res = self._model.inference(imgs)[0]['instances']
        print(type(res.pred_boxes))
        return res.pred_boxes.tensor


if __name__ == '__main__':
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(
        "/hdd02/zhangyiyang/CenterNet-better-plus/configs/centernet_r_18_C4_1x.yaml"
    )
    model = build_model(cfg)

    model2 = CenterNetInferenceModel(model)
    img = torch.rand([1, 3, 224, 224]).cuda()
    print(img.device)
    traced_script_module = torch.jit.trace(model2, (img, ))
