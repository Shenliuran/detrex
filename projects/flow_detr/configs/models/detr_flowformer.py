from detectron2.config import LazyCall as L

from projects.flow_detr.modeling.flowformer_cv import Flowformer

from projects.detr.configs.models.detr_r50_dc5 import model

model.backbone = L(Flowformer)(
    img_size=224,
    patch_size=4,
    num_heads=8,
    embed_dim=[96, 96 * 2, 96 * 4, 96 * 8],
    depth=[3, 3, 10, 3],
    mlp_ratio=[4] * 4,
    num_classes=None, ape=True
)

model.in_channels = 96 * 8
