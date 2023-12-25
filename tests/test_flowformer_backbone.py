import pytest

from projects.flow_detr.modeling.flowformer_cv import Flowformer
import math
from detectron2.modeling.backbone import SwinTransformer
import torch
from torch import nn


# Flowformer(
#     img_size=224,
#     patch_size=4,
#     # num_heads=8,
#     embed_dim=[96, 96 * 2, 96 * 4, 96 * 8],
#     depth=[3, 3, 10, 3],
#     mlp_ratio=[4] * 4,
#     num_classes=1000,
#     # ape=True
# )

@pytest.mark.parametrize("img_size", [224])
@pytest.mark.parametrize("image_size", [(800, 1000)])
@pytest.mark.parametrize("patch_size", [4])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("edm_tuple", [
    ([int(96 * math.pow(2, i)) for i in range(4)], [3, 3, 10, 3], [4] * 4),
    ([int(64 * math.pow(2, i)) for i in range(4)], [2, 2, 4, 10], [4] * 4),
])
def test_flowformer_backbone(img_size, patch_size, edm_tuple, num_heads, image_size):
    model = Flowformer(
        img_size=img_size,
        patch_size=patch_size,
        num_heads=num_heads,
        embed_dim=edm_tuple[0],
        depth=edm_tuple[1],
        mlp_ratio=edm_tuple[2],
        num_classes=None,
        ape=True
    )
    # model = SwinTransformer(pretrain_img_size=224,
    #                         embed_dim=96,
    #                         depths=(2, 2, 6, 2),
    #                         num_heads=(3, 6, 12, 24),
    #                         drop_path_rate=0.2,
    #                         window_size=7,
    #                         out_indices=(1, 2, 3))
    x = torch.randn(16, 3, image_size[0], image_size[1])
    out = model(x)
    proj = nn.Conv2d(out.size(1), 2048, kernel_size=1)
    out = proj(out)

    print(out.shape)
