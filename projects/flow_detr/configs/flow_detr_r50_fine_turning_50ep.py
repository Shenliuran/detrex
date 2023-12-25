# from detrex.config import get_config
from .flow_detr_r50_300ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.flow_detr_r50 import model

# modify training config
# train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "/root/autodl-tmp/train_output/flow_detr_r50_fine_turning_50ep_1/model_final.pth"
train.output_dir = "/root/autodl-tmp/train_output/flow_detr_r50_fine_turning_50ep"
train.fast_dev_run.enabled = False
train.max_iter = 50 * 7500 // 4
train.device = "cuda"
model.device = train.device

# # modify lr_multiplierF
lr_multiplier.scheduler.milestones = [40 * 7500 // 4, 50 * 7500 // 4]
# # modify optimizer config
# optimizer.lr = 1e-4
# optimizer.weight_decay = 1e-4

# model config
# model.backbone.out_features = ["res4"]
#
# model.in_features = ["res4"]
# model.in_channels = 2048 // 2  # 1024

model.fine_turning = True
# model.transformer.encoder.feedforward_dim = 2048 // 2  # 1024
model.transformer.encoder.competition_temperature = 0.2
# model.transformer.encoder.iter_fact = 1

model.transformer.decoder.proj_drop = 0.1
# model.transformer.decoder.feedforward_dim = 2048 // 2
# model.transformer.decoder.fine_turning = True

dataloader.train.total_batch_size = 64

dataloader.evaluator.output_dir = train.output_dir
