from .flow_detr_r50_300ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.flow_detr_r50 import model

# modify training config
# train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.init_checkpoint = "/root/autodl-tmp/pretrain_model/converted_detr_r50_500ep.pth"
train.output_dir = "/root/autodl-tmp/train_output/flow_detr_r50_fine_turning_profile_100ep"
train.max_iter = 750000
train.fast_dev_run.enabled = False
train.device = "cuda"
model.device = train.device

# modify lr_multiplierF
lr_multiplier.scheduler.milestones = [675000, 750000]

# model config
model.backbone.out_features = ["res4"]

model.in_features = ["res4"]
model.in_channels = 2048 // 2  # 1024

model.fine_turning = True
# model.transformer.encoder.feedforward_dim = 2048 // 2  # 1024
model.transformer.encoder.competition_temperature = 0.5
model.transformer.encoder.iter_fact = 7500

# model.transformer.decoder.feedforward_dim = 2048 // 2
# model.transformer.decoder.fine_turning = True
# model.transformer.decoder.large_scale = True
# model.transformer.decoder.competition_temperature = 0.05
# model.transformer.decoder.iter_fact = None

dataloader.evaluator.output_dir = "/root/autodl-tmp/evaluate_output/flow_detr_r50_fine_turning_large_scale_300ep"
