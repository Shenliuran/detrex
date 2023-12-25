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
# train.init_checkpoint = "/root/autodl-tmp/train_output/flow_detr_r50_fine_turning_largescale_50ep_1/model_0029999.pth"
train.output_dir = "/root/autodl-tmp/train_output/flow_detr_r50_fine_turning_largescale_50ep"
train.max_iter = 375000
train.fast_dev_run.enabled = False

# run evaluation every 5000 iters
train.eval_period = 5000

# modify lr_multiplierF
# lr_multiplier.scheduler.milestones = [375000 - 300000, 375000 - 189999]

# log training infomation every 20 iters
train.log_period = 20

optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4

# model config
# model.backbone.out_features = ["res4"]
# model.in_features = ["res4"]
# model.in_channels = 2048 // 2  # 1024

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

model.fine_turning = True
# model.embed_dim = 256 // 2
# model.position_embedding.num_pos_feats = 128 // 2
# model.transformer.encoder.feedforward_dim = 2048 // 2  # 1024
# model.transformer.encoder.competition_temperature = 0.1
# model.transformer.encoder.iter_fact = 7500
model.transformer.encoder.proj_drop = 0.1
# model.transformer.encoder.feedforward_dim = 2048 // 2
# model.transformer.encoder.embed_dim = 256 // 2

model.transformer.decoder.large_scale = True
model.transformer.decoder.fine_turning = True
# model.transformer.decoder.iter_fact = 7500
# model.transformer.decoder.competition_temperature = 0.1
model.transformer.decoder.proj_drop = 0.1
# model.transformer.decoder.feedforward_dim = 2048 // 2
# model.transformer.decoder.embed_dim = 256 // 2

dataloader.evaluator.output_dir = train.output_dir
