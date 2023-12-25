from detrex.config import get_config

dataloader = get_config("common/data/coco_detr.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

from .models.flow_detr_flowformer import model

# modify training config
# train.init_checkpoint = "/root/autodl-tmp/pretrain_model/converted_detr_r50_500ep.pth"
train.init_checkpoint = "/root/autodl-tmp/pretrain_model/flow_detr_flowformer.pth"
# train.init_checkpoint = "/root/autodl-tmp/pretrain_model/flowformer_cv_300ep.pth"
train.output_dir = "/root/autodl-tmp/train_output/flow_detr_flowformer_50ep"
train.max_iter = 375000
train.fast_dev_run.enabled = True

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

# run evaluation every 5000 iters
train.eval_period = 5000

optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4

# log training infomation every 20 iters
train.log_period = 20

# model config
# model.fine_turning = True
# model.backbone.out_features = ["res4"]
# model.in_features = ["res4"]
# model.in_channels = 2048 // 2  # 1024

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# model.transformer.encoder.feedforward_dim = 2048 // 2  # 1024
model.transformer.encoder.competition_temperature = 0.1
model.transformer.encoder.proj_drop = 0.1
# model.transformer.encoder.iter_fact = 7500

# model.transformer.decoder.competition_temperature = 10
model.transformer.decoder.proj_drop = 0.1

dataloader.evaluator.output_dir = train.output_dir

dataloader.train.num_workers = 16
dataloader.train.total_batch_size = 8
