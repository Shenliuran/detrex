export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export DETECTRON2_DATASETS=/root/autodl-tmp

MODEL_NAME=$1
BACKBONE="_"$2
EPOCH="_"$3
TASK=$4
TASK_PATTERN=$5
MODEL_CHECKPOINT=$6
NUM_GPUS=2
TRAIN_EVAL_BIN="python /root/detrex/projects/flow_detr/train_net.py"
#TRAIN_EVAL_BIN="python /root/detrex/tools/train_net.py"
ANALYZE_BIN="python /root/detrex/tools/analyze_model.py"
VISUALIZE_BIN="python /root/detrex/tools/visualize_data.py"
tools_dir="/root/detrex/tools/"
config_file_path="/root/detrex/projects/"${MODEL_NAME}"/configs/"${MODEL_NAME}${BACKBONE}${EPOCH}".py"
train_output_dir="/root/autodl-tmp/train_output/"${MODEL_NAME}${BACKBONE}${EPOCH}"/"
checkpoint=${train_output_dir}${MODEL_CHECKPOINT}
coco_dir="/root/autodl-tmp/coco/annotations/"
visualize_output_dir="/root/autodl-tmp/visualize_output/"

if [[ $TASK == "train" ]]; then
  if [[ $TASK_PATTERN == "install" ]]; then
    pip install -e .
    $TRAIN_EVAL_BIN \
      --config-file ${config_file_path} \
      --num-gpus $NUM_GPUS
  elif [[ $TASK_PATTERN == "resume" ]]; then
    $TRAIN_EVAL_BIN \
      --config-file ${config_file_path} \
      --num-gpus $NUM_GPUS \
      --resume
  elif [[ -z $TASK_PATTERN ]]; then
    $TRAIN_EVAL_BIN \
      --config-file ${config_file_path} \
      --num-gpus $NUM_GPUS
  else
    echo "Training pattern! please choose (install or resume)"
  fi
elif [[ $TASK == "eval" ]]; then
  $TRAIN_EVAL_BIN \
    --config-file ${config_file_path} \
    --num-gpus $NUM_GPUS \
    --eval-only \
    train.init_checkpoint=${checkpoint}
  eval $cmd
elif [[ $TASK == "analyze" ]]; then
  if [[ $TASK_PATTERN == "flop" ]]; then
    $ANALYZE_BIN \
      --num-inputs 100 \
      --tasks flop \
      --config-file ${config_file_path} \
      train.init_checkpoint=${checkpoint}
  elif [[ $TASK_PATTERN == "param" ]]; then
    $ANALYZE_BIN \
      --tasks parameter \
      --config-file ${config_file_path}
  elif [[ $TASK_PATTERN == "activations" ]]; then
    $ANALYZE_BIN \
      --num-inputs 100 \
      --tasks activation \
      --config-file ${config_file_path} \
      train.init_checkpoint=${checkpoint}
  elif [[ $TASK_PATTERN == "structure" ]]; then
    $ANALYZE_BIN --tasks structure \
      --config-file ${config_file_path}
  else
    echo "Analyzing pattern! please choose (flop, param, activations or structure)"
  fi
elif [[ $TASK == "visualize" ]]; then
  if [[ $TASK_PATTERN == "json" ]]; then
    python ${tools_dir}"visualize_json_results.py" \
      --input ${coco_dir}$3 \
      --output ${visualize_output_dir} \
      --dataset coco_2017_val
  elif [[ $TASK_PATTERN == "annotation" ]]; then
    $VISUALZIE_BIN \
      --config-file ${config_file_path} \
      --source annotation \
      --output-dir ${visualize_output_dir}
  elif [[ $TASK_PATTERN == "training" ]]; then
    $VISUALZIE_BIN \
      --config-file ${config_file_path} \
      --source dataloader \
      --output-dir ${visualize_output_dir}
  fi
else
  echo "please choose pattern (train ,eval, analyze or visualize)"
fi
#echo $cmd
