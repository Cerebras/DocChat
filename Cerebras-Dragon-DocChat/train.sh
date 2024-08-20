#!/bin/bash
mz_path="MODIFY THIS TO POINT TO MODELZOO PATH"
MZ_VENV="MODIFY THIS TO POINT TO /bin/activate OF YOUR VIRTUAL ENV"
NAMESPACE="MODIFY THIS TO THE CORRECT NAMESPACE"

INITIAL_CHECKPOINT="checkpoints/CS/facebook_dragon/pytorch_model.mdl"
JOB_LABEL="retrieval_training"
JOB_PRIORITY="p2"
NUM_CSX=1
model_dir=./docchat_retrieval_model_dir
PARAMS_FILE=train_configs/params_train.yaml


# Configure environment:
source $MZ_VENV
export PYTHONPATH="${mz_path}"

        
python ${mz_path}/cerebras/modelzoo/models/nlp/dpr/run.py CSX \
  --mode train \
  --num_csx $NUM_CSX \
  --params $PARAMS_FILE \
  --python_paths ${mz_path} \
  --mount_dirs ${mz_path} ./processed_datasets \
  --credentials_path /opt/cerebras/certs/tls.crt \
  --disable_version_check \
  --mgmt_namespace $NAMESPACE \
  --model_dir ${model_dir} \
  --checkpoint_path ${INITIAL_CHECKPOINT} \
  --job_label Name=${JOB_LABEL} \
  --job_priority ${JOB_PRIORITY}



