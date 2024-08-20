#!/bin/bash

# INPUTS:
mz_path="MODIFY THIS TO POINT TO MODELZOO PATH"
MZ_VENV="MODIFY THIS TO POINT TO /bin/activate OF YOUR VIRTUAL ENV"
NAMESPACE="MODIFY THIS TO YOUR CS CLUSTER NAMESPACE"

INITIAL_CHECKPOINT="checkpoints/CS/Meta-Llama-3-8B/model_to_cs-2.3.mdl"
NUM_CSX=1

# Configure environment:
source $MZ_VENV
export PYTHONPATH="${mz_path}"

# Train stage 1:
PARAMS_FILE=train_configs/stage_1.yaml
model_dir=./chatqa_inst_stage_1
JOB_LABEL="chatqa_inst_stage_1"
JOB_PRIORITY="p2"

python ${mz_path}/cerebras/modelzoo/models/nlp/llama/run.py CSX \
    --mode train \
    --num_csx $NUM_CSX \
    --params $PARAMS_FILE \
    --python_paths ${mz_path} \
    --mount_dirs ${mz_path} ./processed_datasets \
    --credentials_path /opt/cerebras/certs/tls.crt \
    --mgmt_namespace $NAMESPACE \
    --model_dir ${model_dir} \
    --checkpoint_path ${INITIAL_CHECKPOINT}
    --load_checkpoint_states=model
    --job_label Name=${JOB_LABEL} \
    --job_priority ${JOB_PRIORITY}


# Train stage 2:
PARAMS_FILE=train_configs/stage_2.yaml
model_dir=./chatqa_inst_stage_2
JOB_LABEL="chatqa_inst_stage_2"
JOB_PRIORITY="p2"
PHASE1_FINAL_CHECKPOINT="${model_dir}/checkpoint_1948.mdl"

python ${mz_path}/cerebras/modelzoo/models/nlp/llama/run.py CSX \
    --mode train \
    --num_csx $NUM_CSX \
    --params $PARAMS_FILE \
    --python_paths ${mz_path} \
    --mount_dirs ${mz_path} ./processed_datasets \
    --credentials_path /opt/cerebras/certs/tls.crt \
    --mgmt_namespace $NAMESPACE \
    --model_dir ${model_dir} \
    --checkpoint_path ${PHASE1_FINAL_CHECKPOINT}
    --load_checkpoint_states=model
    --job_label Name=${JOB_LABEL} \
    --job_priority ${JOB_PRIORITY}