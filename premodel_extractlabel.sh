stage=1     
stop_stage=3
pretrain_model_checkpoint="pretrain_model_300"
finetuning_model_checkpoint="finetuning_model_18"
num_classes=26

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "Stage 1: Training Pretrain model."
    python3 Premodel_extractlabel/pre_train.py 
    echo "Successfully finished Training Pretrain model ."
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Finetuning model."
    python3 Premodel_extractlabel/finetuning.py \
        --pretrain_model_checkpoint "${pretrain_model_checkpoint}"
    echo "Successfully finished Finetuning model."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Extract label."
    python3 Premodel_extractlabel/label_ext.py \
        --pretrain_model_checkpoint "${pretrain_model_checkpoint}" \
        --finetuning_model_checkpoint "${finetuning_model_checkpoint}"\
        --num_classes "${num_classes}"
    echo "Successfully finished Extracting label."
fi

