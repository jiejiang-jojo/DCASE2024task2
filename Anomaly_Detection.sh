pretrain_model_checkpoint="pretrain_model_300"
finetuning_model_checkpoint="finetuning_model_18"
num_classes=26

echo "Train Anomaly Detection Model."
python3 Wilkinghoff_model/main.py \
    --pretrain_model_checkpoint "${pretrain_model_checkpoint}" \
    --finetuning_model_checkpoint "${finetuning_model_checkpoint}"\
    --num_classes "${num_classes}"
echo "Successfully finished Training."

