export CUDA_VISIBLE_DEVICES=2,3

model_save_folder='./checkpoints/exp_webface_dul/'
log_tensorboard='./logtensorboard/exp_webface_dul/'
logs_file='./logs/exp_webface_dul.log'

# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python ../train_dul.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --logs $logs_file \
    --gpu_id 0 1 \
    --stages 10 18 \
    --kl_scale 0.01 \
    >> $logs_file 2>&1 &
