export CUDA_VISIBLE_DEVICES=4,5,6,7

model_save_folder='./checkpoints/exp_ms1m_dul/'
log_tensorboard='./logtensorboard/exp_ms1m_dul/'
logs_file='./logs/exp_ms1m_dul.log'
trainset_folder='/home/admin/workspace/fuling/data/face_recog/ms1m/imgs/'

# notice: default kl_scale is 0.01 in DUL (base on original paper) 
python ../train_dul.py \
    --model_save_folder $model_save_folder \
    --log_tensorboard $log_tensorboard \
    --logs $logs_file \
    --gpu_id 0 1 2 3 \
    --stages 10 18 \
    --kl_scale 0.01 \
    --batch_size 1024 \
    --trainset_folder $trainset_folder \
    >> $logs_file 2>&1 &
