export CUDA_VISIBLE_DEVICES=3

logs_test_file='./logs_test/testfr_webface_dul.log'

model_for_test='./exp/checkpoints/exp_webface_dul/Backbone_IR_SE_64_DUL_Epoch_22_Batch_19558_Time_2021-09-11-01-39_checkpoint.pth'

python ../test_fr_dul.py \
    --model_for_test $model_for_test \
    >> $logs_test_file 2>&1 &
