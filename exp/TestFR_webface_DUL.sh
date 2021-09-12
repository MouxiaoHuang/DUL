export CUDA_VISIBLE_DEVICES=3

logs_test_file='./logs_test/testfr_webface_dul.log'

model_for_test=''

python ../test_fr_dul.py \
    --model_for_test $model_for_test \
    >> $logs_test_file 2>&1 &
