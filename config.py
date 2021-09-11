import argparse
from backbone.model_irse import *


def dul_args_func():
    parser = argparse.ArgumentParser(description='DUL: Data Uncertainty Learning in Face Recognition')

    # ----- random seed for reproducing
    parser.add_argument('--random_seed', type=int, default=6666)

    # ----- directory (train & test)
    parser.add_argument('--trainset_folder', type=str, default='/home/huangmouxiao.hmx/data/face_rec/casia_maxpy_clean_align/')
    parser.add_argument('--model_save_folder', type=str, default='./checkpoints/')
    parser.add_argument('--log_tensorboard', type=str, default='./logtensorboard/')
    parser.add_argument('--logs', type=str, default='./logs/')
    parser.add_argument('--testset_fr_folder', type=str, default='/home/huangmouxiao.hmx/data/face_rec/usual_test/')
    parser.add_argument('--testset_ood_folder', type=str, default='')
    parser.add_argument('--model_for_test', type=str, default='')

    # ----- training env
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=str, nargs='+')
    
    # ----- resume pretrain details
    parser.add_argument('--resume_backbone', type=str, default='')
    parser.add_argument('--resume_head', type=str, default='')
    parser.add_argument('--resume_epoch', type=int, default=0)
    
    # ----- model & training details
    parser.add_argument('--backbone_name', type=str, default='IR_SE_64_DUL')
    parser.add_argument('--head_name', type=str, default='ArcFace')
    parser.add_argument('--loss_name', type=str, default='Softmax')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--arcface_scale', type=int, default=64)
    parser.add_argument('--input_size', type=list, default=[112, 112]) # support: [112, 112] and [224, 224]
    parser.add_argument('--center_crop', type=bool, default=True)
    parser.add_argument('--rgb_mean', type=list, default=[0.5, 0.5, 0.5])
    parser.add_argument('--rgb_std', type=list, default=[0.5, 0.5, 0.5])
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # ----- hyperparameters
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_epoch', type=int, default=22)
    parser.add_argument('--warm_up_epoch', type=int, default=1)
    parser.add_argument('--image_noise', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--stages', type=str, nargs='+')
    parser.add_argument('--kl_scale', type=float, default=0.01)

    args = parser.parse_args()

    return args

dul_args = dul_args_func()

Backbone_Dict = {
    'IR_50': IR_50(dul_args.input_size),
    'IR_101': IR_101(dul_args.input_size),
    'IR_152': IR_152(dul_args.input_size),
    'IR_SE_50': IR_SE_50(dul_args.input_size),
    'IR_SE_64_DUL': IR_SE_64_DUL(dul_args.input_size),
    'IR_SE_101': IR_SE_101(dul_args.input_size),
    'IR_SE_152': IR_SE_152(dul_args.input_size)
}

Test_FR_Data_Dict = {
    'lfw': 'lfw',
    'cfp_ff': 'cfp_ff',
    'cfp_fp': 'cfp_fp',
    'agedb': 'agedb_30',
    'calfw': 'calfw',
    'cplfw': 'cplfw',
    'vgg2_fp': 'vgg2_fp'
}