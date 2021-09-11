# test face recognition performance of dul model
import torch
import os

from config import dul_args_func, Backbone_Dict, Test_FR_Data_Dict
from util.utils import get_data_pair, perform_face_recog


class DUL_FR_Tester():
    def __init__(self, dul_args) -> None:
        self.dul_args = dul_args
        self.dul_args.multi_gpu = False

    def face_recog(self):
        BACKBONE = Backbone_Dict[self.dul_args.backbone_name]
        if os.path.isfile(self.dul_args.model_for_test):
            print('=' * 60, flush=True)
            print("Model for testing Face Recognition performance is:\n '{}' ".format(self.dul_args.model_for_test), flush=True)
            BACKBONE.load_state_dict(torch.load(self.dul_args.model_for_test))
            BACKBONE = BACKBONE.cuda().eval()
        else:
            print('=' * 60, flush=True)
            print('No model found for testing!', flush=True)
            print('=' * 60, flush=True)
            return
        print('=' * 60, flush=True)
        print('Face Recognition Performance on different dataset is as shown below:', flush=True)
        print('=' * 60, flush=True)
        for value in Test_FR_Data_Dict.values():
            testdata, testdata_issame = get_data_pair(self.dul_args.testset_fr_folder, value)
            accuracy, best_threshold, roc_curve = perform_face_recog(self.dul_args.multi_gpu, self.dul_args.embedding_size,
                                                                    self.dul_args.batch_size, BACKBONE, testdata, testdata_issame)
            print(value.upper(), ': ', accuracy, flush=True)
        print('=' * 60, flush=True)
        print('Testing finished!', flush=True)
        print('=' * 60, flush=True)


if __name__ == '__main__':
    dul_fr_test = DUL_FR_Tester(dul_args_func())
    dul_fr_test.face_recog()
