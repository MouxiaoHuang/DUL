## Implementation of DUL (PyTorch version)

#### Introduction

---

This repo is an ***unofficial*** PyTorch implementation of DUL ([Data Uncertainty Learning in Face Recognition, CVPR2020](https://arxiv.org/abs/2003.11339)). 

NOTE: 

1. *SE-Resnet64 is used as defult backbone in this repo*, you can define others in `./backbone/model_irse.py`
2. *Training (process)* & *Testing (results)* logs can be found in `./exp/logs/` & `./exp/logs_test/`
3. *Implementation details are not exactly the same as the original paper*, can be seen in `./config.py`



#### Getting Started

---

- Clone this repo

```
git clone https://github.com/MouxiaoHuang/DUL.git
```

- Prepare env

```python
conda create --name <env_name> python=3.8
pip install -r requirements.txt
```

- Prepare trainset and testset
  - Trainset: [Casia WebFace or MS-Celeb-1M](https://github.com/ZhaoJ9014/face.evoLVe)
  - Testset: [LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW, VGG2_FP](https://github.com/ZhaoJ9014/face.evoLVe)
- Training

```python
sh ./exp/Exp_webface_DUL.sh
# or
sh ./exp/Exp_ms1m_DUL.sh
```

- Testing

```python
sh ./exp/TestFR_webface_DUL.sh
# or
sh ./exp/TestFR_ms1m_DUL.sh
```



#### Thanks & Refs

---

- [ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe)
- [Ontheway361/dul-pytorch](https://github.com/Ontheway361/dul-pytorch)
