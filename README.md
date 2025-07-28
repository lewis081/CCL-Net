# Underwater Image Enhancement with Cascaded Contrastive Learning
This repository is the official PyTorch implementation of CCL-Net.


## Dataset Preparation 
You should prepare the structure of datasets folder as follows:
``` 
├──path_to_data
    ├── train
            ├── raw
                ├── im1.png
                ├── im2.png
                └── ...
            ├── ref
                ├── im1.png
                ├── im2.png
                └── ...
    ├── test
            ├── raw
                ├── im1.png
                ├── im2.png
                └── ...
```

## Running Environment  
``` 
1. python 3.8
2. pip install -r requirements.txt
3. pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
``` 

## Test with the pre-trained models
```
1. rename directory pre-trained to checkpoints
2. run command 'python test.py --dataroot ./imgs --model HRNet'
```


## CC-Net

### Train
``` 
python train.py --dataroot /the_abs_path_of_data --model CCNet --lr 0.0005
```

### Test
```
python test.py --dataroot /the_abs_path_of_data --model CCNet
```

## HR-Net
### Notice
Training or Testing the HR-Net need load the pre-trained CC-Net. Therefore, you should have trained the CC-Net before you start to train or test the HR-Net.

### Train
``` 
python train.py --dataroot /the_abs_path_of_data --model HRNet --lr 0.001
```

### Test
```
python test.py --dataroot /the_abs_path_of_data --model HRNet
```

[//]: # (You can download the trained model from [here]&#40;https://drive.google.com/file&#41;.)




## Citation

```
@article{liu2024underwater,
  title={Underwater image enhancement with cascaded contrastive learning},
  author={Liu, Yi and Jiang, Qiuping and Wang, Xinyi and Luo, Ting and Zhou, Jingchun},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements
- https://github.com/trentqq/SGUIE-Net_Simple
- https://github.com/Li-Chongyi/Ucolor
- https://github.com/GlassyWu/AECR-Net
- https://github.com/zhilin007/FFA-Net
- https://github.com/swz30/MIRNetv2
