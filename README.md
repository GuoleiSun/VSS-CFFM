# VSS-CFFM
Official PyTorch implementation of CVPR 2022 paper: Coarse-to-Fine Feature Mining for Video Semantic Segmentation

## Abstract
The contextual information plays a core role in semantic segmentation. As for video semantic segmentation, the contexts include static contexts and motional contexts, corresponding to static content and moving content in a video clip, respectively. The static contexts are well exploited in image semantic segmentation by learning multi-scale and global/long-range features. The motional contexts are studied in previous video semantic segmentation. However, there is no research about how to simultaneously learn static and motional contexts which are highly correlated and complementary to each other. To address this problem, we propose a Coarse-to-Fine Feature Mining (CFFM) technique to learn a unified presentation of static contexts and motional contexts. This technique consists of two parts: coarse-to-fine feature assembling and cross-frame feature mining. The former operation prepares data for further processing, enabling the subsequent joint learning of static and motional contexts. The latter operation mines useful information/contexts from the sequential frames to enhance the video contexts of the features of the target frame. The enhanced features can be directly applied for the final prediction. Experimental results on popular benchmarks demonstrate that the proposed CFFM performs favorably against state-of-the-art methods for video semantic segmentation.

![block images](https://github.com/GuoleiSun/VSS-CFFM/blob/main/diagram.png)

Authors: [Guolei Sun](https://scholar.google.com/citations?hl=zh-CN&user=qd8Blw0AAAAJ), [Yun Liu](https://yun-liu.github.io/), [Henghui Ding](https://henghuiding.github.io/), [Thomas Probst](https://probstt.bitbucket.io/), Luc Van Gool.

## Installation
Please follow the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```timm==0.3.0, CUDA11.0, pytorch==1.7.1, torchvision==0.8.2, mmcv==1.3.0, opencv-python==4.5.2```

Download this repository and install by:
```
cd VSS-CFFM && pip install -e . --user
```

## Usage
### Data preparation
Please follow [VSPW](https://github.com/sssdddwww2/vspw_dataset_download) to download VSPW 480P dataset.
After correctly downloading, the file system is as follows:
```
vspw-480
├── video1
    ├── origin
        ├── .jpg
    └── mask
        └── .png
```
The dataset should be put in ```/repo_path/data/vspw/```. Or you can use Symlink: 
```
mkdir -p data/vspw/
ln -s /dataset_path/VSPW_480p data/vspw/
```

### Test
1. Download the trained weights from [here](https://drive.google.com/drive/folders/1YD5Yy6_m3QlS72o6FQWmsFtz7Kw-8OmI?usp=sharing).
2. Run the following commands:
```
# Multi-gpu testing
./tools/dist_test.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2.160k.py /path/to/checkpoint_file <GPU_NUM>
```

### Training
Training requires 4 Nvidia GPUs, each of which has > 20G GPU memory.
```
# Multi-gpu testing
./tools/dist_train.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2.160k.py 4 --work-dir model_path/vspw2/work_dirs_4g_b1
```

## License
This project is only for academic use.

## Acknowledgement
The code is heavily based on the following repositories:
- https://github.com/open-mmlab/mmsegmentation
- https://github.com/NVlabs/SegFormer
- https://github.com/microsoft/Focal-Transformer

Thanks for their amazing works.

## Citation
```bibtex
@inproceedings{sun2022vss,
    title={Coarse-to-Fine Feature Mining for Video Semantic Segmentation},
    author={Sun, Guolei and Liu, Yun and Ding, Henghui and Probst, Thomas and Van Gool, Luc},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision and Patern Recognition (CVPR)},
    year={2022}
}
```
## Contact
- Guolei Sun, sunguolei.kaust@gmail.com
- Yun Liu, yun.liu@vision.ee.ethz.ch
