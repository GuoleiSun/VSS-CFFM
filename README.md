# VSS-CFFM & CFFM++
Official PyTorch implementation of CFFM, proposed in CVPR 2022 paper: Coarse-to-Fine Feature Mining for Video Semantic Segmentation, [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Sun_Coarse-To-Fine_Feature_Mining_for_Video_Semantic_Segmentation_CVPR_2022_paper.pdf)

Official PyTorch implementation of an extension method CFFM++ (Learning Local and Global Temporal Contexts for Video Semantic Segmentation, TPAMI 2024), built upon CFFM by additionally exploiting global temporal contexts from the whole video, [paper](https://arxiv.org/pdf/2204.03330v2.pdf)

## Introduction
### CFFM
The contextual information plays a core role in semantic segmentation. As for video semantic segmentation, the contexts include static contexts and motional contexts, corresponding to static content and moving content in a video clip, respectively. The static contexts are well exploited in image semantic segmentation by learning multi-scale and global/long-range features. The motional contexts are studied in previous video semantic segmentation. However, there is no research about how to simultaneously learn static and motional contexts which are highly correlated and complementary to each other. To address this problem, we propose a Coarse-to-Fine Feature Mining (CFFM) technique to learn a unified presentation of static contexts and motional contexts. This technique consists of two parts: coarse-to-fine feature assembling and cross-frame feature mining. The former operation prepares data for further processing, enabling the subsequent joint learning of static and motional contexts. The latter operation mines useful information/contexts from the sequential frames to enhance the video contexts of the features of the target frame. The enhanced features can be directly applied for the final prediction. Experimental results on popular benchmarks demonstrate that the proposed CFFM performs favorably against state-of-the-art methods for video semantic segmentation.

![block images](https://github.com/GuoleiSun/VSS-CFFM/blob/main/resources/diagram.png)

Authors: [Guolei Sun](https://scholar.google.com/citations?hl=zh-CN&user=qd8Blw0AAAAJ), [Yun Liu](https://yun-liu.github.io/), [Henghui Ding](https://henghuiding.github.io/), [Thomas Probst](https://probstt.bitbucket.io/), [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en).

### CFFM++
Contextual information plays a core role for video semantic segmentation (VSS). This work summarizes contexts for VSS
in two-fold: local temporal contexts (LTC) which define the contexts from neighboring frames, and global temporal contexts (GTC)
which represent the contexts from the whole video. As for LTC, it includes static and motional contexts, corresponding to static and
moving content in neighboring frames, respectively. Previously, both static and motional contexts have been studied. However, there is
no research about simultaneously learning static and motional contexts (highly complementary). Hence, we propose a Coarse-to-Fine
Feature Mining (CFFM) technique to learn a unified presentation of LTC. CFFM contains two parts: Coarse-to-Fine Feature Assembling
(CFFA) and Cross-frame Feature Mining (CFM). CFFA abstracts static and motional contexts, and CFM mines useful information
from nearby frames to enhance target features. To further exploit more temporal contexts, we propose CFFM++ by additionally
learning GTC from the whole video. Specifically, we uniformly sample certain frames from the video and extract global contextual
prototypes by k-means. The information within those prototypes is mined by CFM to refine target features. 

![block images](https://github.com/GuoleiSun/VSS-CFFM/blob/main/resources/diagram-cffm++.jpg)

Authors: [Guolei Sun](https://scholar.google.com/citations?hl=zh-CN&user=qd8Blw0AAAAJ), [Yun Liu](https://yun-liu.github.io/), [Henghui Ding](https://henghuiding.github.io/), [Min Wu](https://sites.google.com/site/wumincf/), [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en).

## Installation
Please follow the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```timm==0.3.0, CUDA11.0, pytorch==1.7.1, torchvision==0.8.2, mmcv==1.3.0, opencv-python==4.5.2```

Download this repository and install by:
```
cd VSS-CFFM && pip install -e . --user
```

## Usage: CFFM
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
cd VSS-CFFM
mkdir -p data/vspw/
ln -s /dataset_path/VSPW_480p data/vspw/
```

### Test
1. Download the trained weights (CFFM) from [here](https://drive.google.com/drive/folders/1YD5Yy6_m3QlS72o6FQWmsFtz7Kw-8OmI?usp=sharing).
2. Run the following commands:
```
# Multi-gpu testing
./tools/dist_test.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2.160k.py /path/to/CFFM_checkpoint_file <GPU_NUM> \
--out /path/to/save_results/res.pkl
```

### Training
1. Download `weights` 
(
[google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) | 
[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ)
) 
pretrained on ImageNet-1K (provided by SegFormer), and put them in a folder ```pretrained/```.

2. Training requires 4 Nvidia GPUs, each of which has > 20G GPU memory.
```
# Multi-gpu training
./tools/dist_train.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2.160k.py 4 --work-dir model_path/vspw2/work_dirs_4g_b1
```
## Usage: CFFM++
### Data preparation
The same dataset as CFFM is used. Before trying CFFM++, please first try CFFM to get familiar with the procedures.

### Test
1. Download the trained weights and global conextural prototypes (cluster centers) from [here](https://drive.google.com/drive/folders/1BzwaR6V771TjKlJ3-E_WtgJRdOExsYNh?usp=sharing). For each backbone, there are two files: a trained weight and a prototypes file containing the global contextual prototypes for all videos.
2. Run the following commands:
```
# Multi-gpu testing, take MiT-B1 as an example 
unzip cluster_centers_b1_100.zip
./tools/dist_test.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2_fine_w_proto.40k.py /path/to/CFFM++_checkpoint_file <GPU_NUM> \
--out /path/to/save_results/res.pkl
```
### Training
For training CFFM++, you need to have a CFFM trained model ready since CFFM++ is built upon CFFM. 
1. Generate global contextual prototypes using CFFM model (take MiT-B1 as an example) 
```
./tools/dist_test.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2_gene_prototype.py  /path/to/CFFM_checkpoint_file 4 --out /path/to/output/res.pkl --eval None
```
After running the above command, the prototypes per video will be save in './cluster_centers/'. You could also skip this step and directly use the prototypes generated by us, which could be downloaded from [here](https://drive.google.com/drive/folders/1BzwaR6V771TjKlJ3-E_WtgJRdOExsYNh?usp=sharing).

2. Finetuning and obtaining CFFM++ model: requires 4 Nvidia GPUs, each of which has > 20G GPU memory. Take MiT-B1 as an example.
```
./tools/dist_train.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2_fine_w_proto.40k.py 4 --load-from /path/to/CFFM_checkpoint_file  \
--work-dir model_path/vspw2/work_dirs_4g_b1_CFFM++
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
If you use our materials, please consider citing:
```bibtex
@inproceedings{sun2022vss,
    title={Coarse-to-Fine Feature Mining for Video Semantic Segmentation},
    author={Sun, Guolei and Liu, Yun and Ding, Henghui and Probst, Thomas and Van Gool, Luc},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision and Patern Recognition (CVPR)},
    year={2022}
}

@article{sun2024learning,
    title={Learning Local and Global Temporal Contexts for Video Semantic Segmentation},
    author={Sun, Guolei and Liu, Yun and Ding, Henghui and Wu, Min and Van Gool, Luc},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2024}
}
```
## Contact
- Guolei Sun, sunguolei.kaust@gmail.com
- Yun Liu, yun.liu@vision.ee.ethz.ch
