#!/bin/bash
# 

cd /cluster/home/guosun/code/video-seg/SegFormer
# cd /cluster/home/celiuce/code2/SegFormer

module load pigz
mkdir -p ${TMPDIR}/datasets_temp/
tar -I pigz -xvf /cluster/work/cvl/celiuce/seg/ADEChallengeData2016.tar.gz -C ${TMPDIR}/datasets_temp/

# ls ${TMPDIR}/datasets/

model_path=/cluster/work/cvl/guosun/models/video-seg/segformer/
# model_path=/cluster/work/cvl/celiuce/video-seg/models/segformer/

rsync -aq ./ ${TMPDIR}
cd $TMPDIR


rm -r data/ade/*
ln -s ${TMPDIR}/datasets_temp/ADEChallengeData2016 data/ade/

# source /cluster/apps/local/env2lmod.sh && module load gcc/6.3.0 python_gpu/3.8.5
# source /cluster/project/cvl/admin/cvl_settings

# source /cluster/home/celiuce/det/bin/activate

source /cluster/home/guosun/envir/swav2/bin/activate


### testing
# Single-gpu testing
# python tools/test.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py \
#     ${model_path}/work_dirs_8g_4/iter_160000.pth   --out ${model_path}/work_dirs_8g_4/res.pkl

 #trained_models/segformer.b1.512x512.ade.160k.pth \

# # Multi-gpu testing
# ./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>

# # Multi-gpu, multi-scale testing
# tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM> --aug-test


### training
# Single-gpu training
# python tools/train.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py --work-dir ${model_path}/work_dirs_test3/

# Multi-gpu training, b1
./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py 8 --work-dir ${model_path}/ade/segformer_b1_aed_mit_pretrained_seg_8g

# Multi-gpu training, b2
./tools/dist_train.sh local_configs/segformer/B2/segformer.b2.512x512.ade.160k.py 8 --work-dir ${model_path}/ade/segformer_b2_aed_mit_pretrained_seg_2_8g