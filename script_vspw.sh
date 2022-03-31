#!/bin/bash
# 

cd /cluster/home/guosun/code/video-seg/SegFormer-code/SegFormer-shared

module load pigz
mkdir -p ${TMPDIR}/datasets_temp/
tar -I pigz -xvf /cluster/work/cvl/celiuce/video-seg/VSPW_480p.tar.gz -C ${TMPDIR}/datasets_temp/

# ls ${TMPDIR}/datasets/

model_path=/cluster/work/cvl/guosun/models/video-seg/segformer/
# model_path=/cluster/work/cvl/celiuce/video-seg/models/segformer/

# rsync -aq /cluster/home/celiuce/code2/SegFormer/ ${TMPDIR}
rsync -aq /cluster/home/guosun/code/video-seg/SegFormer-code/SegFormer-shared/ ${TMPDIR}
cd $TMPDIR

rm -r data/vspw/*
ln -s ${TMPDIR}/datasets_temp/VSPW_480p data/vspw/


# source /cluster/apps/local/env2lmod.sh && module load gcc/6.3.0 python_gpu/3.8.5
# source /cluster/project/cvl/admin/cvl_settings
# source /cluster/home/celiuce/det/bin/activate

source /cluster/home/guosun/envir/swav2/bin/activate



## Test

./tools/dist_test.sh local_configs/cffm/B0/cffm.b0.480x480.vspw2.160k.py  \
 ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-depth1-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-5/iter_160000.pth 4  \
   --out ${model_path}/vspw2/work_dirs_4g_b0_bs2_num-clips4-963-depth1-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-5/res.pkl

./tools/dist_test.sh local_configs/cffm/B1/cffm.b1.480x480.vspw2.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-6/iter_160000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b1_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-6/res.pkl

./tools/dist_test.sh local_configs/cffm/B2/cffm.b2.480x480.vspw2.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/iter_156000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b2_bs2_num-clips4-963-depth2-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/res.pkl

./tools/dist_test.sh local_configs/cffm/B5/cffm.b5.480x480.vspw2.160k.py \
  ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-depth4-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/iter_144000.pth 4 \
   --out ${model_path}/vspw2/work_dirs_4g_b5_bs2_num-clips4-963-depth4-oriWeight0.5-clips_resize1_8_focal_trans3-inverseVideoTrue-3/res.pkl
