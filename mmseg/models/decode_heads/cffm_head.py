import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips, BaseDecodeHead_clips_flow, BaseDecodeHead_clips_flow_city
from mmseg.models.utils import *
import attr

from IPython import embed
from .cffm_module.cffm_transformer import BasicLayer3d3

import cv2
from fast_pytorch_kmeans import KMeans
import os
from .pvt.swin_transformer_2d import BasicLayer_cluster
from .pvt.pvt import BasicLayer_cluster_pvt
import torch.nn.functional as F
import os
import glob
import random

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class CFFMHead_clips_resize1_8(BaseDecodeHead_clips_flow):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(CFFMHead_clips_resize1_8, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)

        depths = decoder_params['depths']

        self.decoder_focal=BasicLayer3d3(dim=embedding_dim,
               depth=depths,
               num_heads=8,
               window_size=7,
               mlp_ratio=4.,
               qkv_bias=True, 
               qk_scale=None,
               drop=0., 
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm, 
               pool_method='fc',
               downsample=None,
               focal_level=2, 
               focal_window=5, 
               expand_size=3, 
               use_conv_embed=False,
               use_shift=False, 
               use_pre_norm=False, 
               use_checkpoint=False, 
               focal_l_clips=[1,2,3],
               focal_kernel_clips=[7,5,3])

        print(self.decoder_focal.blocks[0].focal_kernel_clips)

    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None):
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w)

        # print("_c.shape: ", _c.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]

        h2=int(h/2)
        w2=int(w/2)
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)

        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)
        # print(_c_further.shape)
        # exit()
        _c2=self.decoder_focal(_c_further)
        # _c_further=_c_further.permute(0,2,1,3,4)

        # _c2=_c.reshape(batch_size, num_clips, -1, h, w)

        assert _c_further.shape==_c2.shape

        _c_further2=torch.cat([_c_further[:,-1], _c2[:,-1]],1)

        x2 = self.dropout(_c_further2)
        x2 = self.linear_pred2(x2)
        x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False)
        x2=x2.unsqueeze(1)

        x=torch.cat([x,x2],1)   ## b*(k+1)*124*h*w

        if not self.training:
            return x2.squeeze(1)

        return x


@HEADS.register_module()
class CFFMHead_clips_resize1_8_gene_prototype(BaseDecodeHead_clips_flow):
    """
    generate the cluster centers for each video
    """
    def __init__(self, feature_strides, **kwargs):
        super(CFFMHead_clips_resize1_8_gene_prototype, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)

        depths = decoder_params['depths']

        self.decoder_focal=BasicLayer3d3(dim=embedding_dim,
               depth=depths,
               num_heads=8,
               window_size=7,
               mlp_ratio=4.,
               qkv_bias=True, 
               qk_scale=None,
               drop=0., 
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm, 
               pool_method='fc',
               downsample=None,
               focal_level=2, 
               focal_window=5, 
               expand_size=3, 
               use_conv_embed=False,
               use_shift=False, 
               use_pre_norm=False, 
               use_checkpoint=False, 
               focal_l_clips=[1,2,3],
               focal_kernel_clips=[7,5,3])

        self.n_clusters=100   # default 100
        self.save_path='./cluster_centers/'

        print(self.decoder_focal.blocks[0].focal_kernel_clips)

    def forward_test(self, inputs, img_metas, test_cfg, batch_size=None, num_clips=None, img=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, batch_size, num_clips, img, img_metas)

    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None, img_metas=None):
        # print(img_metas);exit()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w)

        ## clustering
        print("batch_size: ",batch_size,_c.shape)
        assert batch_size==1
        h2=int(h/2)  # 1/8
        w2=int(w/2)  #1/8
        _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)
        _c2=_c2.reshape(batch_size, num_clips, -1, h2, w2)
        _c_cluster=_c2.permute(0,1,3,4,2)
        _c_cluster=_c_cluster.reshape(batch_size, num_clips*h2*w2,-1)   # B, num_clips*h*w, c
        with torch.no_grad():
            centers=[]
            for ii in range(batch_size):
                # print("k-means")
                kmeans = KMeans(n_clusters=self.n_clusters, max_iter=10, mode='euclidean', verbose=0)
                labels = kmeans.fit_predict(_c_cluster[ii])
                centers.append(kmeans.centroids)

            centers=torch.stack(centers,dim=0)
            assert centers.shape[0]==batch_size

        ## save clustering centers
        video_name=img_metas[0]['filename']
        video_name=video_name.split('/')[-3]
        # print(centers.shape, img_metas, video_name)
        path_centers=self.save_path+video_name
        if not os.path.exists(path_centers):
            os.makedirs(path_centers)
        torch.save(centers, path_centers+'/centers.pt')


        # print("_c.shape: ", _c.shape)
        # if not self.training and num_clips!=self.num_clips:
        if not self.training:
            return x[:,-1]


@HEADS.register_module()
class CFFMHead_clips_resize1_8_finetune_w_prototype3(BaseDecodeHead_clips_flow):
    """
    fine-tune the cffm with the cluster center (prototype)
    """
    def __init__(self, feature_strides, **kwargs):
        super(CFFMHead_clips_resize1_8_finetune_w_prototype3, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)

        self.linear_pred3 = nn.Conv2d(embedding_dim*1, self.num_classes, kernel_size=1)

        depths = decoder_params['depths']

        self.decoder_focal=BasicLayer3d3(dim=embedding_dim,
               depth=depths,
               num_heads=8,
               window_size=7,
               mlp_ratio=4.,
               qkv_bias=True, 
               qk_scale=None,
               drop=0., 
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm, 
               pool_method='fc',
               downsample=None,
               focal_level=2, 
               focal_window=5, 
               expand_size=3, 
               use_conv_embed=False,
               use_shift=False, 
               use_pre_norm=False, 
               use_checkpoint=False, 
               focal_l_clips=[1,2,3],
               focal_kernel_clips=[7,5,3])

        self.n_clusters=10   # default 100
        self.save_path='./cluster_centers/'

        print(self.decoder_focal.blocks[0].focal_kernel_clips)

        depths=1
        self.dropout3=nn.Dropout2d(self.dropout_ratio)
        self.decoder_swin=BasicLayer_cluster(
                dim=embedding_dim,
                depth=depths,
                num_heads=8,
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False)
        self.finetune=True
        print("self.finetune: ", self.finetune)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg,batch_size, num_clips,img=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, batch_size, num_clips, img, img_metas)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, batch_size=None, num_clips=None, img=None):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, batch_size, num_clips, img, img_metas)

    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None, img_metas=None):
        ## load centers
        # print(img_metas);exit()
        random_selected=True
        threshold=0.8
        assert batch_size==len(img_metas)
        centers=[]
        for i in range(len(img_metas)):
          video_name=img_metas[i]['filename']
          video_name=video_name.split('/')[-3]
          path_centers=self.save_path+video_name+'/centers.pt'
          if os.path.isfile(path_centers):
            centers.append(torch.load(path_centers, map_location=torch.device('cpu')))
          else:
            path_centers=glob.glob(self.save_path+video_name+'/*.pt')
            centers_i=[]
            for p in path_centers:
              centers_i.append(torch.load(p, map_location=torch.device('cpu')))
            centers_i=torch.cat(centers_i,dim=1)
            # centers_i=torch.cat([random.choice(centers_i)],dim=1)
            # centers_i=torch.cat([centers_i[1]],dim=1)
            if random_selected:
              assert centers_i.dim()==3 and centers_i.shape[0]==1, centers_i.shape
              centers_i=centers_i.squeeze(0)
              mask=torch.rand((centers_i.shape[0]))
              index=torch.topk(mask,int(centers_i.shape[0]*threshold))[1]
              mask=torch.zeros((centers_i.shape[0]))
              mask[index]=1
              centers_i=centers_i[mask>0.5]
              centers_i=centers_i.unsqueeze(0)
            centers.append(centers_i)
            # print(centers_i.shape)
        centers=torch.cat(centers,dim=0).cuda()
        # print(centers.shape)

        # print(img_metas);exit()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        with torch.no_grad():
          self.linear_fuse.eval()
          _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w)

        # print("_c.shape: ", _c.shape)
        # x_ori=x.clone()
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]

        h2=int(h/2)
        w2=int(w/2)
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)

        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)
        # print(_c_further.shape)
        # exit()
        _c2=self.decoder_focal(_c_further)
        # _c_further=_c_further.permute(0,2,1,3,4)

        # _c2=_c.reshape(batch_size, num_clips, -1, h, w)

        assert _c_further.shape==_c2.shape

        _c_further2=torch.cat([_c_further[:,-1], _c2[:,-1]],1)
        x2 = self.dropout(_c_further2)
        x2 = self.linear_pred2(x2)
        x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False)
        x2=x2.unsqueeze(1)

        ## learning from clustering centers
        if self.finetune:
            _c_further=_c_further.detach()
            x=x.detach()
            _c2=_c2.detach()
            x2=x2.detach()
        _c3,_,_,_,_,_=self.decoder_swin(_c_further[:,-1].permute(0,2,3,1).reshape(batch_size,h2*w2,-1), h2, w2, centers)
        _c3=_c3.reshape(batch_size,h2,w2,-1).permute(0,3,1,2)
        _c_further3=torch.cat([_c3],1)
        x3=self.dropout3(_c_further3)
        x3=self.linear_pred3(x3)
        x3=resize(x3,size=(h,w),mode='bilinear',align_corners=False)
        x3=x3.unsqueeze(1)

        # x=torch.cat([x,x2],1)   ## b*(k+1)*124*h*w
        x=torch.cat([x,x3],1)   ## b*(k+1)*124*h*w

        if not self.training:
            # return x2.squeeze(1)
            return x2.squeeze(1)+0.5*x3.squeeze(1)
            # return x_ori[:,-1]+0.5*x3.squeeze(1)

        return x

