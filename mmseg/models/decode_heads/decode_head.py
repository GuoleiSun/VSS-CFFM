from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
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
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
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
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        # print(seg_label.shape, seg_label.min(), seg_label.max())
        # exit()
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        # print(seg_label.shape, seg_logit.shape)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        # print("here: ",loss)
        # exit()
        return loss


class BaseDecodeHead_clips(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead_clips.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 num_clips=5):
        super(BaseDecodeHead_clips, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.num_clips=num_clips

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg,batch_size, num_clips):
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
        seg_logits = self.forward(inputs,batch_size, num_clips)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, batch_size, num_clips):
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
        return self.forward(inputs, batch_size, num_clips)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""

        # print(seg_logit.shape, seg_label.shape)
        # print("sample: ", self.sampler)
        # exit()
        assert seg_logit.dim()==5 and seg_label.dim()==5

        if seg_logit.shape[1]==seg_label.shape[1]+1:
            seg_logit_ori=seg_logit[:,:-1]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-1]

            batch_size, num_clips, _, h ,w=seg_label.shape
            seg_label_ori=seg_label.reshape(batch_size*num_clips,-1,h,w)
            seg_label_lastframe=seg_label[:,-1]
        elif seg_logit.shape[1]==2*seg_label.shape[1]:
            seg_logit_ori=seg_logit[:,:-1]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-1]
            
            seg_label_repeat=torch.cat([seg_label,seg_label],1)
            seg_label_repeat=seg_label_repeat[:,:-1]
            batch_size, num_clips, _, h ,w=seg_label_repeat.shape
            seg_label_ori=seg_label_repeat.reshape(batch_size*num_clips,-1,h,w)
            seg_label_lastframe=seg_label[:,-1]
        else:
            assert (1==0)                    

        loss = dict()
        seg_logit_ori = resize(
            input=seg_logit_ori,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_logit_lastframe = resize(
            input=seg_logit_lastframe,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        seg_label_ori = seg_label_ori.squeeze(1)
        seg_label_lastframe = seg_label_lastframe.squeeze(1)

        loss['loss_seg'] = 0.5*self.loss_decode(
            seg_logit_ori,
            seg_label_ori,
            weight=seg_weight,
            ignore_index=self.ignore_index)+self.loss_decode(
            seg_logit_lastframe,
            seg_label_lastframe,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit_ori, seg_label_ori)
        return loss

class BaseDecodeHead_clips_flow(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead_clips_flow.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 num_clips=5):
        super(BaseDecodeHead_clips_flow, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.num_clips=num_clips

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

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
        seg_logits = self.forward(inputs,batch_size, num_clips,img)
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
        return self.forward(inputs, batch_size, num_clips,img)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def consistency_loss(preds, gts, batch_size):
        # preds: b*k
        assert preds.dim()==4 and gts.dim()==3
        assert preds.size(2)==gts.size(1) and preds.size(3)==gts.size(2)
        dim1,h,w=preds.size(1),preds.size(2),preds.size(3)
        preds=preds.reshape(batch_size,-1,dim1,h,w)
        gts=gts.reshape(batch_size,-1,gts.size(1),gts.size(2))
        preds1=preds[:,:-1,:,:,:]
        preds2=preds[:,1:,:,:,:]
        preds_diff=preds1*preds2
        num_clips=preds.size(1)
        gts_diff=[]
        for i in range(num_clips-1):
            gts_one=gts[:,i]
            gts_one[~(gts[:,i]&gts[:,i+1])]=self.ignore_index
            gts_diff.append(gts_one)
        gts_diff=torch.stack(gts_diff,dim=1)
        preds_diff=preds_diff.reshape(-1,preds_diff.size(2),preds_diff.size(),preds_diff.size())
        loss_consis=self.loss_decode(
            preds_diff,
            gts_diff,
            weight=None,
            ignore_index=self.ignore_index)
        return loss_consis


    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""

        # print(seg_logit.shape, seg_label.shape)
        # print("sample: ", self.sampler)
        # exit()
        assert seg_logit.dim()==5 and seg_label.dim()==5

        if seg_logit.shape[1]==seg_label.shape[1]+1:     # k+1
            seg_logit_ori=seg_logit[:,:-1]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-1]

            batch_size, num_clips, _, h ,w=seg_label.shape
            seg_label_ori=seg_label.reshape(batch_size*num_clips,-1,h,w)
            seg_label_lastframe=seg_label[:,-1]
        elif seg_logit.shape[1]==seg_label.shape[1]+3:        # k+3
            # print("here")
            seg_logit_ori=seg_logit[:,:-3]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-3:]
            seg_logit_lastframe=seg_logit_lastframe.reshape(batch_size*3,-1,h,w)

            batch_size, num_clips, _, h ,w=seg_label.shape
            seg_label_ori=seg_label.reshape(batch_size*num_clips,-1,h,w)
            # seg_label_lastframe=seg_label[:,-1]
            seg_label_lastframe=torch.cat([seg_label[:,-1:], seg_label[:,-1:], seg_label[:,-1:]], 1)
            assert seg_label_lastframe.dim()==5
            seg_label_lastframe=seg_label_lastframe.reshape(batch_size*3,-1,h,w)
        elif seg_logit.shape[1]==2*seg_label.shape[1]:           # 2k
            seg_logit_ori=seg_logit[:,:-1]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-1]
            
            seg_label_repeat=torch.cat([seg_label,seg_label],1)
            seg_label_repeat=seg_label_repeat[:,:-1]
            batch_size, num_clips, _, h ,w=seg_label_repeat.shape
            seg_label_ori=seg_label_repeat.reshape(batch_size*num_clips,-1,h,w)
            seg_label_lastframe=seg_label[:,-1]
        elif seg_logit.shape[1]==2*seg_label.shape[1]+1:            # 2k+1
            # print("here")
            seg_logit_ori=seg_logit[:,:-2]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-2:]
            seg_logit_lastframe=seg_logit_lastframe.reshape(batch_size*2,-1,h,w)
            
            seg_label_repeat=torch.cat([seg_label,seg_label],1)
            seg_label_repeat=seg_label_repeat[:,:-1]
            batch_size, num_clips, _, h ,w=seg_label_repeat.shape
            seg_label_ori=seg_label_repeat.reshape(batch_size*num_clips,-1,h,w)
            seg_label_lastframe=torch.cat([seg_label[:,-1:], seg_label[:,-1:]], 1)
            assert seg_label_lastframe.dim()==5
            seg_label_lastframe=seg_label_lastframe.reshape(batch_size*2,-1,h,w)
        else:
            assert (1==0)                    

        loss = dict()
        seg_logit_ori = resize(
            input=seg_logit_ori,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_logit_lastframe = resize(
            input=seg_logit_lastframe,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        seg_label_ori = seg_label_ori.squeeze(1)
        seg_label_lastframe = seg_label_lastframe.squeeze(1)

        loss['loss_seg'] = 0.5*self.loss_decode(
            seg_logit_ori,
            seg_label_ori,
            weight=seg_weight,
            ignore_index=self.ignore_index)+self.loss_decode(
            seg_logit_lastframe,
            seg_label_lastframe,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit_ori, seg_label_ori)
        return loss

class BaseDecodeHead_clips_flow_city(nn.Module, metaclass=ABCMeta):
    """Base class for BaseDecodeHead_clips_flow.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 decoder_params=None,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 num_clips=5):
        super(BaseDecodeHead_clips_flow_city, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.num_clips=num_clips

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        normal_init(self.conv_seg, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

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
        seg_logits = self.forward(inputs,batch_size, num_clips,img)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg, batch_size, num_clips, img=None):
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
        return self.forward(inputs, batch_size, num_clips,img)

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""

        # print(seg_logit.shape, seg_label.shape)
        # print("sample: ", self.sampler)
        # exit()
        assert seg_logit.dim()==5 and seg_label.dim()==5

        if seg_logit.shape[1]==seg_label.shape[1]+1:     # k+1
            seg_logit_ori=seg_logit[:,-2:-1]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-1]

            batch_size, num_clips, _, h ,w=seg_label.shape
            seg_label_ori=seg_label[:,-1]
            seg_label_lastframe=seg_label[:,-1]
        elif seg_logit.shape[1]==seg_label.shape[1]+3:        # k+3
            assert False
            # print("here")
            seg_logit_ori=seg_logit[:,:-3]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-3:]
            seg_logit_lastframe=seg_logit_lastframe.reshape(batch_size*3,-1,h,w)

            batch_size, num_clips, _, h ,w=seg_label.shape
            seg_label_ori=seg_label.reshape(batch_size*num_clips,-1,h,w)
            # seg_label_lastframe=seg_label[:,-1]
            seg_label_lastframe=torch.cat([seg_label[:,-1:], seg_label[:,-1:], seg_label[:,-1:]], 1)
            assert seg_label_lastframe.dim()==5
            seg_label_lastframe=seg_label_lastframe.reshape(batch_size*3,-1,h,w)
        elif seg_logit.shape[1]==2*seg_label.shape[1]:           # 2k
            assert False
            seg_logit_ori=seg_logit[:,:-1]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-1]
            
            seg_label_repeat=torch.cat([seg_label,seg_label],1)
            seg_label_repeat=seg_label_repeat[:,:-1]
            batch_size, num_clips, _, h ,w=seg_label_repeat.shape
            seg_label_ori=seg_label_repeat.reshape(batch_size*num_clips,-1,h,w)
            seg_label_lastframe=seg_label[:,-1]
        elif seg_logit.shape[1]==2*seg_label.shape[1]+1:            # 2k+1
            assert False
            # print("here")
            seg_logit_ori=seg_logit[:,:-2]
            batch_size, num_clips, _, h ,w=seg_logit_ori.shape
            seg_logit_ori=seg_logit_ori.reshape(batch_size*num_clips,-1,h,w)
            seg_logit_lastframe=seg_logit[:,-2:]
            seg_logit_lastframe=seg_logit_lastframe.reshape(batch_size*2,-1,h,w)
            
            seg_label_repeat=torch.cat([seg_label,seg_label],1)
            seg_label_repeat=seg_label_repeat[:,:-1]
            batch_size, num_clips, _, h ,w=seg_label_repeat.shape
            seg_label_ori=seg_label_repeat.reshape(batch_size*num_clips,-1,h,w)
            seg_label_lastframe=torch.cat([seg_label[:,-1:], seg_label[:,-1:]], 1)
            assert seg_label_lastframe.dim()==5
            seg_label_lastframe=seg_label_lastframe.reshape(batch_size*2,-1,h,w)
        else:
            assert (1==0)                    

        loss = dict()
        seg_logit_ori = resize(
            input=seg_logit_ori,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_logit_lastframe = resize(
            input=seg_logit_lastframe,
            size=seg_label.shape[3:],
            mode='bilinear',
            align_corners=self.align_corners)

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None

        seg_label_ori = seg_label_ori.squeeze(1)
        seg_label_lastframe = seg_label_lastframe.squeeze(1)

        loss['loss_seg'] = 0.5*self.loss_decode(
            seg_logit_ori,
            seg_label_ori,
            weight=seg_weight,
            ignore_index=self.ignore_index)+self.loss_decode(
            seg_logit_lastframe,
            seg_label_lastframe,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit_ori, seg_label_ori)
        return loss


