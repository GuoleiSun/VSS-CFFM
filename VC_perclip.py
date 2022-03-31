import numpy as np
import os
from PIL import Image
#from utils import Evaluator
import sys

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def beforeval(self):
        isval = np.sum(self.confusion_matrix,axis=1)>0
        self.confusion_matrix = self.confusion_matrix*isval

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc


    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        isval = np.sum(self.confusion_matrix,axis=1)>0
        MIoU = np.nansum(MIoU*isval)/isval.sum()
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        #print(mask)
        #print(gt_image.shape)
        #print(gt_image[mask])
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
#        print(label.shape)
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



def get_common(list_,predlist,clip_num,h,w):
    accs = []
    for i in range(len(list_)-clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))

                 
        for j in range(1,clip_num):
            common = (list_[i] == list_[i+j])
            global_common = np.logical_and(global_common,common)
            pred_common = (predlist[i]==predlist[i+j])
            predglobal_common = np.logical_and(predglobal_common,pred_common)
        pred = (predglobal_common*global_common)

        acc = pred.sum()/global_common.sum()
        accs.append(acc)
    return accs
        


DIR='data/vspw//VSPW_480p'

Pred='./models/vspw2/result_submission'
split = 'val.txt'

with open(os.path.join(DIR,split),'r') as f:
    lines = f.readlines()
    for line in lines:
        videolist = [line[:-1] for line in lines]
total_acc=[]
total_acc_8=[]

clip_num=16
clip_num_8=8

num_class=124    # change this when necessary
evaluator = Evaluator(num_class)
evaluator.reset()
evaluator_video = Evaluator(num_class)
evaluator_video.reset()
good_video=[]
for video in videolist:
    evaluator_video.reset()
    if video[0]=='.':
        continue
    imglist = []
    predlist = []

    images = sorted(os.listdir(os.path.join(DIR,'data',video,'mask')))

    if len(images)<=clip_num:
        print("here: ", video)
        continue
    for imgname in images:
        if imgname[0]=='.':
            continue
        img = Image.open(os.path.join(DIR,'data',video,'mask',imgname))
        w,h = img.size
        img = np.array(img)
        ## added by guolei
        img[img==0]=255
        img = img-1
        img[img==254]=255

        imglist.append(img)
        pred = Image.open(os.path.join(Pred,video,imgname))
        pred = np.array(pred)
        predlist.append(pred)
        evaluator.add_batch(img[None,:], pred[None,:])
        evaluator_video.add_batch(img[None,:], pred[None,:])
        # print(img[None,:].shape, pred[None,:].shape)
    
    if evaluator_video.Mean_Intersection_over_Union()>0.8:
        good_video.append(video)
    accs = get_common(imglist,predlist,clip_num,h,w)
    print(sum(accs)/len(accs))
    accs_8 = get_common(imglist,predlist,clip_num_8,h,w)
    print(sum(accs_8)/len(accs_8))
    total_acc.extend(accs)
    total_acc_8.extend(accs_8)
Acc = np.array(total_acc)
Acc = np.nanmean(Acc)
Acc_8 = np.array(total_acc_8)
Acc_8 = np.nanmean(Acc_8)
print(Pred)
print('*'*10)
print('VC{} score: {} on {} set'.format(clip_num,Acc,split))
print('VC{} score: {} on {} set'.format(clip_num_8,Acc_8,split))
print('*'*10)

Acc = evaluator.Pixel_Accuracy()
Acc_class = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
print("Acc, Acc_class, mIoU, FWIoU: ", [Acc, Acc_class, mIoU, FWIoU])


print(good_video)
