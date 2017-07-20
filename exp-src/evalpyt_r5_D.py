import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
import caffe
import torch
import deeplab_resnet_sketchParse_r5 #TODO
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn
#import quant
#import pdb
#import matlab.engine
#eng = matlab.engine.start_matlab()
def get_iou(pred,gt,class_):
    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    max_label_dict = {'cow':4,'horse':4,'cat':4,'dog':4,'sheep':4,'bus':6,'car':5,'bicycle':4,'motorbike':4, 'bird':8, 'airplane':5}
    max_label = max_label_dict[class_]
    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()     
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)
    
        
        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    
    Aiou = np.sum(result_class[1:])/float(len(np.unique(gt))-1)
    
    return Aiou

def merge_parts(map_, i):
    if i == 4:
	map_ = change_parts(map_,7,2)
	map_ = change_parts(map_,8,5)
    return map_
def change_parts(map_,a,b):
    temp = np.where(map_==a)
    map_[temp[0],temp[1]] = b
    return map_
gpu0 = 0 
torch.cuda.set_device(gpu0) 
#caffe.set_mode_gpu()
#caffe.set_device(gpu0)
#net = caffe.Net('data/train_d1_contour1by2.prototxt', 'data/train_d1_contour1by2_iter_20000.caffemodel',caffe.TEST)
sketch_root = '/data1/ravikiran/SketchObjPartSegmentation/data/temp_annotation_processor/SVG/PNG_untouched/'
model = getattr(deeplab_resnet_sketchParse_r5,'Res_Deeplab')() #TODO
model.eval()
counter = 0
model.cuda()
prefix= 'r5_20.0k_bs1_lr5.00e-04_'
for iter in range(1,21):
    saved_state_dict = torch.load('/data1/ravikiran/pytorch-resnet-doc/snapshots/'+prefix+str(iter)+'000.pth')
    #saved_state_dict = torch.load('/data1/ravikiran/pytorch-resnet/snapshots/DeepLab_20k_GB_fix_noCUDNN_bsize1_20k_SegnetLoss_prototype_20000.pth')
    if counter==0:
	print prefix
    counter+=1
    #saved_state_dict = torch.load('/data1/ravikiran/pytorch-resnet/MS_DeepLab_resnet_tained_sketches.pth')
    model.load_state_dict(saved_state_dict)

     
    class_list = ['airplane-4','bird-4'] #TODO
    pytorch_list = [];
    class_ious = []
    for class_selector in class_list:
	pytorch_per_class = []
	class_split = class_selector.split('-')
	class_ = class_split[0]
 	selector = int(class_split[1])
        gt_path = '/data1/ravikiran/SketchObjPartSegmentation/data/temp_annotation_processor/test_GT/'+class_
        img_list = next(os.walk(gt_path))[2] 
        path = sketch_root + class_
        for i in img_list:
            
            img = cv2.imread(path+'/'+i)
            kernel = np.ones((2,2),np.uint8)
           # img = cv2.erode(img[:,:,0],kernel,iterations = 1)
            img = ndimage.grey_erosion(img[:,:,0].astype(np.uint8), size=(2,2))
            img = np.repeat(img[:,:,np.newaxis],3,2)
            gt = cv2.imread(gt_path+'/'+i)
            output = model([Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True).cuda(),selector])
            #for k in range(4):
            #    output_temp = output[k].cpu().data[0].numpy()
            #    output_temp = output_temp.transpose(1,2,0)
            #    output_temp = np.argmax(output_temp,axis = 2)
            #    plt.imshow(output_temp)
            #    plt.show()
	    interp = nn.UpsamplingBilinear2d(size=(321, 321))
	    output = interp(output[3])
            output = output.cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            output = np.argmax(output,axis = 2)
	    output = merge_parts(output, selector)
	    gt = merge_parts(gt, selector) 
            iou_pytorch = get_iou(output,gt,class_)        
            pytorch_list.append(iou_pytorch)
	    pytorch_per_class.append(iou_pytorch)
    	class_ious.append(np.sum(np.asarray(pytorch_per_class))/len(pytorch_per_class))
    print 'pytorch',iter, np.sum(np.asarray(pytorch_list))/len(pytorch_list),'per class', class_ious
