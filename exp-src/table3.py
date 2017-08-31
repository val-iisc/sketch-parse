import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
import caffe
import torch
import resnet_dilated_frozen_r5_D #TODO
import resnet_dilated_frozen_r5_D #TODO
import resnet_dilated_frozen_r5_D_pose #TODO
import resnet_dilated_frozen_r5_D_pose #TODO

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
    
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt)))
    
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
sketch_root = 'data/sketch-dataset/PNG_untouched/'
model_A = getattr(resnet_dilated_frozen_r5_D,'Res_Deeplab')() #TODO
model_B = getattr(resnet_dilated_frozen_r5_D,'Res_Deeplab')() #TODO
model_C = getattr(resnet_dilated_frozen_r5_D_pose,'Res_Deeplab')() #TODO
model_D = getattr(resnet_dilated_frozen_r5_D_pose,'Res_Deeplab')() #TODO
model_E = getattr(resnet_dilated_frozen_r5_D_pose,'Res_Deeplab')() #TODO

model_A.eval()
model_B.eval()
model_C.eval()
model_D.eval()
model_E.eval()

counter = 0
model_A.cuda()
model_B.cuda()
model_C.cuda()
model_D.cuda()
model_E.cuda()

file_data = open('pred_gt.txt').readlines()
dict_pred = {}
dict_label = {}
for i in file_data:
    i_split = i[:-1].split(' ')
    dict_pred[i_split[0]] = int(i_split[1])
    dict_label[i_split[0]] = int(i_split[2])


prefix_A= 'model_r5_C3_14000.pth' #B_r5
prefix_B= 'model_r5_C3seg2_14000.pth' #BS_r5
prefix_C= 'model_r5_p50x_D5_19000.pth' #BP_r5  
prefix_D= 'model_r5_p50x_D1_17000.pth' #BSP_r5
prefix_E= 'model_r5_p50x_D1_17000.pth' #BSP_r5 with 100% router accuracy

for iter in range(1):
    saved_state_dict_A = torch.load('/data1/ravikiran/pytorch-resnet/snapshots/'+prefix_A)
    saved_state_dict_B = torch.load('/data1/ravikiran/pytorch-resnet/snapshots/'+prefix_B)
    saved_state_dict_C = torch.load('/data1/ravikiran/pytorch-resnet/snapshots/'+prefix_C)
    saved_state_dict_D = torch.load('/data1/ravikiran/pytorch-resnet/snapshots/'+prefix_D)
    saved_state_dict_E = torch.load('/data1/ravikiran/pytorch-resnet/snapshots/'+prefix_E)

    #saved_state_dict = torch.load('/data1/ravikiran/pytorch-resnet/snapshots/DeepLab_20k_GB_fix_noCUDNN_bsize1_20k_SegnetLoss_prototype_20000.pth')
    if counter==0:
	print prefix_A
	print prefix_B
	print prefix_C
	print prefix_D
	print prefix_E

    counter+=1
    #saved_state_dict = torch.load('/data1/ravikiran/pytorch-resnet/MS_DeepLab_resnet_tained_sketches.pth')
    model_A.load_state_dict(saved_state_dict_A)
    model_B.load_state_dict(saved_state_dict_B)
    model_C.load_state_dict(saved_state_dict_C)
    model_D.load_state_dict(saved_state_dict_D)
    model_E.load_state_dict(saved_state_dict_E)

     
    class_list = ['cow-0', 'horse-0','cat-1','dog-1','sheep-1','bus-2','car-2','bicycle-3','motorbike-3','airplane-4','bird-4'] #TODO
    pytorch_list_A = []
    pytorch_list_B = []
    pytorch_list_C = []
    pytorch_list_D = []
    pytorch_list_E = []

    class_ious_A = []
    class_ious_B = []
    class_ious_C = []
    class_ious_D = []
    class_ious_E = []

    for class_selector in class_list:
	pytorch_per_class_A = []
	pytorch_per_class_B = []
	pytorch_per_class_C = []
	pytorch_per_class_D = []
	pytorch_per_class_E = []

	class_split = class_selector.split('-')
	class_ = class_split[0]
 	selector = int(class_split[1])
        gt_path = 'data/sketch-dataset/test_GT/'+class_
        img_list = next(os.walk(gt_path))[2] 
        path = sketch_root + class_
        for i in img_list:
            
            img = cv2.imread(path+'/'+i)
            kernel = np.ones((2,2),np.uint8)
           # img = cv2.erode(img[:,:,0],kernel,iterations = 1)
            img = ndimage.grey_erosion(img[:,:,0].astype(np.uint8), size=(2,2))
            img = np.repeat(img[:,:,np.newaxis],3,2)
            gt = cv2.imread(gt_path+'/'+i)
	    selector_pred = dict_pred[i]
            output_A = model_A([Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True).cuda(),selector_pred])
            output_B = model_B([Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True).cuda(),selector_pred])
            output_C = model_C([Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True).cuda(),selector_pred])
            output_D = model_D([Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True).cuda(),selector_pred])
	    output_E = model_E([Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True).cuda(),selector])

            #for k in range(4):
            #    output_temp = output[k].cpu().data[0].numpy()
            #    output_temp = output_temp.transpose(1,2,0)
            #    output_temp = np.argmax(output_temp,axis = 2)
            #    plt.imshow(output_temp)
            #    plt.show()
	    interp = nn.UpsamplingBilinear2d(size=(321, 321))
	    output_A = merge_parts(np.argmax(interp(output_A[3]).cpu().data[0].numpy().transpose(1,2,0),axis =2),selector_pred)
	    output_B = merge_parts(np.argmax(interp(output_B[3]).cpu().data[0].numpy().transpose(1,2,0),axis =2),selector_pred)
	    output_C = merge_parts(np.argmax(interp(output_C[3]).cpu().data[0].numpy().transpose(1,2,0),axis =2),selector_pred)
	    output_D = merge_parts(np.argmax(interp(output_D[3]).cpu().data[0].numpy().transpose(1,2,0),axis =2),selector_pred)
	    output_E = merge_parts(np.argmax(interp(output_D[3]).cpu().data[0].numpy().transpose(1,2,0),axis =2),selector)

	    gt = merge_parts(gt, selector) 
            iou_pytorch_A = get_iou(output_A,gt,class_)       
            iou_pytorch_B = get_iou(output_B,gt,class_)        
            iou_pytorch_C = get_iou(output_C,gt,class_)        
            iou_pytorch_D = get_iou(output_D,gt,class_)        
            iou_pytorch_E = get_iou(output_E,gt,class_)        

            pytorch_list_A.append(iou_pytorch_A)
            pytorch_list_B.append(iou_pytorch_B)
            pytorch_list_C.append(iou_pytorch_C)
            pytorch_list_D.append(iou_pytorch_D)
            pytorch_list_E.append(iou_pytorch_E)

	    pytorch_per_class_A.append(iou_pytorch_A)
	    pytorch_per_class_B.append(iou_pytorch_B)
	    pytorch_per_class_C.append(iou_pytorch_C)
	    pytorch_per_class_D.append(iou_pytorch_D)
	    pytorch_per_class_E.append(iou_pytorch_E)

    	class_ious_A.append(np.sum(np.asarray(pytorch_per_class_A))/len(pytorch_per_class_A))
    	class_ious_B.append(np.sum(np.asarray(pytorch_per_class_B))/len(pytorch_per_class_B))
    	class_ious_C.append(np.sum(np.asarray(pytorch_per_class_C))/len(pytorch_per_class_C))
    	class_ious_D.append(np.sum(np.asarray(pytorch_per_class_D))/len(pytorch_per_class_D))
    	class_ious_E.append(np.sum(np.asarray(pytorch_per_class_E))/len(pytorch_per_class_E))

    print 'B r5', np.sum(np.asarray(pytorch_list_A))/len(pytorch_list_A),'per class', class_ious_A
    print 'BS r5', np.sum(np.asarray(pytorch_list_B))/len(pytorch_list_B),'per class', class_ious_B
    print 'BP r5', np.sum(np.asarray(pytorch_list_C))/len(pytorch_list_C),'per class', class_ious_C
    print 'BSP r5', np.sum(np.asarray(pytorch_list_D))/len(pytorch_list_D),'per class', class_ious_D
    print 'BSP r5 with 100% classifier ', np.sum(np.asarray(pytorch_list_E))/len(pytorch_list_E),'per class', class_ious_E

