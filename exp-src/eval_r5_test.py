import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
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
from docopt import docopt

docstr = """Evaluate ResNet-DeepLab with 5 branches on sketches of 11 categories (5 super categories)

Usage: 
    eval_r5.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: NoFile]
    --testGTpath=<str>              Ground truth path prefix [default: /data1/ravikiran/SketchObjPartSegmentation/data/temp_annotation_processor/test_GT/]
    --testIMpath=<str>              Sketch images path prefix [default: /data1/ravikiran/SketchObjPartSegmentation/data/temp_annotation_processor/SVG/PNG_untouched/]
    --gpu0=<int>                GPU number [default: 0]
"""
args = docopt(docstr, version='v0.1')
print args


def get_iou(pred,gt,class_):
    print pred.shape
    print gt.shape
    assert(pred.shape == gt.shape) 
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
    return map


gpu0 = int(args['--gpu0'])
torch.cuda.set_device(gpu0) 
sketch_root = args['--testIMpath']
model = getattr(deeplab_resnet_sketchParse_r5,'Res_Deeplab')() #TODO
model.eval()
counter = 0
model.cuda()
snapPrefix= args['--snapPrefix']
for iter in range(1,21):
    saved_state_dict = torch.load('snapshots/'+snapPrefix+str(iter)+'000.pth')
    if counter==0:
	print snapPrefix
    counter+=1
    model.load_state_dict(saved_state_dict)

     
    class_list = ['airplane-4','bird-4'] #TODO
    pytorch_list = [];
    class_ious = []
    for class_selector in class_list:
	pytorch_per_class = []
	class_split = class_selector.split('-')
	class_ = class_split[0]
 	selector = int(class_split[1])
        gt_path = args['--testGTpath']+class_
        img_list = next(os.walk(gt_path))[2] 
        path = sketch_root + class_
        for i in img_list:
            print i    
            img = cv2.imread(path+'/'+i)
            kernel = np.ones((2,2),np.uint8)
           # img = cv2.erode(img[:,:,0],kernel,iterations = 1)
            img = ndimage.grey_erosion(img[:,:,0].astype(np.uint8), size=(2,2))
            img = np.repeat(img[:,:,np.newaxis],3,2)
            gt = cv2.imread(gt_path+'/'+i, 0)
            output = model([Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True).cuda(),selector])
	    interp = nn.UpsamplingBilinear2d(size=(321, 321))
            if args['--visualize']:
                output_temp = output[3].cpu().data[0].numpy()
                output_temp = output_temp.transpose(1,2,0)
                output_temp = np.argmax(output_temp,axis = 2)
                plt.subplot(1,3,1)
                plt.imshow(img)
                plt.subplot(1,3,2)
                plt.imshow(gt)
                plt.subplot(1,3,3)
                plt.imshow(output_temp)
                plt.show()
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
