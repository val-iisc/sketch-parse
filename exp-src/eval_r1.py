import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import torch
import deeplab_resnet_sketchParse_r1
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
from docopt import docopt

docstr = """Evaluate ResNet-DeepLab with 5 branches on sketches of 11 categories (5 super categories)

Usage: 
    eval_r5.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: NoFile]
    --testGTpath=<str>              Ground truth path prefix [default: data/sketch-dataset/test_GT/]
    --testIMpath=<str>              Sketch images path prefix [default: data/sketch-dataset/PNG_untouched/]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt(docstr, version='v0.1')
print args

def get_iou(pred,gt):
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = 4
    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)
    
        
        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt)))
    
    return Aiou



gpu0 = int(args['--gpu0']) 
sketch_root = args['--testIMpath']
model = getattr(deeplab_resnet_sketchParse_r1,'Res_Deeplab')()
model.eval()
counter = 0
model.cuda(gpu0)
snapPrefix= args['--snapPrefix']
for iter in range(1,20):
    saved_state_dict = torch.load('data/snapshots/'+snapPrefix+str(iter)+'000.pth')
    if counter==0:
	print snapPrefix
    counter+=1
    model.load_state_dict(saved_state_dict)

     
    class_list = ['cow', 'horse']
    pytorch_list = [];
    for class_ in class_list:
        gt_path = args['--testGTpath']+class_

        img_list = next(os.walk(gt_path))[2] 
        path = sketch_root + class_
        for i in img_list:
            
            img = cv2.imread(path+'/'+i)
            img = ndimage.grey_erosion(img[:,:,0].astype(np.uint8), size=(2,2))
            img = np.repeat(img[:,:,np.newaxis],3,2)
            gt = cv2.imread(gt_path+'/'+i, 0)
            output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(), volatile=True).cuda(gpu0))
            interp = nn.UpsamplingBilinear2d(size=(321, 321))

            if args['--visualize']:
                output_temp = interp(output[3]).cpu().data[0].numpy()
                output_temp = output_temp.transpose(1,2,0)
                output_temp = np.argmax(output_temp,axis = 2)
                plt.subplot(1,3,1)
                plt.imshow(img)
                plt.subplot(1,3,2)
                plt.imshow(gt)
                plt.subplot(1,3,3)
                plt.imshow(output_temp)
                plt.show()
               
            output = interp(output[3]).cpu().data[0].numpy()
            output = output.transpose(1,2,0)
            output = np.argmax(output,axis = 2)
            iou_pytorch = get_iou(output,gt)       
            cv2.imwrite('test/output.png',output)
            cv2.imwrite('test/gt.png',gt)
            pytorch_list.append(iou_pytorch)

    print 'pytorch',iter, np.sum(np.asarray(pytorch_list))/len(pytorch_list)
