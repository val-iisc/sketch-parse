import torch
import torch.nn as nn
import numpy as np
import pickle
import deeplab_resnet_sketchParse_r1
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys, os
import matplotlib.pyplot as plt
from docopt import docopt
import subprocess as subP

docstr = """Train ResNet-DeepLab with 1 branch on sketches of 1 super category

Usage: 
    train_r1.py [options]

Options:
    -h, --help                  Print this message
    --segnetLoss                Weigh each class differently
    --snapPrefix=<str>          Snapshot [default: NoFile]
    --GTpath=<str>              Ground truth path prefix [default: /data1/ravikiran/SketchObjPartSegmentation/data/Training_GT/chosen-all/merge/]
    --IMpath=<str>              Sketch images path prefix [default: /data1/ravikiran/SketchObjPartSegmentation/data/Training_Images/chosen-all/merge/]
    --LISTpath=<str>            Input image number list file [default: /data1/ravikiran/SketchObjPartSegmentation/data/lists/train_val_lists/train_cow_horse.txt]
    --noParts=<int>             Number of parts in the supercategory [default: 5]
    --lr=<float>                Learning Rate [default: 0.0005]
    --lambda1=<float>           Inter loss weight factor for pose task [default: 0]
    -b, --batchSize=<int>       num sample per batch [default: 1]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 1]
    --gpu0=<int>                 GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
"""


args = docopt(docstr, version='v0.1')

print(args)

cudnn.enabled = False
gpu0 = int(args['--gpu0'])

snapPrefix = 'r1_' \
            + str(float(args['--maxIter'])/1000) + 'k_'\
            + 'bs' + args['--batchSize'] + '_' \
            + 'lr' +  ('{:.2e}'.format( float(args['--lr']) ))  + '_' \

if not args['--segnetLoss']:
    snapPrefix = snapPrefix + 'VCELoss_'

if args['--snapPrefix'] != 'NoFile':
    snapPrefix = args['--snapPrefix'] 

print snapPrefix


def find_med_frequency(img_list,max_):
    """This function returns parameters used to calculate the segnet loss weights
    """
    gt_path = args['--GTpath']
    dict_store = {}
    for i in range(max_):
        dict_store[i] = []
    for i,piece in enumerate(img_list):
        gt = cv2.imread(gt_path+piece+'.png')[:,:,0]
        for i in range(max_):
            dict_store[i].append(np.count_nonzero(gt == i))
    global_stats_sum = np.zeros((max_,))
    global_stats_presence = np.zeros((max_,))
    for i in range(max_):
        global_stats_sum[i] = np.sum(dict_store[i])
        global_stats_presence[i] = np.count_nonzero(dict_store[i]) 
    return global_stats_sum,global_stats_presence


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size,size))
    label_resized[:,:,:,:] = interp(Variable(torch.from_numpy(label.transpose(3,2,0,1)))).data.numpy().transpose(2,3,0,1)
    return label_resized


mirrorMap = {}
mirrorMap[2] = 6
mirrorMap[3] = 4
mirrorMap[1] = 1
mirrorMap[4] = 3
mirrorMap[5] = 5
mirrorMap[6] = 2
mirrorMap[7] = 8
mirrorMap[8] = 7


# Load pose labels
HCpose = {}
with open('sketch_pose/Pose_all_label.txt', 'r') as f:
    for line in f:
        line = line.strip()
        imId, pose= line.split(' ')
	if imId[0] == '2': 
	    year, imId, crop = imId.split('_')
            imId = os.path.splitext(year+'_'+imId+'-'+crop)[0]
	else:
	    imId = os.path.splitext(imId)[0]
	HCpose[imId] = int(pose)


def get_data_from_chunk_v2(chunk):
    gt_path =  args['--GTpath']
    img_path = args['--IMpath']
    images = np.zeros((321,321,3,len(chunk)))
    gt = np.zeros((321,321,1,len(chunk)))
    poses = np.zeros((1, len(chunk)))
    for i, piece in enumerate(chunk):
	images[:,:,:,i] = cv2.imread(img_path+piece+'.png')
	imId, aug = piece.split('(')
	pose = int(HCpose[imId])
	# Account for flip augmentations
	if 'm' in aug: # ugly hack because poses are 0-7 in vivek's list
	    pose = mirrorMap[pose+1]
	    pose -=1
	poses[:,i] = pose
        gt[:,:,0,i] = cv2.imread(gt_path+piece+'.png')[:,:,0]

    labels = [resize_label_batch(gt,i) for i in [41,41,21,41]]
#   image shape H,W,3,batch -> batch,3,H,W
    images = images.transpose((3,2,0,1))
    images = torch.from_numpy(images).float()
    return images, labels, poses

def loss_calc_seg(out, label,gpu0,seg_weights):
    """This function returns cross entropy loss for semantic segmentation
    """
   # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    if args['--segnetLoss']:
        criterion = nn.NLLLoss2d(torch.from_numpy(seg_weights).float().cuda(gpu0))
    else:
        criterion = nn.NLLLoss2d()
    out = m(out)
    
    return criterion(out,label)


def loss_calc_pose(out, label, gpu0):
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    criterion = nn.CrossEntropyLoss()
    
    return criterion(out,label)


def lr_poly(base_lr, iter,maxIter,power):
    return base_lr*((1-float(iter)/maxIter)**(power))

def get_1x_lr_params_NOscale(model):
    b = []

    b.append(model.Scale1.conv1)
    b.append(model.Scale1.bn1)
    b.append(model.Scale1.layer1)
    b.append(model.Scale1.layer2)
    b.append(model.Scale1.layer3)
    b.append(model.Scale1.layer4)

    b.append(model.Scale2.conv1)
    b.append(model.Scale2.bn1)
    b.append(model.Scale2.layer1)
    b.append(model.Scale2.layer2)
    b.append(model.Scale2.layer3)
    b.append(model.Scale2.layer4)

    b.append(model.Scale3.conv1)
    b.append(model.Scale3.bn1)
    b.append(model.Scale3.layer1)
    b.append(model.Scale3.layer2)
    b.append(model.Scale3.layer3)
    b.append(model.Scale3.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    b = []
    b.append(model.Scale1.layer5.parameters())
    b.append(model.Scale2.layer5.parameters())
    b.append(model.Scale3.layer5.parameters())
    b.append(model.pose.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


model = getattr(deeplab_resnet_sketchParse_r1,'Res_Deeplab')()

saved_state_dict = torch.load('MS_DeepLab_resnet_pretained_VOC.pth')
old_model_dict = model.state_dict()
for i in old_model_dict:
    if (i in old_model_dict.keys()) and (i not in saved_state_dict.keys()):
        saved_state_dict[i] = old_model_dict[i]

model.load_state_dict(saved_state_dict)

## Training hyper params

maxIter = int(args['--maxIter'])
batch_size = int(args['--batchSize'])
base_lr = float(args['--lr']) 
lambda1 = float(args['--lambda1'])
iterSize = int(args['--iterSize'])
model.float()
model.eval()


img_list = read_file(args['--LISTpath'])
data_list = []
for i in range(100):
    np.random.shuffle(img_list)
    data_list.extend(img_list)

# Calculate SegNet Loss weights
global_stats_sum,global_stats_presence = find_med_frequency(img_list,int(args['--noParts']))
freq_c = global_stats_sum/global_stats_presence
seg_weights = np.median(freq_c)/freq_c

model.cuda(gpu0)
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9)
optimizer.zero_grad()

data_gen = chunker(data_list, batch_size)

for iter in range(maxIter+1):
    chunk = data_gen.next()
    images, label, pose = get_data_from_chunk_v2(chunk)

    images = Variable(images).cuda(gpu0)

    out = model(images)
	
    loss = loss_calc_seg(out[0], label[0], gpu0, seg_weights)
	
    for i in range(len(out)-2):
        loss = loss + loss_calc_seg(out[i+1],label[i+1],gpu0, seg_weights)
    loss = loss + lambda1*loss_calc_pose(out[-1], pose[0], gpu0)

    (loss/iterSize).backward()

    if iter %1 == 0:
        print 'iter = ',iter, 'of',maxIter,'completed, loss = ', loss.data

    if iter %iterSize  == 0:
        optimizer.step()
        lr_ = lr_poly(base_lr,iter,maxIter,0.9)
        print '(poly lr policy) learning rate',lr_
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9)
        optimizer.zero_grad()


    if iter % 1000 == 0 and iter !=0:
        print 'taking snapshot ...'
        snapPath = 'snapshots/DeepLab_RN_auxPose_' + snapPrefix +str(iter)+'.pth'
        torch.save(model.state_dict(), snapPath)

subP.call(['python eval_r1.py', '--snapPrefix', 'DeepLab_RN_auxPose_' + snapPrefix ])
