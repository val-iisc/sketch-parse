import torch
import torch.nn as nn
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import deeplab_resnet_sketchParse_r5 
import torch.backends.cudnn as cudnn
import sys, os
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from torch.autograd import Variable
from docopt import docopt

docstr = """Train ResNet-DeepLab with 5 branches on sketches of 11 categories (5 super categories)

Usage: 
    train_r5.py [options]

Options:
    -h, --help                  Print this message
    --segnetLoss                Weigh each class differently
    --snapPrefix=<str>          Snapshot [default: NoFile]
    --GTpath=<str>              Ground truth path prefix [default: data/gt/]
    --IMpath=<str>              Sketch images path prefix [default: data/im/]
    --LISTpath=<str>            Input image number list file [default: data/lists/]
    --lr=<float>                Learning Rate [default: 0.0005]
    --lambda1=<float>           Inter loss weight factor for pose task [default: 0]
    -b, --batchSize=<int>       num sample per batch [default: 1]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 1]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
"""


args = docopt(docstr, version='v0.1')
print args
cudnn.enabled = False
gpu0 = int(args['--gpu0'])

snapPrefix = 'r5_' \
            + str(float(args['--maxIter'])/1000) + 'k_'\
            + 'bs' + args['--batchSize'] + '_' \
            + 'lr' +  ('{:.2e}'.format( float(args['--lr']) ))  + '_' \

if not args['--segnetLoss']:
    snapPrefix = snapPrefix + 'VCELoss_'

if args['--snapPrefix'] != 'NoFile':
    snapPrefix = args['--snapPrefix'] 

print snapPrefix

cudnn.enabled = False


# Pose mirroring map
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
with open('data/lists/Pose_all_label.txt', 'r') as f:
    for line in f:
        line = line.strip()
        imId, pose= line.split(' ')
	if imId[0] == '2': 
	    year, imId, crop = imId.split('_')
            imId = os.path.splitext(year+'_'+imId+'-'+crop)[0]
	else:
	    imId = os.path.splitext(imId)[0]
	HCpose[imId] = int(pose)


def get_data_from_chunk(chunk):
    gt_path =  args['--GTpath']
    img_path = args['--IMpath']
    images = np.zeros((321,321,3,len(chunk)))
    gt = np.zeros((321,321,1,len(chunk)))
    poses = np.zeros((1,len(chunk)))
    for i,piece in enumerate(chunk):
        images[:,:,:,i] = cv2.imread(img_path+piece+'.png')
        gt[:,:,0,i] = cv2.imread(gt_path+piece+'.png')[:,:,0]
	imId, aug = piece.split('(')
	pose = int(HCpose[imId])
	# Account for flip augmentations
	if 'm' in aug: # ugly hack because poses are 0-7 in vivek's list
	    pose = mirrorMap[pose+1]
	    pose -=1
	poses[:,i] = pose

    labels = [resize_label_batch(gt,i) for i in [41,41,21,41]]
#   image shape H,W,3,batch -> batch,3,H,W
    images = images.transpose((3,2,0,1))
    images = torch.from_numpy(images).float()
    return images, labels, poses

def find_med_frequency(img_list,max_):
    """
    This function returns parameters used to calculate the segnet loss weights
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

def read_txt_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def read_file(path_to_file,i):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append((line[:-1],i))
    return img_list

def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size,size))
    label_resized[:,:,:,:] = interp(Variable(torch.from_numpy(label.transpose(3,2,0,1)))).data.numpy().transpose(2,3,0,1)
    return label_resized



def loss_calc_seg(out, label,gpu0,seg_weights):
    """
    This function returns cross entropy loss for semantic segmentation
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
    """
    This function returns loss for the pose auxilary task
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    criterion = nn.CrossEntropyLoss()
    
    return criterion(out,label)

def lr_poly(base_lr, iter,maxIter,power):
    return base_lr*((1-float(iter)/maxIter)**(power))

def get_im_label(files):
    img_list = []
    for i,file_ in enumerate(files):
        img_list.extend(read_file(args['--LISTpath']+'train_'+file_+'.txt',i))
    return img_list
            
def get_1x_lr_params_NObn_double(model_double):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm, 
    requires_grad is False, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model_double.Scale1.conv1)
    b.append(model_double.Scale1.bn1)
    b.append(model_double.Scale1.layer1)
    b.append(model_double.Scale1.layer2)
    b.append(model_double.Scale1.layer3)
    b.append(model_double.Scale1.layer4)
    b.append(model_double.Scale1.layer4_5_r0)
    b.append(model_double.Scale1.layer4_5_r1)
    b.append(model_double.Scale1.layer4_5_r2)
    b.append(model_double.Scale1.layer4_5_r3)
    b.append(model_double.Scale1.layer4_5_r4)


    b.append(model_double.Scale2.conv1)
    b.append(model_double.Scale2.bn1)
    b.append(model_double.Scale2.layer1)
    b.append(model_double.Scale2.layer2)
    b.append(model_double.Scale2.layer3)
    b.append(model_double.Scale2.layer4)
    b.append(model_double.Scale2.layer4_5_r0)
    b.append(model_double.Scale2.layer4_5_r1)
    b.append(model_double.Scale2.layer4_5_r2)
    b.append(model_double.Scale2.layer4_5_r3)
    b.append(model_double.Scale2.layer4_5_r4)

    b.append(model_double.Scale3.conv1)
    b.append(model_double.Scale3.bn1)
    b.append(model_double.Scale3.layer1)
    b.append(model_double.Scale3.layer2)
    b.append(model_double.Scale3.layer3)
    b.append(model_double.Scale3.layer4)
    b.append(model_double.Scale3.layer4_5_r0)
    b.append(model_double.Scale3.layer4_5_r1)
    b.append(model_double.Scale3.layer4_5_r2)
    b.append(model_double.Scale3.layer4_5_r3)
    b.append(model_double.Scale3.layer4_5_r4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params_double(model_double):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model_double.Scale1.layer5_r0.parameters())
    b.append(model_double.Scale2.layer5_r0.parameters())
    b.append(model_double.Scale3.layer5_r0.parameters())

    b.append(model_double.Scale1.layer5_r1.parameters())
    b.append(model_double.Scale2.layer5_r1.parameters())
    b.append(model_double.Scale3.layer5_r1.parameters())

    b.append(model_double.Scale1.layer5_r2.parameters())
    b.append(model_double.Scale2.layer5_r2.parameters())
    b.append(model_double.Scale3.layer5_r2.parameters())

    b.append(model_double.Scale1.layer5_r3.parameters())
    b.append(model_double.Scale2.layer5_r3.parameters())
    b.append(model_double.Scale3.layer5_r3.parameters())

    b.append(model_double.Scale1.layer5_r4.parameters())
    b.append(model_double.Scale2.layer5_r4.parameters())
    b.append(model_double.Scale3.layer5_r4.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def get_50x_lr_params_double(model_double):
    b = []
    b.append(model_double.pose_r0.parameters())
    b.append(model_double.pose_r1.parameters())
    b.append(model_double.pose_r2.parameters())
    b.append(model_double.pose_r3.parameters())
    b.append(model_double.pose_r4.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


if not os.path.exists('data/snapshots'):
    os.makedirs('data/snapshots')

#############
model_double = getattr(deeplab_resnet_sketchParse_r5,'Res_Deeplab')()
saved_state_dict = torch.load('MS_DeepLab_resnet_pretained_VOC.pth')
old_dict = model_double.state_dict()
############  net surgery for the model with 5 branches
for i in old_dict.keys():   
    if i in saved_state_dict:                   # for the common branches
	    old_dict[i] = saved_state_dict[i]
	    #print i
    else:                                       # for specialist branches for each supercategory
	    i_split = i.split('.')
	    if i_split[1][:-1] == 'layer4_5_r': 
		i_split_copy = i_split
		i_split_copy[1] = i_split_copy[1][:-5]
		i_split_copy[2] = str(int(i_split_copy[2])+2)
		old_dict[i] = saved_state_dict['.'.join(i_split_copy)]

	    if (i_split[1][:] == 'layer5_r0' or i_split[1][:] == 'layer5_r1' or  i_split[1][:] == 'layer5_r3') and i_split[:-1]=='weight':
	        i_split_copy = i_split
	        i_split_copy[1] = i_split_copy[1][:-3]
	        #old_dict[i] = saved_dict_temp['.'.join(i_split_copy)]   #TODO
            #print i, '.'.join(i_split_copy)
	    if i_split[1][:] == 'layer5_r2' and  i_split[:-1]=='weight':  #7
	        i_split_copy = i_split
	        i_split_copy[1] = i_split_copy[1][:-3]
	        #old_dict[i] = torch.from_numpy(np.load('7.npy'))        #TODO
	    if i_split[1][:] == 'layer5_r4' and  i_split[:-1]=='weight': #9
	        i_split_copy = i_split
	        i_split_copy[1] = i_split_copy[1][:-3]
	        #old_dict[i] = torch.from_numpy(np.load('9.npy'))       #TODO

model_double.load_state_dict(copy.deepcopy(old_dict))
#############
base_lr = float(args['--lr']) 
lambda1 = float(args['--lambda1'])
batch_size = int(args['--batchSize'])
iterSize = int(args['--iterSize'])
maxIter = int(args['--maxIter'])

model_double.float()
model_double.eval()
###############

# TODO Change this for custom data
train_txt_list = ['cow_horse','cat_dog_sheep','bus_car','bicycle_motorbike','aeroplane_bird'] #to read .txt files to make train list
no_parts = [5,5,7,5,9]
img_list = get_im_label(train_txt_list)

seg_weights = {}
for temp1 in range(len(train_txt_list)):
    global_stats_sum,global_stats_presence = find_med_frequency(read_txt_file(args['--LISTpath']+'train_'+ train_txt_list[temp1]+ '.txt'),no_parts[temp1])
    freq_c = global_stats_sum/global_stats_presence
    seg_weights[temp1] = np.median(freq_c)/freq_c

data_list = []
for i in range(100):
    np.random.shuffle(img_list)
    data_list.extend(img_list)

# Send model to GPU
model_double.cuda(gpu0)

# Init optimizer
optimizer_double = optim.SGD([{'params': get_1x_lr_params_NObn_double(model_double), 'lr': base_lr }, {'params': get_10x_lr_params_double(model_double), 'lr': 10*base_lr}, {'params': get_50x_lr_params_double(model_double), 'lr': 50*base_lr} ], lr = base_lr, momentum = 0.9)
optimizer_double.zero_grad()
data_gen = chunker(data_list, batch_size)

for iter in range(maxIter+1):
    chunk = data_gen.next()
    images, label, pose = get_data_from_chunk([chunk[0][0]])
    selector = chunk[0][1]
    images_1 = Variable(images).cuda(gpu0)

    out_double = model_double([images_1,selector])

    loss_double = loss_calc_seg(out_double[0], label[0],gpu0,seg_weights[selector])

    for i in range(len(out_double)-2):  # do not iterate over pose output (last element in output list)
        loss_double = loss_double + loss_calc_seg(out_double[i+1],label[i+1],gpu0,seg_weights[selector])
    logseg = loss_double.data.cpu().numpy()
    print iter, 'loss (Seg) = ', logseg

    loss_double = loss_double + lambda1*loss_calc_pose(out_double[-1], pose[0], gpu0)
    print iter, 'loss (Pose) = ', loss_double.data.cpu().numpy() - logseg 


    (loss_double/iterSize).backward()


    lr_ = lr_poly(base_lr,iter,maxIter,0.9)
    print '(poly lr policy) learning rate',lr_
    optimizer_double = optim.SGD([{'params': get_1x_lr_params_NObn_double(model_double), 'lr': lr_ }, {'params': get_10x_lr_params_double(model_double), 'lr': 10*lr_},{'params': get_50x_lr_params_double(model_double), 'lr': 50*lr_}  ], lr = lr_, momentum = 0.9)

    if iter%iterSize==0:
	optimizer_double.step()
	optimizer_double.zero_grad()
    
    if iter%1000==0 and iter!=0:
        snapPath = os.path.join('data/snapshots', snapPrefix + str(iter) + '.pth')
        torch.save(model_double.state_dict(),snapPath)

