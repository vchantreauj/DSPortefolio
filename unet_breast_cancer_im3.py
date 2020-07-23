from PIL import Image
from skimage import data, io, filters
from skimage.transform import resize
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

im = io.imread_collection('data/breast_cancer_cell_seg/Images/*.tif', plugin = 'tifffile')
mask = io.imread_collection('data/breast_cancer_cell_seg/Masks/*.TIF', plugin = 'tifffile')

width = 764
height = 892
width_out = 676
height_out = 804

class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
                    # conv2d Hout = Hin - k + 1
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel), # normalization over 4D input
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                    # convtranspose2d Hout = 2Hin - 3 + k
                    )
            return  block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
            block = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(out_channels),
                    )
            return  block
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(512),
                            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
                            torch.nn.ReLU(),
                            torch.nn.BatchNorm2d(512),
                            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
        
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1) # concatenate on row (add element on line)
    
    def forward(self, x):
        # Encode
        #print('shape x',x.shape)
        encode_block1 = self.conv_encode1(x)
        #print('encode_block1',encode_block1.shape)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        #print('encode_pool1',encode_pool1.shape)
        encode_block2 = self.conv_encode2(encode_pool1)
        #print('encode_block2',encode_block2.shape)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        #print('encode_pool2',encode_pool2.shape)
        encode_block3 = self.conv_encode3(encode_pool2)
        #print('encode_block3',encode_block3.shape)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        #print('bottleneck1',bottleneck1.shape)
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        #print('decode_block3',decode_block3.shape)
        cat_layer2 = self.conv_decode3(decode_block3)
        #print('catlayer2',cat_layer2.shape)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        #print('decode_block2',decode_block2.shape)
        cat_layer1 = self.conv_decode2(decode_block2)
        #print('cat_layer1',cat_layer1.shape)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        #print('decode_block1',decode_block1.shape)
        final_layer = self.final_layer(decode_block1)
        #print('final_layer',final_layer.shape)
        return  final_layer 
 
def train_step(inputs, labels, optimizer, criterion, i):
    print('in train step')
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = unet(inputs)
    if i == 5:
        save_image(outputs[0], 'outputunet/out2307.png')
	# outputs.shape =(batch_size, n_classes, img_cols, img_rows) 
    outputs = outputs.permute(0, 2, 3, 1)
	# outputs.shape =(batch_size, img_cols, img_rows, n_classes) 
    print(outputs.shape)
    outputs = outputs.resize(batch_size*width_out*height_out, 2)
    print(outputs.shape)
    labels = labels.resize(batch_size*width_out*height_out)
    print(labels.shape)
    loss = criterion(outputs, labels)
    print('after loss',str(type(loss)))
    loss.backward()
    print('backward')
    optimizer.step()
    print('optimizer')
    return loss

def get_val_loss(x_val, y_val):
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).long()
    #if use_gpu:
    x_val = x_val.cuda()
    y_val = y_val.cuda()
    m = x_val.shape[0]
    outputs = unet(x_val)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows) 
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes) 
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = y_val.resize(m*width_out*height_out)
    loss = F.cross_entropy(outputs, labels)
    return loss.data

nb_im = 3
x_train = []
labels = []
for x in range(nb_im):
    x_train.append(im[x][:764,:892])
    labels.append(mask[x][44:720,44:848])

x_train = np.array(x_train)
x_train = resize(x_train, (x_train.shape[0], 3, x_train.shape[1], x_train.shape[2]))
#train_tens = torch.tensor(train_tens).float().cuda()
labels = np.array(labels)
labels = resize(labels, (labels.shape[0], labels.shape[1], labels.shape[2]))
#labels = torch.tensor(labels).float().cuda()

batch_size = 1 #3 #9
epochs = 4 #1000
epoch_lapse = 1 #50
#threshold = 0.5
#sample_size = None
unet = UNet(in_channel=3,out_channel=2)
unet = unet.cuda()
criterion = torch.nn.CrossEntropyLoss()
#criterion = criterion.cuda()
optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
#optimizer = optimizer.cuda()

epoch_iter = np.ceil(nb_im / batch_size).astype(int)
#t = trange(epochs, leave=True)
for _ in range(epochs):
    total_loss = 0
    for i in range(epoch_iter):
        print(i)
        #batch_train_x = torch.from_numpy(x_train[i * batch_size : (i + 1) * batch_size]).float()
        #batch_train_y = torch.from_numpy(labels[i * batch_size : (i + 1) * batch_size]).long()
        batch_train_x = torch.as_tensor(x_train[i * batch_size : (i + 1) * batch_size]).float()
        batch_train_y = torch.as_tensor(labels[i * batch_size : (i + 1) * batch_size]).long()
        batch_train_x = batch_train_x.cuda()
        batch_train_y = batch_train_y.cuda()
        batch_loss = train_step(batch_train_x , batch_train_y, optimizer, criterion,i+epochs)
        total_loss += batch_loss
        torch.cuda.empty_cache()
    #if (_+1) % epoch_lapse == 0:
    #    val_loss = get_val_loss(x_val, y_val)
    #    print(f"Total loss in epoch {_+1} : {total_loss / epoch_iter} and validation loss : {val_loss}")
