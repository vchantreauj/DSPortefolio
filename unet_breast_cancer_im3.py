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
import time
from datetime import date

class UNet(nn.Module):
    """encoder decoer method to analyse medical images"""
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
                    torch.nn.Sigmoid(), # ReLU(), #add this to have binary mask as output
                    )
            return  block
    
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
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
        """encoding block is 2 convolutions layers each followed by a max pooling layers 
        with a stride of 2 for downsampling
        for each two layers conv/maxpool, Hout = Hin/2 - (k+1)"""
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        """decoding block is upsampling, concatenation and convolutions
        for each two layers crop and conv, Hout = 2Hin - 3k + 1"""
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return  final_layer 
 
def train_step(inputs, labels, optimizer, criterion):
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = unet(inputs)
    outputs = outputs.permute(0, 2, 3, 1)
    outputs = outputs.resize(batch_size*width_out*height_out, 2)
    labels = labels.resize(batch_size*width_out*height_out)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
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

im = io.imread_collection('data/breast_cancer_cell_seg/Images/*.tif', plugin = 'tifffile')
mask = io.imread_collection('data/breast_cancer_cell_seg/Masks/*.TIF', plugin = 'tifffile')

# width and height are the sizes of the input images
width = 764
height = 892
# width_out and height_out are the size of the image outputed by unet, due to the convolutions cropping
width_out = 676
height_out = 804

nb_im = 50
x_train = []
labels = []
for x in range(nb_im):
    x_train.append(im[x][:764,:892])
    labels.append(mask[x][44:720,44:848])

x_train = np.array(x_train)/255
x_train = np.transpose(x_train,(0,3,1,2))
labels = np.array(labels)/255

batch_size = 1
epochs = 1000 
#epoch_lapse = 1
unet = UNet(in_channel=3,out_channel=2)
unet = unet.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)

epoch_iter = np.ceil(nb_im / batch_size).astype(int)
t0 = time.time()
for _ in tqdm(range(epochs)):
    total_loss = 0
    for i in range(epoch_iter):
        batch_train_x = torch.as_tensor(x_train[i * batch_size : (i + 1) * batch_size]).float()
        batch_train_y = torch.as_tensor(labels[i * batch_size : (i + 1) * batch_size]).long()
        batch_train_x = batch_train_x.cuda()
        batch_train_y = batch_train_y.cuda()
        batch_loss = train_step(batch_train_x , batch_train_y, optimizer, criterion)
        total_loss += batch_loss
        torch.cuda.empty_cache()

t1 = time.time()
with open('logunet.txt', 'w') as f:
    f.write("training %d epochs on %d images\n"%(epochs, nb_im))
    f.write("last total loss %d last batch loss %d \n"%(total_loss, batch_loss))
    f.write("took %0.2f seconds\n"% (t1 - t0))

print("took %0.2f seconds"% (t1 - t0))

today = date.today()
unetfile = today.strftime("%y%m%d")+'unet'+str(epochs)+'ep.pt'
torch.save(unet.state_dict(), unetfile)

# on known data
imx = batch_train_x.squeeze(0).detach().cpu().numpy()
plt.imshow(np.transpose(imx, (1,2,0)))
plt.savefig('outputunet/train_x0.jpg')
output = unet(batch_train_x)
outputnp = output.squeeze(0).detach().cpu().numpy()
maskim = np.transpose(outputnp, (1,2,0)) > 0.5
plt.imshow(maskim[:,:,1])
plt.savefig('outputunet/train_pred01.jpg')
plt.imshow(maskim[:,:,0])
plt.savefig('outputunet/train_pred00.jpg')
targety = batch_train_y.squeeze(0).detach().cpu().numpy()
plt.imshow(targety)
plt.savefig('outputunet/train_y0.jpg')
torch.cuda.empty_cache()
# on test data
x_test = []
y_test = []
for x in range(nb_im, len(mask)):
    x_test.append(im[x][:764,:892])
    y_test.append(mask[x][44:720,44:848])

j=0
x_test = np.array(x_test)
x_test = np.transpose(x_test,(0,3,1,2))
batch_test_x = torch.as_tensor(x_test[j * batch_size : (j + 1) * batch_size]).float()
batch_test_x = batch_test_x.cuda()
test_xnp = batch_test_x.squeeze(0).detach().cpu().numpy()
y_test = np.array(y_test)
#y_test = np.transpose(y_test,(0,3,1,2))
batch_test_y = torch.as_tensor(y_test[j * batch_size : (j + 1) * batch_size]).long()
batch_test_y = batch_test_y.cuda()
test_ynp = batch_test_y.squeeze(0).detach().cpu().numpy()
torch.cuda.empty_cache()
test_pred = unet(batch_test_x)
test_pred = test_pred.squeeze(0).detach().cpu().numpy()
plt.imshow(np.transpose(test_xnp, (1,2,0))[:,:,0])
plt.savefig('outputunet/testx'+str(j)+'.jpg')
maskim = np.transpose(test_pred, (1,2,0)) > 0.5
plt.imshow(maskim[:,:,1])
plt.savefig('outputunet/testpred'+str(j)+'1.jpg')
plt.imshow(maskim[:,:,0])
plt.savefig('outputunet/testpred'+str(j)+'0.jpg')
plt.imshow(test_ynp)
plt.savefig('outputunet/testy'+str(j)+'.jpg')
torch.cuda.empty_cache()

# load model
datestr = '200728'
unetfile = datestr+'unet'+str(epochs)+'ep.pt'
device = torch.device("cuda")
unet = UNet(in_channel=3,out_channel=2)
unet.load_state_dict(torch.load(unetfile))
unet.to(device)
unet.eval()

    #if (_+1) % epoch_lapse == 0:
    #    val_loss = get_val_loss(x_val, y_val)
    #    print(f"Total loss in epoch {_+1} : {total_loss / epoch_iter} and validation loss : {val_loss}")

xori2 = batch_train_x.squeeze(0).detach().cpu().numpy()
plt.imshow(np.transpose(xori2, (1,2,0)))
plt.savefig('outputunet/imori2.jpg')


def plot_examples(datax, datay, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = unet(torch.from_numpy(datax[image_indx:image_indx+1]).float().cuda())
        .squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx], (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.transpose(image_arr, (1,2,0))[:,:,0])
        ax[row_num][1].set_title("Segmented Image")
        ax[row_num][2].imshow(image_arr.argmax(0))
        ax[row_num][2].set_title("Segmented Image localization")
        ax[row_num][3].imshow(np.transpose(datay[image_indx], (1,2,0))[:,:,0])
        ax[row_num][3].set_title("Target image")
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    #plt.show()




