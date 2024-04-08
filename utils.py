import torch
from torch.utils.data import Dataset
import numpy as np
import logging 
import time 
import os
import cv2
import math 
import albumentations
from sklearn.metrics import mean_squared_error as MSE

class TrainData(Dataset):
    def __init__(self,args):
        self.img_path = args.train_data_path
        self.size_h, self.size_w = args.size
        num_patch = 50
        img_names = os.listdir(self.img_path)
        self.img_names = img_names * num_patch

    def __getitem__(self,index):
        gt = np.zeros([self.size_h, self.size_w],dtype=np.float32)

        image = cv2.imread(os.path.join(self.img_path,self.img_names[index]))
        image_h, image_w = image.shape[:2]
        if image_h > image_w:
            image = cv2.flip(image, 1)
            image = cv2.transpose(image)
            image_h, image_w = image.shape[:2]

        crop_flag = np.random.randint(0,10)>5
        crop_h = np.random.randint(self.size_h//2,image_h)
        if crop_flag:
            crop_w = np.random.randint(self.size_w//2,image_w)
        else:
            crop_w = crop_h     

        transform = albumentations.Compose([
            albumentations.RandomCrop(height=crop_h,width=crop_w,p=1),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Resize(self.size_h,self.size_w)
        ])
        gt = transform(image=image)['image']
        gt = cv2.cvtColor(gt,cv2.COLOR_BGR2YCrCb)[:,:,0]
        nrows,ncols = gt.shape
        if nrows != self.size_h or ncols != self.size_w:
            raise ValueError("Training image dimension is not the desired one.")
        gt = gt.astype(np.float32)
        gt /= 255.
        return gt

    def __len__(self,):
        return len(self.img_names)

class TestData(Dataset):
    def __init__(self,args):
        self.size = args.size
        self.data_path = args.benchmark_path
        self.data_list = os.listdir(self.data_path)

    def __getitem__(self,index):
        pic = cv2.imread(os.path.join(self.data_path,self.data_list[index]))
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2YCrCb)[:,:,0]
        # h,_ = pic.shape
        # if h > self.size[0]:
        #     pic = einops.rearrange(pic,'(a h) (b w)-> (a b) h w',a=2,b=2)
        # else:
        #     pic = einops.rearrange(pic,'(a h) (b w)-> (a b) h w',a=1,b=1)
        # gt = pic / 255
        gt = pic
        return gt
    def __len__(self,):
        return len(self.data_list)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compare_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))


def compare_psnr(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def Logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s')

    localtime = time.strftime('%Y_%m_%d_%H_%M_%S')
    logfile = os.path.join(log_dir,localtime+'.log')
    fh = logging.FileHandler(logfile,mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 


def checkpoint(epoch, model, optimizer, model_out_path):
    torch.save({'pretrain_epoch':epoch,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()}, model_out_path)

def load_checkpoint(model, pretrained_dict, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys()} # filtering parameters
    # for k,p in pretrained_model_dict.items():
    #     if k == 'H':
    #         Phi_H = p.detach().clone().cpu().numpy()
    #     if k == 'W':
    #         Phi_W = p.detach().clone().cpu().numpy()
    # scio.savemat('./phi.mat', {'H': Phi_H, 'W': Phi_W})
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

def load_checkpoint_withoutHW(model, pretrained_dict, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys() and k != 'H' and k != 'W'} # filtering parameters  and k != 'H' and k != 'W'
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

def compute_ortho(A,B):
    h1, w1 = A.shape
    h2, w2 = B.shape
    AT = np.transpose(A)
    BT = np.transpose(B)
    return np.sqrt(MSE(np.matmul(A,AT), np.identity(h1))) + np.sqrt(MSE(np.matmul(B,BT), np.identity(h2)))