from torch.utils.data import DataLoader 
import torch 
import os 
import os.path as osp
import scipy.io as scio
import numpy as np 
import einops
from opts import parse_args
from model.network import HATNet
from utils import Logger, load_checkpoint, TestData, compare_ssim, compare_psnr, load_checkpoint_withoutHW
import cv2
from skimage.metrics import structural_similarity as ski_ssim

torch.set_float32_matmul_precision('highest')

def test(args, network, logger, test_dir, epoch=1):
    network = network.eval()
    test_data = TestData(args)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    rec_list,gt_list = [],[]
    for iter, data in enumerate(test_data_loader):
        gt = data[0]
        gt = gt.float().numpy()
        if gt.shape[0]==args.size[0] and gt.shape[1]==args.size[1]:
            inp = einops.rearrange(gt,'(a h) (b w)-> (a b) h w',a=1,b=1)
        else:
            inp = einops.rearrange(gt,'(a h) (b w)-> (a b) h w',a=2,b=2)

        with torch.no_grad():
            x0 = inp / 255.0
            x = torch.from_numpy(x0).to(args.device)
            out,_,_,_,_ = network(x)
        out = out.cpu().numpy()
        batch,_,_ = out.shape
        psnr_t = 0
        ssim_t = 0
        for k in range(batch):
            psnr_t += compare_psnr(inp[k,:,:],out[k,:,:]*255)
            ssim_t += ski_ssim(inp[k,:,:],out[k,:,:]*255,data_range=255)
        psnr = psnr_t / batch
        ssim = ssim_t / batch
        psnr_list.append(np.round(psnr,4))
        ssim_list.append(np.round(ssim,4))
        if batch>1:
            pic = einops.rearrange(out,'(a b) h w-> (a h) (b w)',a=2,b=2)
        else:
            pic = einops.rearrange(out,'(a b) h w-> (a h) (b w)',a=1,b=1)
        rec_list.append(pic)
        gt_list.append(gt)

    for i,name in enumerate(test_data.data_list):
        _name,_ = name.split(".")
        psnr_dict[_name] = psnr_list[i]
        ssim_dict[_name] = ssim_list[i]
        image_name = os.path.join(test_dir, _name+"_"+"epoch_"+str(epoch)+".png")
        result_img = np.concatenate([gt_list[i]/255.0,rec_list[i]],axis=1)*255
        result_img = result_img.astype(np.float32)
        cv2.imwrite(image_name,result_img)

    if logger is not None:
        logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
        logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
    return psnr_dict, ssim_dict

if __name__=="__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    test_path = "test_results"
    network = HATNet(imag_size=args.size,
                    meas_size=args.meas_size,
                    img_channels=args.color_channels,
                    channels=args.channels,
                    mid_blocks=args.mid_blocks,
                    enc_blocks=args.enc_blocks,
                    dec_blocks=args.dec_blocks,
                    stages=args.stages,
                    matrix_train = args.matrix_train).to(args.device)
    log_dir = os.path.join("test_results","log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    if args.test_weight_path is not None:
        logger.info('Loading pretrained model...')
        pretrained_dict = torch.load(args.test_weight_path)
        load_checkpoint(network, pretrained_dict)
        # load_checkpoint_withoutHW(network, pretrained_dict)
    else:
        raise ValueError('Please input a weight path for testing.')
    result_path2 = 'results' + '/' + '{}'.format(args.decoder_type) + '/' + '0' + '/test'
    psnr_dict, ssim_dict = test(args, network, logger, result_path2, epoch=1)
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))
    logger.info("test finish!")
