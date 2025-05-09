import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model.network import HATNet
import torch.optim as optim
import os
import cv2
import scipy.io as scio
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from opts import parse_args
from test import test
import time
import datetime
from utils import load_checkpoint, checkpoint, TrainData, Logger, time2file_name, load_checkpoint_withoutHW, \
    compute_ortho

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set the precision to high
torch.set_float32_matmul_precision('highest')


def train(args, network, optimizer, logger, weight_path, mask_path, result_path1, result_path2=None):
    criterion = nn.MSELoss()
    criterion = criterion.to(args.device)
    rank = 0
    Eye_H = torch.eye(args.meas_size[0]).to(args.device)
    Eye_W = torch.eye(args.meas_size[1]).to(args.device)
    if args.distributed:
        rank = dist.get_rank()
    dataset = TrainData(args)

    # 多GPU训练数据加载
    if args.distributed:
        dist_sampler = DistributedSampler(dataset, shuffle=True)
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                       sampler=dist_sampler, num_workers=args.num_workers)
    else:
        train_data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers)

    # 模型训练
    for epoch in range(pretrain_epoch + 1, pretrain_epoch + args.epochs + 1):

        epoch_loss = 0
        network = network.train()
        start_time = time.time()
        for iteration, data in enumerate(train_data_loader):
            gt = data
            gt = gt.float().to(args.device)
            optimizer.zero_grad()
            out, H, W, HT, WT = network(gt)

            loss_ortho = torch.sqrt(criterion(torch.mm(H, HT), Eye_H)) + torch.sqrt(criterion(torch.mm(W, WT), Eye_W))
            loss_fidelity = torch.sqrt(criterion(out, gt))
            loss = loss_fidelity + 0.01 * loss_ortho

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            if rank == 0 and (iteration % args.iter_step) == 0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                logger.info(
                    'epoch: {:<3d}, iter: {:<4d}, loss: {:.5f}, fidelity loss: {:.5f}, ortho loss: {:.10f}, lr: {:.6f}.'
                    .format(epoch, iteration, loss.item(), loss_fidelity.item(), loss_ortho.item(), lr))
            # 训练期间视频可视化结果
            if rank == 0 and (iteration % args.save_train_image_step) == 0:
                image_out = out[0].detach().cpu().numpy()
                image_gt = gt[0].cpu().numpy()
                image_path = './' + result_path1 + '/' + 'epoch_{}_iter_{}.png'.format(epoch, iteration)
                result_img = np.concatenate([image_gt, image_out], axis=1) * 255
                result_img = result_img.astype(np.float32)
                cv2.imwrite(image_path, result_img)
        end_time = time.time()
        if rank == 0:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info(
                'epoch: {}, avg. loss: {:.5f}, lr: {:.6f}, time: {:.2f}s.\n'.format(epoch, epoch_loss / (iteration + 1),
                                                                                    lr, end_time - start_time))

        # 模型参数保存
        if rank == 0 and (epoch % args.save_model_step) == 0:
            model_out_path = './' + weight_path + '/' + 'epoch_{}.pth'.format(epoch)
            if args.distributed:
                checkpoint(epoch, network.module, optimizer, model_out_path)
            else:
                checkpoint(epoch, network, optimizer, model_out_path)
            mask_path1 = './' + mask_path + '/' + 'Cr_{}_epoch_{}.mat'.format(args.cr, epoch)
            Phi_H = H.detach().clone()
            Phi_W = W.detach().clone()
            Phi_H = Phi_H.cpu().numpy()
            Phi_W = Phi_W.cpu().numpy()
            # Phi = np.kron(W,H)
            scio.savemat(mask_path1, {'H': Phi_H, 'W': Phi_W})

        # scheduler.step()

        # 模型测试
        if rank == 0 and args.test_flag:
            logger.info('epoch: {}, psnr and ssim test results:'.format(epoch))
            if args.distributed:
                psnr_dict, ssim_dict = test(args, network.module, logger, result_path2, epoch=epoch)
            else:
                psnr_dict, ssim_dict = test(args, network, logger, result_path2, epoch=epoch)

            logger.info('psnr_dict: {}.'.format(psnr_dict))
            logger.info('ssim_dict: {}.\n'.format(ssim_dict))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('highest')
    args = parse_args()
    rank = 0
    pretrain_epoch = 0
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    if rank == 0:
        result_path1 = 'results' + '/' + '{}'.format(args.decoder_type) + '/' + date_time + '/train'
        weight_path = 'weights' + '/' + '{}'.format(args.decoder_type) + '/' + date_time
        mask_path = 'masks' + '/' + '{}'.format(args.decoder_type) + '/' + date_time
        log_path = 'log/log' + '/' + '{}'.format(args.decoder_type)
        show_path = 'log/show' + '/' + '{}'.format(args.decoder_type) + '/' + date_time
        if not os.path.exists(result_path1):
            os.makedirs(result_path1, exist_ok=True)
        if not os.path.exists(weight_path):
            os.makedirs(weight_path, exist_ok=True)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path, exist_ok=True)
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        if not os.path.exists(show_path):
            os.makedirs(show_path, exist_ok=True)
        if args.test_flag:
            result_path2 = 'results' + '/' + '{}'.format(args.decoder_type) + '/' + date_time + '/test'
            if not os.path.exists(result_path2):
                os.makedirs(result_path2, exist_ok=True)
        else:
            result_path2 = None

    logger = Logger(log_path)

    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device('cuda', local_rank)  ##########
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()

    if rank == 0:
        logger.info('\n' + 'Date:' + date_time + '\n' +
                    'Network Architecture: {}-{}-{}-{}'.format(args.decoder_type, args.enc_blocks, args.mid_blocks,
                                                               args.dec_blocks) + '\n' +
                    'Stage Number: {}'.format(args.stages) + '\n' +
                    'Channel: {}'.format(args.channels) + '\n' +
                    'Batch Size: {}'.format(args.batch_size) + '\n' +
                    'Image Size: {}'.format(args.size) + '\n' +
                    'Meas Size: {}'.format(args.meas_size) + '\n' +
                    'Compressive Ratio: {}'.format(args.cr) + '\n' +
                    'Learning Rate: {:.6f}'.format(args.lr) + '\n' +
                    'Train Epochs: {}'.format(args.epochs) + '\n' +
                    'Matrix Train: {}'.format(args.matrix_train) + '\n' +
                    'Test or Not: {}'.format(args.test_flag) + '\n' +
                    'Pretrain Model: {}'.format(args.pretrained_model_path)
                    )

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    network = HATNet(imag_size=args.size,
                     meas_size=args.meas_size,
                     img_channels=args.color_channels,
                     channels=args.channels,
                     mid_blocks=args.mid_blocks,
                     enc_blocks=args.enc_blocks,
                     dec_blocks=args.dec_blocks,
                     stages=args.stages,
                     matrix_train=args.matrix_train).to(args.device)
    has_compile = hasattr(torch, 'compile')
    if args.torchcompile:
        # FIXME dynamo might need move below DDP wrapping? TBD
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'

        network = torch.compile(network, backend=args.torchcompile)
    optimizer = optim.Adam([{'params': network.parameters()}], lr=args.lr)

    # 学习率更新策略，warmup_steps个epoch后学习率增至设定值，之后，每step_size个epoch，学习率降低gamma (https://arxiv.org/abs/2109.06548)
    # scheduler = WarmupStepLR(warmup_steps = 1, optimizer = optimizer, step_size = 5, gamma = 0.9)

    # 模型参数加载
    if rank == 0:
        if args.pretrained_model_path is not None:
            logger.info('Loading pretrained model...')
            pretrained_dict = torch.load(args.pretrained_model_path)
            if 'pretrain_epoch' in pretrained_dict.keys():
                pretrain_epoch = pretrained_dict['pretrain_epoch']
            load_checkpoint(network, pretrained_dict)
            # load_checkpoint_withoutHW(network, pretrained_dict)
        else:
            logger.info('No pretrained model.')

    # 多GPU训练
    if args.distributed:
        network = DDP(network, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=True)  # ,find_unused_parameters=True

    train(args, network, optimizer, logger, weight_path, mask_path, result_path1, result_path2)

