import argparse
import copy
import os
import random
import time
from builtins import print

import imageio
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import ms_ssim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from tqdm import tqdm, trange

from lib import tineuvox, utils
from lib.load_data import load_data
from lib.VoxelDeformation import *

from StyLoader import *

import cv2
from VGGUtils import *

from CCPL.ccpl import *

def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--eval_psnr", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=2000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--fre_test", type=int, default=30000,
                        help='frequency of test')
    parser.add_argument("--step_to_half", type=int, default=19000,
                        help='The iteration when fp32 becomes fp16')
    return parser

@torch.no_grad()
def render_viewpoints_hyper(model, data_class, ndc, render_kwargs, test=True, 
                                all=False, savedir=None, eval_psnr=False):
    
    rgbs = []
    rgbs_gt =[]
    rgbs_tensor =[]
    rgbs_gt_tensor =[]
    depths = []
    psnrs = []
    ms_ssims =[]

    if test:
        if all:
            idx = data_class.i_test
        else:
            idx = data_class.i_test[::16]
    else:
        if all:
            idx = data_class.i_train
        else:
            idx = data_class.i_train[::16]
    for i in tqdm(idx):
        rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(i, not_dic=True)
        keys = ['rgb_marched', 'depth']
        time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
        cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
        bacth_size = 1000
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts, cams,**render_kwargs).items() if k in keys}
            for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                             viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(data_class.h,data_class.w,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb_gt = rgb_gt.reshape(data_class.h,data_class.w,-1).cpu().numpy()
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        rgbs.append(rgb)
        depths.append(depth)
        rgbs_gt.append(rgb_gt)
        if eval_psnr:
            p = -10. * np.log10(np.mean(np.square(rgb - rgb_gt)))
            psnrs.append(p)
            rgbs_tensor.append(torch.from_numpy(np.clip(rgb,0,1)).reshape(-1,data_class.h,data_class.w))
            rgbs_gt_tensor.append(torch.from_numpy(np.clip(rgb_gt,0,1)).reshape(-1,data_class.h,data_class.w))
        if i==0:
            print('Testing', rgb.shape)
    if eval_psnr:
        rgbs_tensor = torch.stack(rgbs_tensor,0)
        rgbs_gt_tensor = torch.stack(rgbs_gt_tensor,0)
        ms_ssims = ms_ssim(rgbs_gt_tensor, rgbs_tensor, data_range=1, size_average=True )
    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        print('Testing ms_ssims', ms_ssims, '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs,depths


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, test_times=None, render_factor=0, eval_psnr=False,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor
    rgbs = []
    depths = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                H, W, K, c2w, ndc)

        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        bacth_size=4096
        # for test in rays_o.split(bacth_size, 0):
        #     print(test.shape)
        # print(len(rays_o.split(bacth_size, 0)))
        # print(type(rays_o.split(bacth_size, 0)))

        viewdirs = viewdirs.flatten(0,-2)
        time_one = test_times[i]*torch.ones_like(rays_o[:,0:1])


        for ro, rd, vd ,ts in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0), viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0)):
            resnder_results = model(ro, rd, vd,ts, **render_kwargs)
        return
        bacth_size=1000    
        keys = ['rgb_marched', 'depth']
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd,ts, **render_kwargs).items() if k in keys}
            for ro, rd, vd ,ts in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0), viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor == 0:
            if eval_psnr:
                p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
                psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'alex', device = c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name = 'vgg', device = c2w.device))

    if len(psnrs):
        if eval_psnr: print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if savedir is not None:
        print(f'Writing images to {savedir}')
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    return rgbs, depths


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)
    if cfg.data.dataset_type == 'hyper_dataset':
        kept_keys = {
            'data_class',
            'near', 'far',
            'i_train', 'i_val', 'i_test',}
        for k in list(data_dict.keys()):
            if k not in kept_keys:
                data_dict.pop(k)
        return data_dict

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images','times','render_times','focal'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device = 'cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    BBX_TIme = []
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w,ndc=cfg.data.ndc)
        if cfg.data.ndc:
            pts_nf = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        else:
            pts_nf = torch.stack([rays_o+viewdirs*near, rays_o+viewdirs*far])
        min_curtime = pts_nf.amin((0,1,2))
        max_curtime = pts_nf.amax((0,1,2))
        BBX_TIme.append([min_curtime,max_curtime ] )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1,2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1,2)))


    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max,BBX_TIme


def compute_bbox_by_cam_frustrm_hyper(args, cfg,data_class):
    print('compute_bbox_by_cam_frustrm: start')
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    BBX_TIme = []
    for i in data_class.i_train:
        rays_o, _, viewdirs,_ = data_class.load_idx(i,not_dic=True)
        pts_nf = torch.stack([rays_o+viewdirs*data_class.near, rays_o+viewdirs*data_class.far])
        min_curtime = pts_nf.amin((0,1))
        max_curtime = pts_nf.amax((0,1))
        # print(pts_nf.shape)
        # print(max_curtime.shape)
        BBX_TIme.append([min_curtime,max_curtime ] )
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0,1)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0,1)))
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max,BBX_TIme


def OPReferenceSimilarity(X,Y):
    # return F.mse_loss(X,Y)
    ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=1, size_average=True)
    return ms_ssim_loss

def OriginalImageSimilarity(X,Y):
    #return F.mse_loss(X,Y)
    ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=1, size_average=True)
    return ms_ssim_loss


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, 
                             BBX_Time,
                             data_dict):


    xyz_min_overwrite = torch.tensor([-1.1, -0.26, -1.34])
    xyz_max_overwrite = torch.tensor([1.1, 0.37, 1.1])
    print(xyz_min)
    print(xyz_max)
    
    xyz_min_overwrite = xyz_min / 1.0
    xyz_max_overwrite = xyz_max / 1.0 
    
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stylizied_path = os.path.join(cfg.basedir, cfg.expname, f'render_train_stylizied')

    os.makedirs(stylizied_path, exist_ok=True)
    
    stylizied_path_final = os.path.join(cfg.basedir, cfg.expname, f'render_train_stylizied_final')
    os.makedirs(stylizied_path_final, exist_ok=True)
    
    # print(stylizied_path)
    # return
    import shutil
    current_file_path = os.path.abspath(__file__)
    shutil.copy(current_file_path, stylizied_path)
    

    IsDnerf = cfg.data.dataset_type !='hyper_dataset'
    if IsDnerf:
        batch_base = 2048 * 2
    else:
        batch_base = 4096 
    TimeScale = 1.0
    c2wFrameOverwriteIndex = 15
    EnableFrameOverwrite = False
    EnableTimeOverwrite = False
    itercount = 300
    TotalTime = 1.0

    DeformNetUseSameInitPara = True
    FreezeOpt = False   
    NumBatchSize = 2048 * 2 *2
    deform_approach_weight  = 1.0
    stylized_deform_tv_weight = 1e3 * 0.0
    deform_weight_tv_k0 = 1000000000000.0 *0.0
    
    PureAdvectionOnly = False
    
    Dynamic_OPT_Start_Frame = 1
    stylizied_deformed_Scale_global = 1.0
    lrscale = 0.2
    lrscale_deform_rebuild = 0.1
    UseNNFM = False
    substeps_count = 30
    align_substeps_count = 25  

    finalPrepareAlpha = 0.001
    
    
    ori_score = 5000 * 1000
    sty_score = 1.0  
    
    #load style img
    # will be all read
    Target_resolution = 500
    # style_img = image_loader("starskyMonoColor.png")
    style_img = image_loader("windows2.png")
    style_img = image_loader("waves.png")
    style_img = image_loader("handdraw.png")
    # style_img = image_loader("picasso.jpg")
    # style_img = image_loader("starsky.png")

    EnableGamma = False
    
    style_img = style_img.squeeze(0)
    style_img = style_img.permute(1,2,0)
    
    
    # print(style_img.shape)
    # return
    # resized_image = style_img.cpu().numpy()[:Target_resolution,:Target_resolution,:]
    resized_image = cv2.resize(style_img.cpu().numpy(), (Target_resolution, Target_resolution)) 
    
    style_img = torch.tensor(resized_image)
    style_img = style_img.permute(2,0,1)
    style_img = style_img.unsqueeze(0)
    style_img = style_img[:,:3,:,:]
    if EnableGamma:
        style_img = torch.pow(style_img,1.0)
    
    # VGG models load
    cnn = models.vgg19(pretrained=True).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    
    content_layers_default = ['conv_3','conv_4']
    style_layers_default = [ 'conv_8','conv_9']
    # style_layers_default = [ 'conv_5','conv_6']
    # style_layers_default = [ 'conv_13','conv_14']
    with torch.no_grad():
        VGGmodel, style_losses, content_losses,content_models = get_style_model_and_losses_withModels(cnn,
        cnn_normalization_mean, cnn_normalization_std, style_img,content_layers_default, style_layers_default)
    print(VGGmodel)
    VGGmodel.eval()
    VGGmodel.requires_grad_(False)

    USECCPL = False
    CCPLOptMLP = False
    
    CPPLModel = InitCPPL()


    # load model_dict
    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    # ckpt_name = ckpt_path.split('/')[-1][:-4]
    model_class = tineuvox.TiNeuVox
    NerfPreTrained_model = utils.load_model(model_class, ckpt_path).to(device)
    
    NerfPreTrained_model.eval()
    NerfPreTrained_model.requires_grad_(False)
    timcount = len(BBX_Time) * 1.0
    times = torch.arange(timcount) / timcount
    times = times.cuda()
    
    xyz_min_overwrite,xyz_max_overwrite = NerfPreTrained_model.compute_bbox_by_coarse_geo_deform(0.5,times,0.5,BBX_Time)
    
    # xyz_min_overwrite = xyz_min / 2.0
    # xyz_max_overwrite = xyz_max / 2.0 
    print(xyz_min_overwrite)
    print(xyz_max_overwrite)
    #print(res)
    near=data_dict['near']
    far=data_dict['far']
    
    # color trans ralated

    EnableTrainColor = True
    NNFMContent = 10.0
    EnableTrainColor_Continue = True
    coloroptScale = 0.002
    coloroptScale_bgein = 1.0
    coloroptScale = coloroptScale_bgein * 0.01
    
    MSE_LR = 100.0
    
    MSE_LR_Real = MSE_LR / coloroptScale_bgein
    
    # add field not used
    Enable_RaidanceTrain = False
    if EnableTrainColor:
        Enable_RaidanceTrain = False
    optOriNeRFColorField = True
    if not EnableTrainColor:
        optOriNeRFColorField = False

    stepsize = cfg.model_and_render.stepsize


    # init rendering setup
    render_kwargs = {
        'near': near,
        'far': far,
        'bg': 1 if cfg.data.white_bkgd else 0,
        'stepsize': cfg_model.stepsize,
    }
  
    torch.cuda.empty_cache()
    
    test = True
    FrameCount = 0

    if IsDnerf:
        HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images ,times,render_times= [
            data_dict[k] for k in [
                'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 
                'render_poses', 'images',
                'times','render_times',
            ]
        ]
        times = torch.Tensor(times)
        times_i_train = times[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)
    else:
        data_class = data_dict['data_class']
        near = data_class.near
        far = data_class.far
        i_train = data_class.i_train
        i_test = data_class.i_test
        
    if IsDnerf:
        # if test:
        #     render_poses=data_dict['poses'][data_dict['i_test']],
        #     test_times = data_dict['times'][data_dict['i_test']]
        # else:
        #     render_poses=data_dict['poses'][data_dict['i_train']],
        #     test_times = data_dict['times'][data_dict['i_train']] 
        render_poses = [render_poses]
        test_times = render_times
        # print((render_poses.shape))
        FrameCount = len(render_poses[0])
    else:
              
        if test:
            idx = data_class.i_test
        else:
            idx = data_class.i_train
        FrameCount = len(idx)  
    
    if IsDnerf:
        render_resoultion = 400
        H = render_resoultion
        W = render_resoultion
        focal = Ks[0][0,0]
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        #Ks = K[None].repeat(len(poses), axis=0)
        K = torch.tensor(K).to('cpu').numpy()
        

        for ind in range(len(Ks)):
            Ks[ind] = K
            HW[ind] = [H,W]
    else:
        render_resoultion = 512
        H = render_resoultion
        W = render_resoultion
        focal = 0.0
        K = None


  
    
    # print(len(render_poses[0]))
    print('totalFrameCount:')
    print(FrameCount)
    # hyperdata


    enableColorTransfer_ARF = True

    enableColorTransfer_ARF_load = False
    
    enableColorTransfer_ARF_oriMode = False
    
    if not EnableTrainColor:
        enableColorTransfer_ARF = False
    
    stylizedDeformGrid_FrameIndexed = []
    #TODO add another mode copy from first
    # stylizedDeformGrid = StyliziedPureVoxelGO(
    #         xyz_min=xyz_min_overwrite, xyz_max=xyz_max_overwrite,
    #         deform_num_voxels= 190**3,
    #         deform_num_voxels_base=190**3,
    #         H =render_resoultion,W = render_resoultion,
    #         focal=focal,c2w= None,far=far,near=near
            
    #     ).to('cpu')
    if IsDnerf:
        c2wFrameOverwrite = render_poses[0][c2wFrameOverwriteIndex,...]
        
    for i in range(FrameCount):
        if IsDnerf:
            c2w = render_poses[0][i,...]
            if EnableFrameOverwrite:
                c2w = c2wFrameOverwrite 
        else:
            c2w = None
        stylizedDeformGrid = StyliziedPureVoxelGO(
                xyz_min=xyz_min_overwrite, xyz_max=xyz_max_overwrite,
                deform_num_voxels= 190**3,
                deform_num_voxels_base=190**3,
                H =render_resoultion,W = render_resoultion,
                focal=focal,c2w= c2w,far=far,near=near
                
            ).to('cpu')
        stylizedDeformGrid_FrameIndexed.append(copy.deepcopy(stylizedDeformGrid))
    
    DeltaDeformGrid_FrameIndexed = []
    Stylizied_Results = []
    

        
    
    stylizedDeformGrid = StyliziedPureVoxelGO(
            xyz_min=xyz_min_overwrite, xyz_max=xyz_max_overwrite,
            deform_num_voxels= 190**3,
            deform_num_voxels_base=190**3,
            H =render_resoultion,W = render_resoultion,
            focal=focal,c2w= c2w,far=far,near=near
            
        ).to('cpu') 

    for i in range(FrameCount):
        DeltaDeformGrid_FrameIndexed.append(copy.deepcopy(stylizedDeformGrid))
        Stylizied_Results.append(None)
        
    TotalModel = Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth(
        NerfPreTrained_model,
        stylizedDeformGrid_FrameIndexed[1].cuda(),
        stylizedDeformGrid_FrameIndexed[0].cuda(),
        DeltaDeformGrid_FrameIndexed[0].cuda(),
         DeltaDeformGrid_FrameIndexed[0].cuda()
    )
    if EnableTrainColor:
        styRadNet_FrameIndexed = []
        styRadNet = StylizedRadianceNet(xyz_min=xyz_min_overwrite, xyz_max=xyz_max_overwrite,
                num_voxels= 160**3,
                num_voxels_base=160**3).cpu()
        for i in range(FrameCount):
            styRadNet_FrameIndexed.append(copy.deepcopy(styRadNet))
            styRadNet_FrameIndexed.append((styRadNet))
            
        TotalModel.setStyRadField(styRadNet_FrameIndexed[0].cuda(),styRadNet_FrameIndexed[0].cuda() )
    
    
    ndc = False

    global_step = -1
    start =0
    # optimizer = utils.create_optimizer_or_freeze_model(NerfPreTrained_model, cfg_train, global_step=0)

    # FrameCount = render_poses.shape[0]
    deltaT = TotalTime / FrameCount
    print(deltaT)
    contentImages = []
    depthImages = []
   
  
    def modify(idx,rays_o):
        return rays_o
        upVector = torch.tensor([0.0,1.0,0.0])
        upVector = upVector.reshape(1,3)
        
        rays_o = rays_o + upVector * np.sin([np.pi/8 * idx])[0] * 0.1
        return rays_o
    
    def readFilesUnderPath(path):
        path = path+'/'
        files= os.listdir(path) #得到文件夹下的所有文件名称
        s = []
        for file in files: #遍历文件夹
            if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
                s.append(path+file)
        return s

    CAP_files = []
    if enableColorTransfer_ARF_load:
        CAP_files = readFilesUnderPath('./CAPFolder')
        
    TotalModel.EnableRefMode = True
    TotalModel.AdaptStyRaidanceFieldMode = False
    with torch.no_grad():
    # prepare reference
        for i in tqdm(range(FrameCount)):
            FrameNum = i
            # if i==41:
            #     break
            SampleTime = 0.0
            if IsDnerf:
                if EnableFrameOverwrite:
                    c2w = c2wFrameOverwrite
                c2wFrameOverwrite = render_poses[0][c2wFrameOverwriteIndex,...]
                c2w = render_poses[0][i,...]
                if EnableFrameOverwrite:
                    c2w = c2wFrameOverwrite
                # print(c2w)
                # print(c2w.shape)
                H, W = HW[i]
                K = Ks[i]
                rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                        H, W, K, c2w, ndc)

                rays_o = rays_o.flatten(0,-2)
                rays_d = rays_d.flatten(0,-2)


                viewdirs = viewdirs.flatten(0,-2)
                time_one = test_times[i]*torch.ones_like(rays_o[:,0:1])
                SampleTime = torch.tensor(test_times[i])
                #dummy
                cam_one = test_times[i]*torch.ones_like(rays_o[:,0:1])
            else:
                # rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(i, not_dic=True)
                # keys = ['rgb_marched', 'depth']
                # time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
                # cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
                # bacth_size = 1000
                
                loadDataIdx = i
                if EnableFrameOverwrite:
                    loadDataIdx = c2wFrameOverwriteIndex
                rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(loadDataIdx, not_dic=True)
                
                rays_o = modify(i,rays_o)
                
                time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])
                SampleTime = torch.tensor(data_class.all_time[i])
                cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
                # print(rays_o.shape)
                # print(rays_d.shape)
                # print(viewdirs.shape)
                # print(time_one.shape)
                # print(SampleTime)
                # print(cam_one.shape)
                
                H = data_class.h
                W = data_class.w
            bacth_size = batch_base 
            # bacth_size = 512
            TotalModel.prepare(SampleTime,DeltaTime_T=deltaT)
              
            render_rgb = []
            render_depth = []
            for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                    viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0)):
                with torch.no_grad():
                    results = TotalModel.forward(ro, rd, vd,ts, cams, FrameNum= FrameNum,DeltaTime_T=deltaT, **render_kwargs)
                render_rgb.append(results['rgb_marched'])
                render_depth.append(results['depth'])
            depth_res = torch.cat(render_depth) #.reshape(H,W,-1)
            res = torch.cat(render_rgb).reshape(H,W,-1)
            noGradRGBTarget = res.permute(2, 0, 1)
            noGradRGBTarget = noGradRGBTarget.unsqueeze(0)
            if enableColorTransfer_ARF and i == 0 :
                noGradRGBTarget, color_tr = utils.match_colors_for_image_set(noGradRGBTarget,style_img)
            elif enableColorTransfer_ARF:
                noGradRGBTarget = utils.apply_CT(noGradRGBTarget,color_tr)
            # if enableColorTransfer_ARF and (i !=0) :
            #     noGradRGBTarget, _ = utils.match_colors_for_image_set(noGradRGBTarget,style_img)
  
            # print(noGradRGBTarget.shape)
            if enableColorTransfer_ARF_load:
                print('loaede')
                # image = Image.open(image_name)
                imgs = cv2.imread(CAP_files[i]) /256.0 #,dtype=torch.float32
                
                print(imgs.shape)
                print( noGradRGBTarget.squeeze(0).permute(1,2,0).shape[:2])
                resized_image = cv2.resize(imgs, [427,240]) 
                
                print(imgs.shape)
                
                noGradRGBTarget = torch.tensor(resized_image,dtype=torch.float32 )
                print(noGradRGBTarget.shape)
                noGradRGBTarget = noGradRGBTarget.permute(2,0,1).unsqueeze(0)
                print(noGradRGBTarget.shape)
                print('==end')
                # dim=(0,0,0,0,0,1)
                # noGradRGBTarget=F.pad(noGradRGBTarget,dim,"constant",value=0)
                # print(noGradRGBTarget.shape)
            depthImages.append(depth_res.to('cpu'))
            contentImages.append(noGradRGBTarget.to('cpu'))
    
    
    Enable_allignTrain = False
    
    import logging

    logging.basicConfig(filename=stylizied_path +'\output.log', level=logging.INFO)
    from lib import nnfm_loss
    nnfm_loss_fn = nnfm_loss.NNFMLoss(device='cuda')



    import logging

    # 创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件（日志级别为ERROR）
    fh1 = logging.FileHandler(filename=stylizied_path +'\output.log')
    fh1.setLevel(logging.INFO)

    # 再创建一个handler，用于输出到控制台（日志级别为DEBUG）
    fh2 = logging.FileHandler(filename=stylizied_path +'\\results.log')
    fh2.setLevel(logging.WARNING)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh1.setFormatter(formatter)
    fh2.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh1)
    logger.addHandler(fh2)


    logger.warning('This is warning message')


    

    
    for global_step in trange(1+start, 1+cfg_train.N_iters):
        
        if global_step == 2:
            break
        
        for i in tqdm(range(FrameCount)):
            SampleTime = 0.0
            # timestep_j = timestep_j * TimeScale
            
            print('begin iterations' + str(global_step))
            print('Frame count' + str(i))
            logger.warning('Frame count' + str(i))
            if IsDnerf:
                c2wFrameOverwrite = render_poses[0][c2wFrameOverwriteIndex,...]
                c2w = render_poses[0][i,...]
                if EnableFrameOverwrite:
                    c2w = c2wFrameOverwrite
                
                # print(c2w)
                # print(c2w.shape)
                H, W = HW[i]
                K = Ks[i]
                rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
                        H, W, K, c2w, ndc)

                rays_o = rays_o.flatten(0,-2)
                rays_d = rays_d.flatten(0,-2)
                


                viewdirs = viewdirs.flatten(0,-2)
                time_one = test_times[i]*torch.ones_like(rays_o[:,0:1]) * TimeScale
                SampleTime = torch.tensor(test_times[i])  * TimeScale
                print(SampleTime)
                #dummy
                cam_one = test_times[i]*torch.ones_like(rays_o[:,0:1])
            else:
                loadDataIdx = i
                if EnableFrameOverwrite:
                    loadDataIdx = c2wFrameOverwriteIndex
                rays_o, rays_d, viewdirs,rgb_gt = data_class.load_idx(loadDataIdx, not_dic=True)
                rays_o = modify(i,rays_o)
                
                time_one = data_class.all_time[i]*torch.ones_like(rays_o[:,0:1])  * TimeScale
                SampleTime = torch.tensor(data_class.all_time[i])
                print(SampleTime)
                cam_one = data_class.all_cam[i]*torch.ones_like(rays_o[:,0:1])
                H = data_class.h
                W = data_class.w
            bacth_size = batch_base * 1
            
            j = i
            WriteNameFrame = j
            # if EnableTimeOverwrite:
            #     j = 0
            
            current_Frame = j
            current_Save_Frame = j


            FrameNum = j
            targetFrameIndex = 0

            substeps_count = 30 * 4
            # substeps_count = 1
            
            enableTVLoss = False
            
            UsingBeforeImageTraining = False
            
            style_weight = 1000000
            
            style_weight_ori = style_weight
            
            
            lrscale_overall  = 0.5
            USECCPL = False
            CPPL_Weight = 1000
            lrscale = 0.5 * lrscale_overall
            if i < Dynamic_OPT_Start_Frame:

                if j != targetFrameIndex: 
                    continue  
            else:
                
                if j != targetFrameIndex: 
                    lrscale = 0.1 * lrscale_overall
                    Enable_allignTrain = True
                    PureAdvectionOnly = False
                    substeps_count = 30
                    align_substeps_count = 30
                    USECCPL = False
                    
            PureAdvectionOnly = False  
            
            lrscale_deform_rebuild = 0.25     *lrscale_overall    
            # substeps_count = 30
            # # lrscale = 0.0000001
            # substeps_count = 1
            # align_substeps_count = 1
            
            OP_reference_continueStyleLoss = 0.0
            allign_variance_weight = 0.0
            
            rgbResult_origin = None
            UseNNFM = True
            
            if UseNNFM:
                style_weight = 100000
                
                style_weight_ori = style_weight
                
            
            if UseNNFM:           
                def GetStyliziedImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one,Model,FrameNum,content_weight = 100, NNFMContent = NNFMContent, style_weight=style_weight,**render_kwargs):
                    # accumulate grad phase
                    with torch.no_grad():
                        Renderresults = []
                        for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                                viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0)):
                            #with torch.no_grad():
                            results = Model.forward(ro, rd, vd,ts, cams,renderStyImgaeWithoutGrad = True, FrameNum= FrameNum,DeltaTime_T=deltaT, **render_kwargs)
                            Renderresults.append(results['rgb_marched'])
                        noGradRGBTarget = torch.cat(Renderresults).reshape(H,W,-1)
                        
                        
                        noGradRGBTarget = noGradRGBTarget.permute(2, 0, 1)
                        noGradRGBTarget = noGradRGBTarget.unsqueeze(0)
                        # noGradRGBTarget = F.interpolate(noGradRGBTarget,scale_factor=(dialtedInterval,dialtedInterval),mode ='bilinear')
                        # 1 3 N N
                    noGradRGBTarget.requires_grad_(True)

                    if enableColorTransfer_ARF:
                        noGradRGBTarget_altered = utils.match_colors_for_image_set_newShape(noGradRGBTarget,style_img)
                        # noGradRGBTarget_altered = utils.apply_CT(noGradRGBTarget,color_tr)
                    else:
                        noGradRGBTarget_altered = noGradRGBTarget
                    
                    curImage = contentImages[current_Frame].cuda()      
            
                    w_variance = torch.mean(torch.pow(noGradRGBTarget_altered[:, :, :, :-1] - noGradRGBTarget_altered[:, :, :, 1:], 2))
                    h_variance = torch.mean(torch.pow(noGradRGBTarget_altered[:, :, :-1, :] - noGradRGBTarget_altered[:, :, 1:, :], 2))
                    if enableTVLoss:
                        img_tv_loss =  (h_variance + w_variance) / 2.0
                    else:
                        img_tv_loss = 0.0
                    loss_dict = nnfm_loss_fn(
                    noGradRGBTarget_altered,
                    style_img,
                    blocks=[
                    2,
                    ],
                    loss_names=["nnfm_loss", "content_loss"],
                    contents=curImage,
                    )
                    loss_dict["content_loss"] *= 1e-3 * NNFMContent * style_weight_ori
                    loss_dict["img_tv_loss"] = img_tv_loss * 0.000001 * 1000000
                    loss_dict["nnfm_loss"] = loss_dict["nnfm_loss"] * style_weight
                    # style_weight = 1000000
                    loss_ccp = 0.0
                    loss = sum(list(loss_dict.values())) 
                    
                    if enableColorTransfer_ARF_oriMode:
                        if style_weight == 0:
                            loss = torch.mean((curImage-noGradRGBTarget_altered)**2) * MSE_LR_Real
                            print('mse loss')
                    
                    refImg = contentImages[current_Frame].to('cuda')
                    # print(refImg.shape)
                    # print(noGradRGBTarget.shape)
                    # return
                    if USECCPL:
                        loss_ccp = CPPLModel(refImg, noGradRGBTarget_altered) * CPPL_Weight
                        loss += loss_ccp
                    print('current loss = ' + str(loss.item()))
                    print('current content_score = ' + str(loss_dict["content_loss"].item() ))
                    print('current style_score = ' + str(loss_dict["nnfm_loss"].item()   ))
                    if enableTVLoss:
                        print('current tvkloss = ' + str(img_tv_loss.item() * 10000))
                    if USECCPL:
                        print('CCP loss = ' + str(loss_ccp.item() ))
                    # logging.info('current loss = ' + str(loss.item()))
                    # logging.info('current content_score = ' + str(content_score.item()))
                    # logging.info('current style_score = ' + str(style_score.item()))
                    loss.backward()
                    RGBLoss = noGradRGBTarget.grad.detach().clone()
                    img = noGradRGBTarget_altered.detach().clone().cpu()
                    del noGradRGBTarget
                    return RGBLoss,img
            else:
                def GetStyliziedImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one,Model,FrameNum,content_weight = 100, **render_kwargs):
                    # accumulate grad phase
                    with torch.no_grad():
                        Renderresults = []
                        for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                                viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0)):
                            #with torch.no_grad():
                            results = Model.forward(ro, rd, vd,ts, cams,renderStyImgaeWithoutGrad = True, FrameNum= FrameNum,DeltaTime_T=deltaT, **render_kwargs)
                            Renderresults.append(results['rgb_marched'])
                        noGradRGBTarget = torch.cat(Renderresults).reshape(H,W,-1)
                        
                        noGradRGBTarget = noGradRGBTarget.permute(2, 0, 1)
                        noGradRGBTarget = noGradRGBTarget.unsqueeze(0)
                        # noGradRGBTarget = F.interpolate(noGradRGBTarget,scale_factor=(dialtedInterval,dialtedInterval),mode ='bilinear')
                        # 1 3 N N
                    noGradRGBTarget.requires_grad_(True)
                    style_score =0
                    content_score = 0
                    VGGmodel(noGradRGBTarget)
                    count =0 
                    for sl in style_losses:
                        style_score += sl.loss
                        count = count+1
                        
                    count =0 
                    for cl in content_losses:
                        with torch.no_grad():
                            feature_vector = content_models[count](contentImages[current_Frame].to('cuda')).detach().clone()
                        content_score += cl.loss(feature_vector)
                        count = count+1
                    
                
                    # style_weight = 1000000
                    
                    refImg = contentImages[current_Frame].to('cuda')
                    # style_weight = 0
                    # content_weight = 0
                    content_score *= content_weight
                    style_score *= style_weight
                    if enableTVLoss:
                        w_variance = torch.mean(torch.pow(noGradRGBTarget[:, :, :, :-1] - noGradRGBTarget[:, :, :, 1:], 2))
                        h_variance = torch.mean(torch.pow(noGradRGBTarget[:, :, :-1, :] - noGradRGBTarget[:, :, 1:, :], 2))
                        img_tv_loss =  (h_variance + w_variance) / 2.0 * 30000.0   
                        loss = (style_score + content_score)   + img_tv_loss 
                    else:
                        loss = (style_score + content_score)
                    # loss = F.mse_loss(noGradRGBTarget , refImg) * 10000
                    
                    
                    
                    if USECCPL:
                        loss_ccp = CPPLModel(refImg, noGradRGBTarget) * CPPL_Weight
                    
                        # loss += loss_ccp
                        loss_ccp.backward()
                    loss.backward()
                    print('current loss = ' + str(loss.item()))
                    print('current content_score = ' + str(content_score.item()))
                    print('current style_score = ' + str(style_score.item()))
                    if enableTVLoss:
                        print('img_tv_loss_score = ' + str(img_tv_loss.item() ))
                    if USECCPL:
                        print('CCP loss = ' + str(loss_ccp.item() ))
                    logger.info('current loss = ' + str(loss.item()))
                    logger.info('current content_score = ' + str(content_score.item()))
                    logger.info('current style_score = ' + str(style_score.item()))
                    logger.warning('current content_score = ' + str(content_score.item()))
                    logger.warning('current style_score = ' + str(style_score.item()))
                    RGBLoss = noGradRGBTarget.grad.detach().clone()
                    img = noGradRGBTarget.detach().clone().cpu()
                    del noGradRGBTarget
                    return RGBLoss,img

            def GetImageRGBLossWithPrevious(rays_o, rays_d, time_one, viewdirs,cam_one,Model,
                                            newVGG,new_styloss,FrameNum,
                                            content_weight = 100,
                                            **render_kwargs):

                # accumulate grad phase
                with torch.no_grad():
                    Renderresults = []
                    for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                            viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0)):
                        #with torch.no_grad():
                        results = Model.forward(ro, rd, vd,ts, cams,renderStyImgaeWithoutGrad = True, FrameNum= FrameNum,DeltaTime_T=deltaT, **render_kwargs)
                        Renderresults.append(results['rgb_marched'])
                    noGradRGBTarget = torch.cat(Renderresults).reshape(H,W,-1)
                    
                    noGradRGBTarget = noGradRGBTarget.permute(2, 0, 1)
                    noGradRGBTarget = noGradRGBTarget.unsqueeze(0)
                    # noGradRGBTarget = F.interpolate(noGradRGBTarget,scale_factor=(dialtedInterval,dialtedInterval),mode ='bilinear')
                    # 1 3 N N
                noGradRGBTarget.requires_grad_(True)
                style_score =0
                content_score = 0
                newVGG(noGradRGBTarget)
                VGGmodel(noGradRGBTarget)
                count =0 
                for sl in new_styloss:
                    style_score += sl.loss
                    count = count+1
                
                count =0 
                for cl in content_losses:
                    with torch.no_grad():
                        feature_vector = content_models[count](contentImages[current_Frame].to('cuda')).detach().clone()
                    content_score += cl.loss(feature_vector)
                    count = count+1
                
            
                # style_weight = 1000000
                
                refImg = contentImages[current_Frame].to('cuda')
                # style_weight = 0
                # content_weight = 0
                content_score *= content_weight
                style_score *= style_weight
                
                if enableTVLoss:
                    w_variance = torch.mean(torch.pow(noGradRGBTarget[:, :, :, :-1] - noGradRGBTarget[:, :, :, 1:], 2))
                    h_variance = torch.mean(torch.pow(noGradRGBTarget[:, :, :-1, :] - noGradRGBTarget[:, :, 1:, :], 2))
                    img_tv_loss =  (h_variance + w_variance) / 2.0 * 30000.0   
                    loss = (style_score + content_score)   + img_tv_loss 
                else:
                    loss = (style_score + content_score)
                # loss = F.mse_loss(noGradRGBTarget , refImg) * 10000
                
                
                
                if USECCPL:
                    loss_ccp = CPPLModel(refImg, noGradRGBTarget) * CPPL_Weight
                
                    # loss += loss_ccp
                    loss_ccp.backward()
                loss.backward()
                print('current loss = ' + str(loss.item()))
                print('current content_score = ' + str(content_score.item()))
                print('current style_score = ' + str(style_score.item()))
                if enableTVLoss:
                    print('img_tv_loss_score = ' + str(img_tv_loss.item() ))
                if USECCPL:
                    print('CCP loss = ' + str(loss_ccp.item() ))
                logger.info('current loss = ' + str(loss.item()))
                logger.info('current content_score = ' + str(content_score.item()))
                logger.info('current style_score = ' + str(style_score.item()))
                logger.warning('current content_score = ' + str(content_score.item()))
                logger.warning('current style_score = ' + str(style_score.item()))
                RGBLoss = noGradRGBTarget.grad.detach().clone()
                img = noGradRGBTarget.detach().clone().cpu()
                del noGradRGBTarget
                return RGBLoss,img
    

            def GetOriginImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one,Model,FrameNum, **render_kwargs):
                # accumulate grad phase
                with torch.no_grad():
                    Renderresults = []
                    for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                                viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0)):
                            #with torch.no_grad():
                        results = Model.forward(ro, rd, vd,ts, cams,renderStyImgaeWithoutGrad = True,stylizied_deformed_Scale = 0.0, FrameNum= FrameNum,DeltaTime_T=deltaT,  **render_kwargs)
 
                        noGrad_RGB_Results = results['rgb_marched']
                        Renderresults.append(noGrad_RGB_Results)
                        # depthResult = noGradResults['depth']
                        # Depthresults.append(depthResult)
                        
                    noGradRGBTarget = torch.cat(Renderresults)
                    noGradRGBTarget = noGradRGBTarget.reshape([H,W,3])
                    
                    noGradRGBTarget = noGradRGBTarget.permute(2, 0, 1)
                    noGradRGBTarget = noGradRGBTarget.unsqueeze(0)
                    # noGradRGBTarget = F.interpolate(noGradRGBTarget,scale_factor=(dialtedInterval,dialtedInterval),mode ='bilinear')
                    # 1 3 N N
                noGradRGBTarget.requires_grad_(True)
                if enableColorTransfer_ARF:
                    # noGradRGBTarget_altered = utils.apply_CT(noGradRGBTarget,color_tr)
                    noGradRGBTarget_altered = utils.match_colors_for_image_set_newShape(noGradRGBTarget,style_img)
                else:
                    noGradRGBTarget_altered = noGradRGBTarget
                
                refImg = contentImages[current_Frame].to('cuda')
                Imagedelta = noGradRGBTarget_altered-refImg
                w_variance = torch.mean(torch.pow(Imagedelta[:, :, :, :-1] - Imagedelta[:, :, :, 1:], 2))
                h_variance = torch.mean(torch.pow(Imagedelta[:, :, :-1, :] - Imagedelta[:, :, 1:, :], 2))
                img_tv_loss =  (h_variance + w_variance) / 2.0 * allign_variance_weight 
                loss = OriginalImageSimilarity(noGradRGBTarget_altered , refImg) * (ori_score * 0.00000001)
                
                # loss = loss * (ori_score * 0.001)
                
                print('Origin Image loss = ' + str(loss.item()))
                logger.info('Origin Image loss = ' + str(loss.item()))
                logger.warning('Origin Image loss = ' + str(loss.item()))
                loss += img_tv_loss * 5000* 100000.0
                loss.backward()
                print('Origin Image tv loss = ' + str(img_tv_loss.item() * 5000* 100000.0))
                logger.info('Origin Image tv loss = ' + str(img_tv_loss.item() * 5000* 100000.0))
                RGBLoss = noGradRGBTarget.grad.detach().clone()
                img= noGradRGBTarget_altered.detach().clone().cpu()
                del noGradRGBTarget
                return RGBLoss,img   

            def GetOP_reference_ImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one,Model,FrameNum,content_weight = 100, NNFMContent = NNFMContent, style_weight=style_weight,**render_kwargs):
                # accumulate grad phase

                with torch.no_grad():
                    Renderresults = []
                    for ro, rd, vd ,ts,cams in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                            viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0)):
                        #with torch.no_grad():
                        results = Model.forward(ro, rd, vd,ts, cams,renderStyImgaeWithoutGrad = True, FrameNum= FrameNum,DeltaTime_T=deltaT, **render_kwargs)
                        Renderresults.append(results['rgb_marched'])
                    noGradRGBTarget = torch.cat(Renderresults).reshape(H,W,-1)
                    
                    
                    noGradRGBTarget = noGradRGBTarget.permute(2, 0, 1)
                    noGradRGBTarget = noGradRGBTarget.unsqueeze(0)
                    # noGradRGBTarget = F.interpolate(noGradRGBTarget,scale_factor=(dialtedInterval,dialtedInterval),mode ='bilinear')
                    # 1 3 N N
                noGradRGBTarget.requires_grad_(True)

                if enableColorTransfer_ARF:
                    # noGradRGBTarget_altered = utils.apply_CT(noGradRGBTarget,color_tr)
                    noGradRGBTarget_altered = utils.match_colors_for_image_set_newShape(noGradRGBTarget,style_img)
                else:
                    noGradRGBTarget_altered = noGradRGBTarget
                
                refImg = OPSyn_Target_Reference
                # print(OPSyn_Target_Reference.shape)
                # print(refImg.shape)
                # print('---')
                # print(OPSyn_Target_Reference[0,:,0,0])
                # print(refImg[0,:,0,0])
                # print(noGradRGBTarget_altered)
                
                Imagedelta = noGradRGBTarget_altered-refImg
                w_variance = torch.mean(torch.pow(Imagedelta[:, :, :, :-1] - Imagedelta[:, :, :, 1:], 2))
                h_variance = torch.mean(torch.pow(Imagedelta[:, :, :-1, :] - Imagedelta[:, :, 1:, :], 2))
                img_tv_loss =  (h_variance + w_variance) / 2.0 * allign_variance_weight 
                loss = OPReferenceSimilarity(noGradRGBTarget_altered , refImg) 
                print('OP refer Image loss = ' + str(loss.item()))
                logger.info('OP refer Image loss = ' + str(loss.item()))
                logger.warning('OP refer Image loss = ' + str(loss.item()))
                loss = loss * (ori_score * 0.1)
                print('OP refer Image weighted loss = ' + str(loss.item()))
                # loss += img_tv_loss * 5000* 100000.0
                loss_new = loss
                loss.backward()
                print('OP refer Image tv loss = ' + str(img_tv_loss.item() * 5000* 100000.0))
                logger.info('OP refer Image tv loss = ' + str(img_tv_loss.item() * 5000* 100000.0))
                RGBLoss = noGradRGBTarget.grad.detach().clone()
                img = noGradRGBTarget_altered.detach().clone().cpu()
                
                
                # rgb = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # rgb8 = utils.to8b(rgb)
                # filename = os.path.join(stylizied_path, 'testnow' + '{:03d}.png'.format(WriteNameFrame))
                # imageio.imwrite(filename, rgb8)
                
                # rgb = refImg.squeeze(0).permute(1, 2, 0).cpu().numpy()
                # rgb8 = utils.to8b(rgb)
                # filename = os.path.join(stylizied_path, 'testcompare' + '{:03d}.png'.format(WriteNameFrame))
                # imageio.imwrite(filename, rgb8)
                
                del noGradRGBTarget
                return RGBLoss,img
                
                # curImage = contentImages[current_Frame].cuda()      
                # w_variance = torch.mean(torch.pow(noGradRGBTarget_altered[:, :, :, :-1] - noGradRGBTarget_altered[:, :, :, 1:], 2))
                # h_variance = torch.mean(torch.pow(noGradRGBTarget_altered[:, :, :-1, :] - noGradRGBTarget_altered[:, :, 1:, :], 2))
                # if enableTVLoss:
                #     img_tv_loss =  (h_variance + w_variance) / 2.0
                # else:
                #     img_tv_loss = 0.0
                # loss_dict = nnfm_loss_fn(
                # noGradRGBTarget_altered,
                # style_img,
                # blocks=[
                # 2,
                # ],
                # loss_names=["nnfm_loss", "content_loss"],
                # contents=curImage,
                # )
                # loss_dict["content_loss"] *= 1e-3 * NNFMContent * style_weight_ori
                # loss_dict["img_tv_loss"] = img_tv_loss * 0.000001 * 1000000
                # loss_dict["nnfm_loss"] = loss_dict["nnfm_loss"] * style_weight
                # # style_weight = 1000000
                # loss_ccp = 0.0
                
                
                
                # loss = sum(list(loss_dict.values()))  * OP_reference_continueStyleLoss
                
                # if enableColorTransfer_ARF_oriMode:
                #     if style_weight == 0:
                #         loss = torch.mean((curImage-noGradRGBTarget_altered)**2) * MSE_LR_Real
                #         print('mse loss')
                
                # refImg = contentImages[current_Frame].to('cuda')
                # # print(refImg.shape)
                # # print(noGradRGBTarget.shape)
                # # return
                # if USECCPL:
                #     loss_ccp = CPPLModel(refImg, noGradRGBTarget_altered) * CPPL_Weight
                #     loss += loss_ccp
                # print('current loss = ' + str(loss.item()))
                # print('current content_score = ' + str(loss_dict["content_loss"].item() ))
                # print('current style_score = ' + str(loss_dict["nnfm_loss"].item()   ))
                # if enableTVLoss:
                #     print('current tvkloss = ' + str(img_tv_loss.item() * 10000))
                # if USECCPL:
                #     print('CCP loss = ' + str(loss_ccp.item() ))
                # # logging.info('current loss = ' + str(loss.item()))
                # # logging.info('current content_score = ' + str(content_score.item()))
                # # logging.info('current style_score = ' + str(style_score.item()))
                # loss += loss_new
                # loss.backward()
                # RGBLoss = noGradRGBTarget.grad.detach().clone()
                # img = noGradRGBTarget_altered.detach().clone().cpu()
                # del noGradRGBTarget
                # return RGBLoss,img
            
            
            
                
                
                
                
                
                
                RGBLoss = noGradRGBTarget.grad.detach().clone()
                img= noGradRGBTarget_altered.detach().clone().cpu()
                del noGradRGBTarget
                return RGBLoss,img   


            Loss_ScaleItem = 65536
            Loss_ScaleItem = 1.0
            rgbResult_origin = None
            if j != targetFrameIndex:
                OPSyn_Target_Reference = synthesizeImage(Stylizied_Results[current_Save_Frame-1],contentImages[current_Save_Frame-1],contentImages[current_Save_Frame],current_Save_Frame)
            
            if Enable_allignTrain:     
                bacth_size = batch_base *2  *2
                TotalModel2 = Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth(NerfPreTrained_model, stylizedDeformGrid_FrameIndexed[current_Save_Frame].to('cuda'),
                                                                                                stylizedDeformGrid_FrameIndexed[current_Save_Frame-1].to('cuda'),
                                                                                                DeltaDeformGrid_FrameIndexed[current_Save_Frame].to('cuda'),
                                                                                                DeltaDeformGrid_FrameIndexed[current_Save_Frame-1].to('cuda')
                                                                                                ) 
                
                TotalModel = TotalModel2
                TotalModel.EnableRefMode = False
                if EnableTrainColor:
                    TotalModel.setStyRadField(copy.deepcopy(styRadNet_FrameIndexed[current_Save_Frame]).cuda(),styRadNet_FrameIndexed[current_Save_Frame-1].cuda())
                if UsingBeforeImageTraining:
                    with torch.no_grad():
                        prev_VGGmodel, prev_style_losses, _,_ = get_style_model_and_losses_withModels(cnn,
                        cnn_normalization_mean, cnn_normalization_std, 
                        Stylizied_Results[current_Save_Frame - 1].cuda()
                        ,content_layers_default, style_layers_default)
                        
                    
                    prev_VGGmodel.eval()
                    prev_VGGmodel.requires_grad_(False)

                
                with torch.no_grad():
                    if global_step == Dynamic_OPT_Start_Frame:
                        TotalModel.Init_TimeDeltaField(SampleTime,DeltaTime_T=deltaT)
                            
                optimizer = utils.create_optimizer_for_deltaSmooth(TotalModel, cfg_train, global_step=0,lrscale= lrscale_deform_rebuild,lrate_decay=cfg_train.lrate_decay)
                
                
            
                # dynamic 对齐deform train
                for substeps in range(align_substeps_count):
                    # with profile(activities=[ProfilerActivity.CPU,
                    # ProfilerActivity.CUDA], record_shapes=True) as prof:
                    #     with record_function("model_inference"):
                    if j == 0:
                        break
                    TotalModel = Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth(NerfPreTrained_model, stylizedDeformGrid_FrameIndexed[current_Save_Frame].to('cuda'),
                                                                                                            stylizedDeformGrid_FrameIndexed[current_Save_Frame-1].to('cuda'),
                                                                                                            TotalModel.TimeDeltaField,DeltaDeformGrid_FrameIndexed[current_Save_Frame-1].to('cpu')
                                                                                                            ,
                                                                                                            prevModel=TotalModel
                                                                                                            ) 
                    if EnableTrainColor:
                        TotalModel.setStyRadField(styRadNet_FrameIndexed[current_Save_Frame].cuda(),styRadNet_FrameIndexed[current_Save_Frame-1].cuda())
                    TotalModel.EnableRefMode = False
                    torch.cuda.synchronize()
                    time_a = time.time()
                    
                    # with autocast():
                    with torch.no_grad():
                        TotalModel.prepare(SampleTime,OverWriteAdvection=False, DeltaTime_T = deltaT,alpha = 0.01)                   

                    torch.cuda.synchronize()
                    time_b = time.time()     
                    

                    if UsingBeforeImageTraining:
                        RGBGradTotal, rgbResult= GetImageRGBLossWithPrevious(rays_o, rays_d, time_one, viewdirs,cam_one, TotalModel,
                                                                            prev_VGGmodel,prev_style_losses,
                                                                            content_weight = 1000,
                                                                            FrameNum= FrameNum,**render_kwargs)
                    else:
                        RGBGradTotal, rgbResult = GetOP_reference_ImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one, TotalModel,
                                                                            # prev_VGGmodel,prev_style_losses,
                                                                            content_weight = 1000,NNFMContent = 10,
                                                                            FrameNum= FrameNum,**render_kwargs)
                    RGBGradTotal = RGBGradTotal.squeeze(0)
                    RGBGradTotal = RGBGradTotal.permute(1,2,0)
                    RGBGradTotal = RGBGradTotal.reshape(-1,3)            
                    
                    torch.cuda.synchronize()
                    time_c = time.time()    

                    TotalModel.AdaptStyRaidanceFieldMode = False
                    RGBGradTotal_origin, rgbResult_origin = GetOriginImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one, TotalModel, FrameNum= FrameNum,**render_kwargs)                   
                    RGBGradTotal_origin = RGBGradTotal_origin.squeeze(0)
                    RGBGradTotal_origin = RGBGradTotal_origin.permute(1,2,0)
                    RGBGradTotal_origin = RGBGradTotal_origin.reshape(-1,3)            

                    torch.cuda.synchronize()
                    time_d = time.time()    
                    optimizer.zero_grad(set_to_none=True)
                    depthlossTotal = 0.0

                    for ro, rd, vd ,ts,cams,grad_before,depthimg_slice in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                                    viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0),RGBGradTotal_origin.split(bacth_size, 0),depthImages[j].split(bacth_size, 0) ):
                                #with torch.no_grad():
                        results = TotalModel.forward(ro, rd, vd,ts, cams, Enablek0Bypass=True,stylizied_deformed_Scale=0.0,FrameNum= FrameNum,DeltaTime_T=deltaT, 
                                                     **render_kwargs)        
                        
                        noGrad_RGB_Results = results['rgb_marched']
                        depth_withGrad = results['depth']
                        refDepth = depthimg_slice.cuda()
                        # print(refDepth.shape)
                        # print(depth_withGrad.shape)
                        # 1 N 3
                        rgbgrad = grad_before
                        loss =  torch.sum(rgbgrad * noGrad_RGB_Results)
                        #TODO sth wrong with depth loss
                        # depthloss =  F.mse_loss(refDepth , depth_withGrad) #  * ori_score
                        # depthlossTotal += depthloss.item()
                        # loss += depthloss
                        loss = loss / Loss_ScaleItem
                        loss.backward()
                        
                    print('depth loss = ' + str(depthlossTotal))
                    torch.cuda.synchronize()
                    time_e = time.time()    
                    # TotalModel.DynamicRender.deformgrid.eval()
                    # TotalModel.DynamicRender.deformgrid.requires_grad_(False)   


                    
                    TotalModel.AdaptStyRaidanceFieldMode = True
                    print('rec')
                    with torch.no_grad():
                        TotalModel.prepare(SampleTime,OverWriteAdvection=False , DeltaTime_T = deltaT,alpha = 0.01,UseHalf = False)        
                    print('rec1')
                    TotalModel.StylizedDeformVol_current_k0Bypass.requires_grad_(True)
                    if TotalModel.StylizedDeformVol_current_k0Bypass_stylzied != None:
                        TotalModel.StylizedDeformVol_current_k0Bypass_stylzied.requires_grad_(True)
                        
                    torch.cuda.synchronize()
                    time_f = time.time()
                    
                    
                    for ro, rd, vd ,ts,cams,grad_before in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                                    viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0),RGBGradTotal.split(bacth_size, 0) ):
                                #with torch.no_grad():
                        results = TotalModel.forward(ro, rd, vd,ts, cams, Enablek0Bypass=True,FrameNum= FrameNum,stylizied_deformed_Scale= stylizied_deformed_Scale_global, DeltaTime_T=deltaT, **render_kwargs)                  

                        noGrad_RGB_Results = results['rgb_marched']
                        # 1 N 3s
                        rgbgrad = grad_before


                        loss = sty_score * torch.sum(rgbgrad * noGrad_RGB_Results) #  + noGradResults['deform_loss'] * deform_approach_weight
                        loss = loss / Loss_ScaleItem
                        loss.backward()
                        #halfPrecscaler.scale(loss).backward()

                        # gradAccmulate += TotalModel.StylizedDeformVol_current_k0Bypass.grad
                        # loss.backward(retain_graph=True)
                    print('rec1.5')
                    torch.cuda.synchronize()
                    time_g = time.time()   
                    # self.StylizedDeformVol_current_k0Bypass_stylzied
                    gradAccmulate = TotalModel.StylizedDeformVol_current_k0Bypass.grad.detach().clone()
                    if TotalModel.StylizedDeformVol_current_k0Bypass_stylzied != None:
                        gradAccmulate_styrad = TotalModel.StylizedDeformVol_current_k0Bypass_stylzied.grad.detach().clone()
                    # TotalModel.DynamicRender.deformgrid.train()
                    # TotalModel.DynamicRender.deformgrid.requires_grad_(True)
                    torch.cuda.empty_cache()

                    TotalModel.prepare(SampleTime,OverWriteAdvection=False , DeltaTime_T = deltaT,alpha = 0.01,UseHalf = False)        
                
                    loss =  torch.sum(TotalModel.StylizedDeformVol_current_k0Bypass * gradAccmulate) #  + noGradResults['deform_loss'] * deform_approach_weight  
                    if TotalModel.StylizedDeformVol_current_k0Bypass_stylzied != None:
                        loss +=  torch.sum(TotalModel.StylizedDeformVol_current_k0Bypass_stylzied * gradAccmulate_styrad)                      
                    torch.cuda.synchronize()
                    time_h = time.time()  
                    print('rec2')
                    loss.backward()
                    torch.cuda.empty_cache()
                    del TotalModel.StylizedDeformVol_current_k0Bypass
                    del TotalModel.StylizedDeformVol_current_k0Bypass_stylzied

                    optimizer.step()
                        
                    # halfPrecscaler.step(optimizer)
                    # halfPrecscaler.update()
                    print('rec3')
                    torch.cuda.synchronize()
                    time_i = time.time()
                    print(time_b - time_a)
                    print(time_c - time_b) # od
                    print(time_d - time_c) # od
                    print(time_e - time_d) # od
                    print(time_f - time_e)
                    print(time_g - time_f) # od
                    print(time_h - time_g)
                    print(time_i - time_h)

                    logger.info(time_b - time_a)
                    logger.info(time_c - time_b)
                    logger.info(time_d - time_c)
                    logger.info(time_e - time_d)
                    logger.info(time_f - time_e)
                    logger.info(time_g - time_f)
                    logger.info(time_h - time_g)
                    logger.info(time_i - time_h)

                    # print(prof.key_averages().table(sort_by="cuda_time_total"))
                logger.warning('End allignment')    
            
            if not FreezeOpt:
                DeltaDeformGrid_FrameIndexed[current_Save_Frame] = TotalModel.TimeDeltaField.to('cpu')
  


            
            TotalModel = None
            if j == targetFrameIndex:
                TotalModel = Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth(NerfPreTrained_model, stylizedDeformGrid_FrameIndexed[j].to('cuda'), 
                                                                                                                stylizedDeformGrid_FrameIndexed[j].to('cuda'),DeltaDeformGrid_FrameIndexed[current_Save_Frame].to('cuda'),
                                                                                                                DeltaDeformGrid_FrameIndexed[1].to('cpu') 
                                                                                                                ) 
                TotalModel.EnableRefMode = False
                if EnableTrainColor:
                    TotalModel.setStyRadField(styRadNet_FrameIndexed[0].cuda(),styRadNet_FrameIndexed[0].cuda())
                
                with torch.no_grad():
                    TotalModel.prepare(SampleTime,OverWriteAdvection=True,DeltaTime_T = deltaT,alpha = 0.01)
                # print(stylizedDeformGrid_FrameIndexed)
                print('is first frame')
                optimizer = utils.create_optimizer_for_styliziedDefromationGrid_transportBased_withAdditionalRadField(TotalModel, cfg_train, global_step=0,lrscale= lrscale,lrate_decay=cfg_train.lrate_decay,
                                                                                                opt_prevField=False,opt_firstField=True,opt_radiance_filed=Enable_RaidanceTrain,
                                                                                                opt_radNet = True,opt_OriNeRFModelColors = optOriNeRFColorField,coloroptScale = coloroptScale_bgein
                                                                                                )
                

                
            else:
                # lrscale = 0.001

                TotalModel = Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth(NerfPreTrained_model, stylizedDeformGrid_FrameIndexed[current_Save_Frame].to('cuda'), 
                                                                                                                stylizedDeformGrid_FrameIndexed[current_Save_Frame-1].to('cuda'),DeltaDeformGrid_FrameIndexed[current_Save_Frame].to('cuda'),
                                                                                                                DeltaDeformGrid_FrameIndexed[current_Save_Frame-1].to('cuda') 
                                                                                                                ) 
                TotalModel.EnableRefMode = False
                if PureAdvectionOnly:
                    print('hit')
                    print('deltaT')
                    TotalModel.Init_TimeDeltaField(SampleTime,DeltaTime_T=deltaT)
                if EnableTrainColor:
                    TotalModel.setStyRadField(styRadNet_FrameIndexed[current_Save_Frame].cuda(),styRadNet_FrameIndexed[current_Save_Frame-1].cuda())
                
                with torch.no_grad():
                    TotalModel.prepare(SampleTime,OverWriteAdvection=False,DeltaTime_T = deltaT,alpha = finalPrepareAlpha)
 
                optimizer = utils.create_optimizer_for_styliziedDefromationGrid_transportBased_withAdditionalRadField(TotalModel, cfg_train, global_step=0,lrscale= lrscale,lrate_decay=cfg_train.lrate_decay,
                                                                                            opt_prevField=False,opt_firstField=True,opt_radiance_filed=Enable_RaidanceTrain,
                                                                                            opt_radNet = False,opt_OriNeRFModelColors = (optOriNeRFColorField & EnableTrainColor_Continue),coloroptScale = coloroptScale
                                                                                            )     
            torch.cuda.empty_cache()
            if CCPLOptMLP:
                optimizerCCPL = torch.optim.Adam(itertools.chain(CPPLModel.CCPL.parameters()), lr=0.001)
                CPPLModel.CCPL.requires_grad_(True)
            # for param_group in optimizer.param_groups:# 
            #     tensors = (param_group['params'])
            #     for ten in tensors:
            #         print(ten.grad)


            
            

                
                  
            TotalModel.EnableRefMode = False
            # substeps_count = 500
            for substeps in range(substeps_count):
                
                if enableColorTransfer_ARF and j == targetFrameIndex and enableColorTransfer_ARF_oriMode:
                    if substeps < 20:
                        style_weight = 0
                        stylizied_deformed_Scale_global = 0.0
                    else:
                        style_weight = style_weight_ori
                        stylizied_deformed_Scale_global = 1.0
                else:
                    style_weight = style_weight_ori
                
                if CCPLOptMLP:
                    optimizerCCPL.zero_grad()
                
                bacth_size =  batch_base *1 
                
                # RGBGradTotal, rgbResult= GetOriginImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one, TotalModel, FrameNum= FrameNum,**render_kwargs)
                RGBGradTotal, rgbResult= GetStyliziedImageRGBLoss(rays_o, rays_d, time_one, viewdirs,cam_one, TotalModel, FrameNum= FrameNum,
                                                                  content_weight = 100,style_weight=style_weight,
                                                                  **render_kwargs)
                if CCPLOptMLP:
                    optimizerCCPL.step()
                
                print('------------')
                RGBGradTotal = RGBGradTotal.squeeze(0)
                RGBGradTotal = RGBGradTotal.permute(1,2,0)
                RGBGradTotal = RGBGradTotal.reshape(-1,3)            
                # RGBGradTotal = RGBGradTotal.unsqueeze(0)
                
                # optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad()   

                # print(stylizied_model.StylizedDeformVol.k0)
                
                for ro, rd, vd ,ts,cams,grad_before in zip(rays_o.split(bacth_size, 0), rays_d.split(bacth_size, 0),
                                                                viewdirs.split(bacth_size, 0),time_one.split(bacth_size, 0),cam_one.split(bacth_size, 0),RGBGradTotal.split(bacth_size, 0) ):
                            #with torch.no_grad():
                    results = TotalModel.forward(ro, rd, vd,ts, cams, FrameNum= FrameNum, stylizied_deformed_Scale= stylizied_deformed_Scale_global,DeltaTime_T=deltaT, **render_kwargs)
                    noGrad_RGB_Results = results['rgb_marched']

 
                    # 1 N 3
                    rgbgrad = grad_before
                    # print(rgbgrad.shape)
                    # print(RGBGradTotal.shape)
                    # print(noGrad_RGB_Results.shape)
                    loss = torch.sum(rgbgrad * noGrad_RGB_Results)  #  + noGradResults['deform_loss'] * deform_approach_weight
                    loss.backward()
                    del loss
                
                # del rgbgrad
                # import gc
                # gc.collect()
                if stylized_deform_tv_weight > 0:
                    dtv = TotalModel.StylizedDeformVol.k0_total_variation()
                    loss = stylized_deform_tv_weight * dtv
                    loss.backward()
                    del loss
                    
                TotalModel.stySmoothe()
                optimizer.step() 


            if USECCPL or j == targetFrameIndex:
                trainCCPL(CPPLModel,contentImages[current_Frame].cuda(),rgbResult.cuda(),5000,0.001)
            

            stylizedDeformGrid_FrameIndexed[current_Save_Frame] = (TotalModel.StylizedDeformVol.to('cpu'))
            if TotalModel.StylizedRadianceField_prev!=None:
                TotalModel.StylizedRadianceField_prev.to('cpu')
            if EnableTrainColor:
                if TotalModel.StylizedRadianceField!=None:
                    styRadNet_FrameIndexed[current_Save_Frame] = TotalModel.StylizedRadianceField.to('cpu')
                if current_Save_Frame != 0:
                    styRadNet_FrameIndexed[current_Save_Frame - 1] = TotalModel.StylizedRadianceField_prev.to('cpu')
            
            
                if j == 0:
                    if styRadNet_FrameIndexed[0].rgbnet !=None:
                        for i in range(len(styRadNet_FrameIndexed)):
                            styRadNet_FrameIndexed[i].rgbnet = styRadNet_FrameIndexed[0].rgbnet.to('cpu')
            
                
                
            Stylizied_Results[current_Save_Frame] = rgbResult
            
            # if enableColorTransfer_ARF:
            #     rgbResult, _ = utils.match_colors_for_image_set(rgbResult,style_img)
            
            rgb = rgbResult.squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgb_ori = None
            if (rgbResult_origin)!= None:
                
                rgb_ori = rgbResult_origin.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
            
            # contentImages[current_Frame]
            if stylizied_path is not None:
                rgb8 = utils.to8b(rgb)
                filename = os.path.join(stylizied_path, 'test' + '{:03d}.png'.format(WriteNameFrame))
                imageio.imwrite(filename, rgb8)
                
                rgb8 = utils.to8b(contentImages[current_Frame].squeeze(0).permute(1, 2, 0).cpu().numpy() )
                filename = os.path.join(stylizied_path, 'ori' + '{:03d}.png'.format(WriteNameFrame))
                imageio.imwrite(filename, rgb8)
                
                if (rgbResult_origin)!= None:
                    rgb8 = utils.to8b(rgb_ori)
                    filename = os.path.join(stylizied_path, 'oritest' + '{:03d}.png'.format(WriteNameFrame))
                    imageio.imwrite(filename, rgb8)
                    del rgbResult_origin
            del rgbResult
            
            
            if j == targetFrameIndex:
                stylizedDeformGrid_FrameIndexed[current_Save_Frame].to('cpu')
                DeltaDeformGrid_FrameIndexed[current_Save_Frame].to('cpu')
                if EnableTrainColor:
                    styRadNet_FrameIndexed[current_Save_Frame].to('cpu')
            else:
                stylizedDeformGrid_FrameIndexed[current_Save_Frame].to('cpu')
                stylizedDeformGrid_FrameIndexed[current_Save_Frame-1].to('cpu')
                if EnableTrainColor:
                    styRadNet_FrameIndexed[current_Save_Frame].to('cpu')
                    styRadNet_FrameIndexed[current_Save_Frame - 1].to('cpu')
                DeltaDeformGrid_FrameIndexed[current_Save_Frame].to('cpu')
                DeltaDeformGrid_FrameIndexed[current_Save_Frame-1].to('cpu')

            # for locates in 
            # gc.collect()
            torch.cuda.empty_cache()
        
    
        # if global_step%(args.fre_test) == 0:
        #     render_viewpoints_kwargs = {
        #         'model': model,
        #         'ndc': cfg.data.ndc,
        #         'render_kwargs': {
        #             'near': near,
        #             'far': far,
        #             'bg': 1 if cfg.data.white_bkgd else 0,
        #             'stepsize': cfg_model.stepsize,

        #             },
        #         }
        #     testsavedir = os.path.join(cfg.basedir, cfg.expname, f'{global_step}-test')
        #     if os.path.exists(testsavedir) == False:
        #         os.makedirs(testsavedir)
        #     if cfg.data.dataset_type != 'hyper_dataset': 
        #         rgbs,disps = render_viewpoints(
        #             render_poses=data_dict['poses'][data_dict['i_test']],
        #             HW=data_dict['HW'][data_dict['i_test']],
        #             Ks=data_dict['Ks'][data_dict['i_test']],
        #             gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
        #             savedir=testsavedir,
        #             test_times=data_dict['times'][data_dict['i_test']],
        #             eval_psnr=args.eval_psnr, eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
        #             **render_viewpoints_kwargs)
        #     else:
        #         rgbs,disps = render_viewpoints_hyper(
        #             data_class=data_class,
        #             savedir=testsavedir, all=False, test=True, eval_psnr=args.eval_psnr,
        #             **render_viewpoints_kwargs)



def train(args, cfg, data_dict=None):

    # init
    print('train: start')
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # coarse geometry searching
    if cfg.data.dataset_type == 'hyper_dataset':
        xyz_min, xyz_max,BBX_Time = compute_bbox_by_cam_frustrm_hyper(args = args, cfg = cfg,data_class = data_dict['data_class'])
    else:
        xyz_min, xyz_max,BBX_Time = compute_bbox_by_cam_frustrm(args = args, cfg = cfg, **data_dict)
    coarse_ckpt_path = None

    # fine detail reconstruction
    eps_time = time.time()
    scene_rep_reconstruction(
            args=args, cfg=cfg,
            cfg_model=cfg.model_and_render, cfg_train=cfg.train_config,
            xyz_min=xyz_min, xyz_max=xyz_max,BBX_Time=BBX_Time,
            data_dict=data_dict)
    eps_loop = time.time() - eps_time
    eps_time_str = f'{eps_loop//3600:02.0f}:{eps_loop//60%60:02.0f}:{eps_loop%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')

if __name__=='__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()
    data_dict = None
    # load images / poses / camera settings / data split
    data_dict = load_everything(args = args, cfg = cfg)

    # train
    if not args.render_only :
        train(args, cfg, data_dict = data_dict)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        if args.ft_path:
            ckpt_path = args.ft_path
        else:
            ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model_class = tineuvox.TiNeuVox
        model = utils.load_model(model_class, ckpt_path).to(device)
        near=data_dict['near']
        far=data_dict['far']
        stepsize = cfg.model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': near,
                'far': far,
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'render_depth': True,
            },
        }
    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok = True)
        if cfg.data.dataset_type  != 'hyper_dataset':
            rgbs, disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_train']],
                    HW=data_dict['HW'][data_dict['i_train']],
                    Ks=data_dict['Ks'][data_dict['i_train']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_train']],
                    eval_psnr=args.eval_psnr, eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
        elif cfg.data.dataset_type == 'hyper_dataset':   
            rgbs,disps = render_viewpoints_hyper(
                    data_calss=data_dict['data_calss'],
                    savedir=testsavedir, all=True, test=False,
                    eval_psnr=args.eval_psnr,
                    **render_viewpoints_kwargs)
        else:
            raise NotImplementedError

        imageio.mimwrite(os.path.join(testsavedir, 'train_video.rgb.mp4'), utils.to8b(rgbs), fps = 30, quality = 8)
        imageio.mimwrite(os.path.join(testsavedir, 'train_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps = 30, quality = 8)

    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        if cfg.data.dataset_type  != 'hyper_dataset':
            rgbs, disps = render_viewpoints(
                    render_poses=data_dict['poses'][data_dict['i_test']],
                    HW=data_dict['HW'][data_dict['i_test']],
                    Ks=data_dict['Ks'][data_dict['i_test']],
                    gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                    savedir=testsavedir,
                    test_times=data_dict['times'][data_dict['i_test']],
                    eval_psnr=args.eval_psnr,eval_ssim = args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                    **render_viewpoints_kwargs)
        elif cfg.data.dataset_type == 'hyper_dataset':   
            rgbs,disps = render_viewpoints_hyper(
                    data_class=data_dict['data_class'],
                    savedir=testsavedir,all=True,test=True,
                    eval_psnr=args.eval_psnr,
                    **render_viewpoints_kwargs)
        else:
            raise NotImplementedError

        imageio.mimwrite(os.path.join(testsavedir, 'test_video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'test_video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality=8)

    # render video
    if args.render_video:
        if cfg.data.dataset_type  != 'hyper_dataset':
            testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}_time')
            os.makedirs(testsavedir, exist_ok=True)
            rgbs, disps = render_viewpoints(
                    render_poses=data_dict['render_poses'],
                    HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
                    render_factor=args.render_video_factor,
                    savedir=testsavedir,
                    test_times=data_dict['render_times'],
                    **render_viewpoints_kwargs)
            imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30, quality =8)
        else:
            raise NotImplementedError

    print('Done')

