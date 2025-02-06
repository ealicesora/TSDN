import os
import time
import math
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ThreedGausSmooth import *

from lib import utils
from lib.deformRegular import *
import copy
        
# 'cuda' 'ori' 'torch'
Deform_Grid_type = 'torch'

if Deform_Grid_type == 'cuda':
    # from lib.cuda_gridsample import *
    from grid_sample import cuda_gridsample as cu


disableViewDependentColor = True
OverWritedisableOcclusionMask = False
EnableAnotherElaMode = False
NotenableSemiLagragin = False

DisableGeoChanges = False

EnableNewAdvectionScehems = False
EnableIndependentMode = False

EnableStySmooth = True

OverwriteEmbedding = False


targetSlice_base = 4096 * 4

'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 nearest=False, pre_act_density=False, in_act_density=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_color_thres=0,
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 posbase_pe=5, viewbase_pe=4, use_appcode=False,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        print("----------Init Radiance Grid-----------")
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        print("dvgo: xyz_min, xyz_max: ", self.xyz_min, ", ", self.xyz_max)
        self.fast_color_thres = fast_color_thres
        self.nearest = nearest
        self.pre_act_density = pre_act_density
        self.in_act_density = in_act_density
        if self.pre_act_density:
            print('dvgo: using pre_act_density may results in worse quality !!')
        if self.in_act_density:
            print('dvgo: using in_act_density may results in worse quality !!')

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        # self.act_shift  =
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'posbase_pe': posbase_pe, 'viewbase_pe': viewbase_pe, 'use_appcode': use_appcode,
        }
        self.rgbnet_full_implicit = rgbnet_full_implicit
        self.use_appcode = use_appcode
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            if disableViewDependentColor:
                dim0 = (3+3*posbase_pe*2)
            else:
                dim0 = (3+3*posbase_pe*2) + (3+3*viewbase_pe*2)
            if self.use_appcode:
                dim0 += 1 + viewbase_pe * 2
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dvgo: feature voxel grid', self.k0.shape)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_path is not None and mask_cache_path:
            self.mask_cache = MaskCache(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self._set_nonempty_mask()
        else:
            self.mask_cache = None
            self.nonempty_mask = None
        print("--------------- Finish ----------------")

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_color_thres': self.fast_color_thres,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        # 这个buffer会被保存，但是不会被更新梯度
        
        self.density[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.density).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation(self):
        tv = total_variation(self.activate_density(self.density, 1), self.nonempty_mask)
        return tv

    def k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)
    

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox

    
    # accoding viewdirs and ray_pts get the density
    # question where is deform ?
    # sample points are points on rays and viedirs is only used for color
    def forward(self, rays_pts, mask_outbbox, interval, viewdirs, occlusion_mask=None, app_code=None, **render_kwargs):
        '''
            give alpha and rgb according to given positions
        '''
        # mask_cache false means not used
        # update mask for query points in known free space
        if self.mask_cache is not None:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))
        
        # query for alpha
        alpha = torch.zeros_like(rays_pts[...,0]).to(rays_pts)
        if self.pre_act_density:
            # pre-activation
            alpha[~mask_outbbox] = self.grid_sampler(
                    rays_pts[~mask_outbbox], self.activate_density(self.density, interval))
        elif self.in_act_density:
            # in-activation
            density = self.grid_sampler(rays_pts[~mask_outbbox], F.softplus(self.density + self.act_shift))
            alpha[~mask_outbbox] = 1 - torch.exp(-density * interval)
        else:
            # post-activation
            density = self.grid_sampler(rays_pts[~mask_outbbox], self.density).to(rays_pts)
            alpha[~mask_outbbox] = self.activate_density(density, interval).to(rays_pts)
        
        if occlusion_mask is not None:
            alpha = alpha * occlusion_mask.squeeze()
            
        # compute accumulated transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # query for color
        mask = (weights > self.fast_color_thres)
        k0 = torch.zeros(*weights.shape, self.k0_dim).to(weights)
        
        if not self.rgbnet_full_implicit:
            k0[mask] = self.grid_sampler(rays_pts[mask], self.k0)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[..., 3:]
                k0_diffuse = k0[..., :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
            rays_xyz = rays_pts[mask]
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)
            timestep_emb = (app_code.unsqueeze(-1) * self.viewfreq).flatten(-2)
            timestep_emb = torch.cat([app_code, timestep_emb.sin(), timestep_emb.cos()], -1)
            if self.use_appcode:
                rgb_feat = torch.cat([
                    k0_view[mask],
                    xyz_emb,
                    # TODO: use `rearrange' to make it readable
                    viewdirs_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)],
                    timestep_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)],
                ], -1)
            else:
                if disableViewDependentColor:
                    rgb_feat = torch.cat([
                        k0_view[mask],
                        xyz_emb,
                        # TODO: use `rearrange' to make it readable
                        # viewdirs_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)],
                    ], -1).to(rays_pts)
                else:
                    rgb_feat = torch.cat([
                        k0_view[mask],
                        xyz_emb,
                        # TODO: use `rearrange' to make it readable
                        viewdirs_emb.flatten(0,-2).unsqueeze(-2).repeat(1,weights.shape[-1],1)[mask.flatten(0,-2)],
                    ], -1).to(rays_pts)
            rgb_logit = torch.zeros(*weights.shape, 3).to(rays_pts)
            rgb_logit[mask] = self.rgbnet(rgb_feat)
            if self.rgbnet_direct:
                rgb = torch.sigmoid(rgb_logit)
            else:
                rgb_logit[mask] = rgb_logit[mask] + k0_diffuse
                rgb = torch.sigmoid(rgb_logit)
        if occlusion_mask is not None:
            rgb = rgb * occlusion_mask
        return  alpha, alphainv_cum, rgb, weights, mask


AddFeatureVector = False
class StylizedRadianceNet(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 posbase_pe=5, viewbase_pe=4, additionRGBNetwrokDim = 0,
                 **kwargs):
        super(StylizedRadianceNet, self).__init__()
        print("----------Init Radiance Grid-----------")
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        print("dvgo: xyz_min, xyz_max: ", self.xyz_min, ", ", self.xyz_max)



        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)


        # determine init grid resolution
        self._set_grid_resolution(num_voxels)
        

        # feature voxel grid + shallow MLP  (fine stage)

        self.k0_dim = 8
        
        self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]).cuda() )

        self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]).cuda() )
        self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]).cuda() )
        
        if disableViewDependentColor:
            dim0 = (3+3*posbase_pe*2)
        else:
            dim0 = (3+3*posbase_pe*2) + (3+3*viewbase_pe*2)

        if not AddFeatureVector:
            additionRGBNetwrokDim =0
        dim0 += self.k0_dim + additionRGBNetwrokDim
        #dim0 = 8 + additionRGBNetwrokDim
        # dim0 = 264
        self.rgbnet = nn.Sequential(
            nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth-2)
            ],
            nn.Linear(rgbnet_width, 3),
        )
        nn.init.constant_(self.rgbnet[-1].bias, 0)
        print('dvgo: feature voxel grid', self.k0.shape)
        print('dvgo: mlp', self.rgbnet)


        print("--------------- Finish ----------------")
        
    def get_grid_worldcoords3(self,):
        grid_coord = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)

        return grid_coord
    
    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        # 这个buffer会被保存，但是不会被更新梯度
        
        self.density[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    # accoding viewdirs and ray_pts get the density
    # question where is deform ?
    # sample points are points on rays and viedirs is only used for color
    def forward(self, rays_pts, viewdirs = None,featurevecEmbedding = None,k0bypass = None,**render_kwargs):
        '''
            give alpha and rgb according to given positions
        '''
        # mask_cache false means not used
        # update mask for query points in known free space


        k0 = torch.zeros([rays_pts.shape[0], self.k0_dim]).to(rays_pts)

        if k0bypass!=None:
            k0 = self.grid_sampler(rays_pts,k0bypass)
        else:
            k0 = self.grid_sampler(rays_pts, self.k0)
        if len(k0.shape) == 1:
            k0 = k0.unsqueeze(0)

        # view-dependent color emission

        k0_view = k0

        # viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        # viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

        # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
        rays_xyz = rays_pts
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)


        if AddFeatureVector:
            rgb_feat = torch.cat( [
                k0_view,
                
                featurevecEmbedding,
                # TODO: use `rearrange' to make it readable

            ], -1).to(rays_pts)   
        else:
        # TODO make sure 
            rgb_feat = torch.cat( [
                k0_view,
                xyz_emb,
                # TODO: use `rearrange' to make it readable
                # viewdirs
            ], -1).to(rays_pts)


        rgb_logit = torch.zeros([rays_pts.shape[0], 3]).to(rays_pts)
        rgb_logit = self.rgbnet(rgb_feat)

        rgb = torch.sigmoid(rgb_logit) #* 2.0


        return rgb
    
    def forward_grid(self, rays_pts, viewdirs = None,featurevecEmbedding = None,k0bypass = None,**render_kwargs):
        '''
            give alpha and rgb according to given positions
        '''
        # mask_cache false means not used
        # update mask for query points in known free space


        k0 = torch.zeros([rays_pts.shape[0], self.k0_dim]).to(rays_pts)

        if k0bypass!=None:
            k0 = self.grid_sampler(rays_pts,k0bypass)
        else:
            k0 = self.grid_sampler(rays_pts, self.k0)
        if len(k0.shape) == 1:
            k0 = k0.unsqueeze(0)

        # view-dependent color emission

        k0_view = k0
        return k0_view
        # viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        # viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

        # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
        rays_xyz = rays_pts
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)


        if AddFeatureVector:
            rgb_feat = torch.cat( [
                k0_view,
                xyz_emb,
                featurevecEmbedding,
                # TODO: use `rearrange' to make it readable
                viewdirs
            ], -1).to(rays_pts)   
        else:
        # TODO make sure 
            rgb_feat = torch.cat( [
                k0_view,
                # xyz_emb,
                # TODO: use `rearrange' to make it readable
                # viewdirs
            ], -1).to(rays_pts)


        rgb_logit = torch.zeros([rays_pts.shape[0], 3]).to(rays_pts)
        rgb_logit = self.rgbnet(rgb_feat)

        rgb = torch.sigmoid(rgb_logit)


        return rgb

class StylizedRadianceNet_Grid(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 rgbnet_depth=2, rgbnet_width=32,
                 posbase_pe=3, viewbase_pe=4, additionRGBNetwrokDim = 0,
                 **kwargs):
        super(StylizedRadianceNet_Grid, self).__init__()
        print("----------Init Radiance Grid-----------")
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        print("dvgo: xyz_min, xyz_max: ", self.xyz_min, ", ", self.xyz_max)



        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)


        # determine init grid resolution
        self._set_grid_resolution(num_voxels)
        

        # feature voxel grid + shallow MLP  (fine stage)

        self.k0_dim = 3
        noise = torch.rand([1, 1, *self.world_size]).repeat(1,self.k0_dim,1,1,1)
        self.k0 = torch.nn.Parameter(noise.cuda() )

        self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]).cuda() )
        self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]).cuda() )
        
        print('dvgo: feature voxel grid', self.k0.shape)
        self.rgbnet = None

        print("--------------- Finish ----------------")
        
    def get_grid_worldcoords3(self,):
        grid_coord = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)

        return grid_coord
    
    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            **self.rgbnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        # 这个buffer会被保存，但是不会被更新梯度
        
        self.density[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.density.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.density.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.density.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.density = torch.nn.Parameter(
            F.interpolate(self.density.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def k0_total_variation(self):
        if self.rgbnet is not None:
            v = self.k0
        else:
            v = torch.sigmoid(self.k0)
        return total_variation(v, self.nonempty_mask)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    # accoding viewdirs and ray_pts get the density
    # question where is deform ?
    # sample points are points on rays and viedirs is only used for color
    def forward(self, rays_pts, viewdirs = None,featurevecEmbedding = None,k0bypass = None,**render_kwargs):
        '''
            give alpha and rgb according to given positions
        '''
        # mask_cache false means not used
        # update mask for query points in known free space


        k0 = torch.zeros([rays_pts.shape[0], self.k0_dim]).to(rays_pts)

        if k0bypass!=None:
            k0 = self.grid_sampler(rays_pts,k0bypass)
        else:
            k0 = self.grid_sampler(rays_pts, self.k0)
        if len(k0.shape) == 1:
            k0 = k0.unsqueeze(0)

        # view-dependent color emission

        k0_view = k0

    

        rgb = torch.relu(k0_view) # * 5.0


        return rgb
    
    def forward_grid(self, rays_pts, viewdirs = None,featurevecEmbedding = None,k0bypass = None,**render_kwargs):
        '''
            give alpha and rgb according to given positions
        '''
        # mask_cache false means not used
        # update mask for query points in known free space


        k0 = torch.zeros([rays_pts.shape[0], self.k0_dim]).to(rays_pts)

        if k0bypass!=None:
            k0 = self.grid_sampler(rays_pts,k0bypass)
        else:
            k0 = self.grid_sampler(rays_pts, self.k0)
        if len(k0.shape) == 1:
            k0 = k0.unsqueeze(0)

        # view-dependent color emission

        k0_view = k0
        return k0_view
        # viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        # viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)

        # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
        rays_xyz = rays_pts
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)


        if AddFeatureVector:
            rgb_feat = torch.cat( [
                k0_view,
                xyz_emb,
                featurevecEmbedding,
                # TODO: use `rearrange' to make it readable
                viewdirs
            ], -1).to(rays_pts)   
        else:
        # TODO make sure 
            rgb_feat = torch.cat( [
                k0_view,
                # xyz_emb,
                # TODO: use `rearrange' to make it readable
                # viewdirs
            ], -1).to(rays_pts)


        rgb_logit = torch.zeros([rays_pts.shape[0], 3]).to(rays_pts)
        rgb_logit = self.rgbnet(rgb_feat)

        rgb = torch.sigmoid(rgb_logit)


        return rgb


class DeformVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 deform_num_voxels=1664000, deform_num_voxels_base=1664000,
                 #NEW
                 additionTimeEncoding = 0,
                 nearest=False, pre_act_density=False, in_act_density=False,
                 mask_cache_path=None, mask_cache_thres=1e-3,
                 fast_deform_thres=0,
                 deformnet_dim=4, deformnet_full_implicit=False,
                 deformnet_depth=3, deformnet_width=128, deformnet_output=3,
                 posbase_pe=5, timebase_pe=5,
                 train_times=None,
                 **kwargs):
        super(DeformVoxGO, self).__init__()
        print("---------Init Deformation Grid---------")
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        print("dvgo: xyz_min, xyz_max: ", self.xyz_min, ", ", self.xyz_max)
        self.fast_deform_thres = fast_deform_thres
        self.nearest = nearest
        self.pre_act_density = pre_act_density
        self.in_act_density = in_act_density
        if self.pre_act_density:
            print('dvgo: using pre_act_density may results in worse quality !!')
        if self.in_act_density:
            print('dvgo: using in_act_density may results in worse quality !!')

        # determine based grid resolution
        self.num_voxels_base = deform_num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # # determine the density bias shift
        # self.alpha_init = alpha_init
        # self.act_shift = np.log(1/(1-alpha_init) - 1)
        # print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(deform_num_voxels)

        # init occlusion voxel grid
        self.occlusion = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size]))
        if OverWritedisableOcclusionMask:
            deformnet_output = 3
        # init color representation
        self.deformnet_kwargs = {
            'deformnet_dim': deformnet_dim,
            'deformnet_full_implicit': deformnet_full_implicit,
            'deformnet_depth': deformnet_depth, 'deformnet_width': deformnet_width,
            'posbase_pe': posbase_pe, 'timebase_pe': timebase_pe, 'deformnet_output': deformnet_output,
        }
        self.deformnet_full_implicit = deformnet_full_implicit

        self.deformnet_output = deformnet_output

        # feature voxel grid + shallow MLP  (fine stage)
        if self.deformnet_full_implicit:
            self.k0_dim = 0
        else:
            self.k0_dim = deformnet_dim
        self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        # torch.nn.init.xavier_normal_(self.k0)
        # self.k0 = torch.nn.Parameter(self.get_grid_worldcoords3().permute(3, 0, 1, 2).unsqueeze(0))
        self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('timefreq', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.additionTimeEncoding = additionTimeEncoding
        if additionTimeEncoding != 0:
            dim0 = (3+3*posbase_pe*2) + additionTimeEncoding
        else:
            dim0 = (3+3*posbase_pe*2) + (1+timebase_pe*2)
        
        if self.deformnet_full_implicit:
            pass
        else:
            dim0 += self.k0_dim * (1+timebase_pe*2)
            
        self.deformnet = nn.Sequential(
            nn.Linear(dim0, deformnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(deformnet_width, deformnet_width), nn.ReLU(inplace=True))
                for _ in range(deformnet_depth-2)
            ],
            nn.Linear(deformnet_width, self.deformnet_output),
        )
        nn.init.constant_(self.deformnet[-1].bias, 0)
        self.deformnet[-1].weight.data *= 0.0
        print('dvgo: feature voxel grid', self.k0.shape)
        print('dvgo: mlp', self.deformnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)

        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        self.mask_cache = None
        self.nonempty_mask = None
        self.train_times = train_times

        if mask_cache_path is not None:
            # mask cache
            print('mask cache path: ', mask_cache_path)
            self.mask_cache = MaskCacheDeform(
                    path=mask_cache_path,
                    train_times=train_times,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self._set_nonempty_mask()

            # reload grid and network
            cache_model = torch.load(mask_cache_path)
            cache_model_occlusion = cache_model['model_state_dict']['deformgrid.occlusion']
            cache_model_k0 = cache_model['model_state_dict']['deformgrid.k0']
            cache_xyz_min = torch.FloatTensor(cache_model['MaskCache_kwargs']['xyz_min']).to(cache_model_k0.device)
            cache_xyz_max = torch.FloatTensor(cache_model['MaskCache_kwargs']['xyz_max']).to(cache_model_k0.device)

            grid_xyz = self.get_grid_worldcoords3().unsqueeze(0)

            ind_norm = ((grid_xyz - cache_xyz_min) / (cache_xyz_max - cache_xyz_min)).flip((-1,)) * 2 - 1

            self.occlusion = torch.nn.Parameter(
                F.grid_sample(cache_model_occlusion, ind_norm, align_corners=True))

            if self.k0_dim > 0:
                self.k0 = torch.nn.Parameter(
                    F.grid_sample(cache_model_k0, ind_norm, align_corners=True))

            # load deformnet weights
            dn_static_dict = self.deformnet.state_dict()
            for k, v in dn_static_dict.items():
                if 'deformgrid.deformnet.' + k in cache_model['model_state_dict'].keys():
                    v = cache_model['model_state_dict']['deformgrid.deformnet.' + k]
                    dn_static_dict.update({k: v})
            self.deformnet.load_state_dict(dn_static_dict)
        print("--------------- Finish ----------------")

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long() # float 2 int
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'deform_num_voxels': self.num_voxels,
            'deform_num_voxels_base': self.num_voxels_base,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'fast_deform_thres': self.fast_deform_thres,
            'train_times': self.train_times,
            **self.deformnet_kwargs,
        }

    def get_MaskCache_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'voxel_size_ratio': self.voxel_size_ratio,
            'nearest': self.nearest,
            'pre_act_density': self.pre_act_density,
            'in_act_density': self.in_act_density,
        }

    @torch.no_grad()
    def _set_nonempty_mask(self):
        # Find grid points that is inside nonempty (occupied) space
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.occlusion.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.occlusion.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.occlusion.shape[4]),
        ), -1)
        nonempty_mask = self.mask_cache(self_grid_xyz)[None,None].contiguous()
        if hasattr(self, 'nonempty_mask'):
            self.nonempty_mask = nonempty_mask
        else:
            self.register_buffer('nonempty_mask', nonempty_mask)
        self.occlusion[~self.nonempty_mask] = -100

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near):
        self_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.occlusion.shape[2]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.occlusion.shape[3]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.occlusion.shape[4]),
        ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.occlusion[nearest_dist[None,None] <= near] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.occlusion = torch.nn.Parameter(
            F.interpolate(self.occlusion.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        eps_time = time.time()
        N_samples = int(np.linalg.norm(np.array(self.occlusion.shape[2:])+1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.occlusion.detach())
        device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = torch.ones_like(self.occlusion).requires_grad_()
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                self.grid_sampler(rays_pts, ones).sum().backward()
            with torch.no_grad():
                count += (ones.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def k0_total_variation(self):
        v = self.k0
        return total_variation(v)
        return total_variation(v, self.nonempty_mask)

    def occlusion_mean(self):
        return torch.mean(torch.sigmoid(self.occlusion))

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]        

        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def grid_sampler_elastic(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        
        if mode is None:
            # bilinear is actually trilinear if 5D input is given to grid_sample
            mode = 'nearest' if self.nearest else 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if Deform_Grid_type == 'ori':
            ret_lst = [
                # TODO: use `rearrange' to make it readable
                F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
                for grid in grids
            ]        
        elif Deform_Grid_type == 'cuda':
            ret_lst = [
                # TODO: use `rearrange' to make it readable
                cu.grid_sample_3d(grid, ind_norm,  align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
                for grid in grids
            ]
        elif Deform_Grid_type == 'torch':
            ret_lst = [
                # TODO: use `rearrange' to make it readable
                grid_sample_3d_customize(grid, ind_norm).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
                for grid in grids
            ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.k0.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox

    def get_grid_worldcoords(self,):
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.range(0, self.world_size[0]-1),
            torch.range(0, self.world_size[1]-1),
            torch.range(0, self.world_size[2]-1),
            indexing='ij'
        )
        grid_coord = torch.stack([grid_z, grid_y, grid_x], dim=-1) # grid_sample use pixel positions, inverse
        grid_coord = 2 * grid_coord / (self.world_size.flip((-1,)) - 1) - 1 # [-1 1]

        grid_coord = (grid_coord + 1) / 2
        grid_coord = grid_coord.flip((-1,))
        grid_coord = grid_coord * (self.xyz_max - self.xyz_min) + self.xyz_min

        return grid_coord

    def get_grid_worldcoords2(self,):
        interp = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, self.world_size[0]),
            torch.linspace(0, 1, self.world_size[1]),
            torch.linspace(0, 1, self.world_size[2]),
        ), -1)
        grid_coord = self.xyz_min * (1-interp) + self.xyz_max * interp

        return grid_coord

    # it is 3 in use
    # grid sample point
    def get_grid_worldcoords3(self,):
        grid_coord = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)

        return grid_coord

    def getdeformOnly(self, rays_pts, timestep):
        # update mask for query points in known free space
    
        rays_pts = rays_pts.unsqueeze(0)
        
        # query for occlusion mask
        mask_outbbox = torch.ones_like(rays_pts)[...] > 0.0
        occlusion_mask = torch.zeros_like(rays_pts[...,0])
        
        # query for deform
        # mask = (occlusion_mask > self.fast_deform_thres)
        mask = ~mask_outbbox
        k0 = torch.zeros(*occlusion_mask.shape, self.k0_dim).to(occlusion_mask)

        k0 = self.grid_sampler_elastic(rays_pts, self.k0)
        # print(rays_pts.device)
        # print(k0.device)
        # return k0
        k0_view = k0
        k0_view = k0_view.unsqueeze(0)
        timestep_emb = (timestep.unsqueeze(-1) * self.timefreq).flatten(-2)
        timestep_emb = torch.cat([timestep, timestep_emb.sin(), timestep_emb.cos()], -1)

        # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
        rays_xyz = rays_pts
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        k0_view_mask = k0_view
        
        k0_emb = (k0_view_mask.unsqueeze(-1) * self.timefreq).flatten(-2)
        k0_emb = torch.cat([k0_view_mask, k0_emb.sin(), k0_emb.cos()], -1)

        # print('end')
        # print(k0_view_mask.shape)
        # print(k0_view.shape)
        # print(k0_emb.shape)
        # print(xyz_emb.shape)
        # print(timestep_emb.shape)
        timestep_emb = timestep_emb.unsqueeze(0)

        deform_feat = torch.cat([
            k0_emb,
            xyz_emb,
            # TODO: use `rearrange' to make it readable
            # Todo not correct for one
            timestep_emb.flatten(0,-2).repeat(occlusion_mask.shape[-1],1) # [mask.flatten(0,-2)]
        ], -1)

        deform_logit = torch.zeros(*occlusion_mask.shape, self.deformnet_output).to(occlusion_mask)
        if self.deformnet_output > 3:
            deform_logit[..., -1] = 100000

        deform_logit = self.deformnet(deform_feat)
        if self.deformnet_output > 3:
            deform = deform_logit[..., :-1]
        else:
            deform = deform_logit


        return  deform
    
    def getElasticLoss(self,jacobian):
        elastics_loss,residual = compute_elastic_loss_fancy(jacobian)
        return elastics_loss

    def getElastcJac(self,rays_pts, timestep, mask_outbbox,usefulMask, EnableMask=True,DeltaTimeT = 1.0/40.0, **render_kwargs):
        # deform = self.getdeformOnly(rays_pts, timestep, mask_outbbox,EnableMask=True, **render_kwargs)
        # rays_pts_ori_shape = rays_pts.shape
        # rays_pts = rays_pts.reshape([-1,3])
        from torch.func import jacfwd, vmap,jacrev
        # print('elas')
        # print(usefulMask.shape)
        # print(timestep.shape)
        # useful_points = rays_pts[usefulMask]
        # dummyMask = torch.ones_like(useful_points)[...,0] > 0.0

        with torch.no_grad():
            if len(timestep.shape) == 2:
                new_time_step = timestep # .unsqueeze(-2).repeat(1,rays_pts.shape[-2],1)
                newMask = new_time_step > 0.0
                # print(usefulMask.shape)
                newMask = newMask.squeeze(-1)
                 # print(usefulMask.shape)
                usefulMask = newMask & usefulMask

                new_time_step = new_time_step[usefulMask]
                useful_points = rays_pts[usefulMask]
            else:
                new_time_step = timestep.unsqueeze(-2).repeat(1,1,rays_pts.shape[-2],1)
                newMask = new_time_step > 0.0
                newMask = newMask.squeeze(-1)
                usefulMask = newMask & usefulMask
                new_time_step = new_time_step[usefulMask]
                useful_points = rays_pts[usefulMask]
                
            def elafunc2(rays_pts_in,input_time):
                deform = self.getdeformOnly(rays_pts_in, input_time)
                deform_prev = self.getdeformOnly(rays_pts_in, input_time - DeltaTimeT)
                return  deform_prev - deform 
            def elafunc(rays_pts_in,input_time):
                deform = self.getdeformOnly(rays_pts_in, input_time)
                return deform 
                #return torch.sum(deform,dim = 0)
            if EnableAnotherElaMode:
                jacobian = vmap(jacrev(elafunc2,argnums=0))(useful_points,new_time_step)
            else:
                jacobian = vmap(jacrev(elafunc,argnums=0))(useful_points,new_time_step)
            # jacobian = jacobian[:300]
            # print(jacobian.shape)
    
            jacobian = jacobian.squeeze(1)

            return jacobian
         
    # what is this occlusion_mask?
    # return occlusion_mask, deform, mask, time_dict
    def forward(self, rays_pts, timestep, mask_outbbox,EnableMask=True, **render_kwargs):
        '''
            give occlusion mask and deformation according to given positions
        '''
        
        # update mask for query points in known free space
        if (self.mask_cache is not None) and EnableMask:
            mask_outbbox[~mask_outbbox] |= (~self.mask_cache(rays_pts[~mask_outbbox]))

        # ------------------- time a --------------------
        torch.cuda.synchronize()
        time_a = time.time()
        # -----------------------------------------------

        # query for occlusion mask
        occlusion_mask = torch.zeros_like(rays_pts[...,0]).to(rays_pts)

        if self.pre_act_density:
            # pre-activation
            occlusion_mask[~mask_outbbox] = self.grid_sampler(
                    rays_pts[~mask_outbbox], torch.sigmoid(self.occlusion))
        elif self.in_act_density:
            # in-activation : same with pre-activation in terms of occlusion mask
            occlusion_mask[~mask_outbbox] = self.grid_sampler(
                    rays_pts[~mask_outbbox], torch.sigmoid(self.occlusion))
        else:
            # post-activation
            occlusion_feat = self.grid_sampler(rays_pts[~mask_outbbox], self.occlusion).to(rays_pts)
            occlusion_mask[~mask_outbbox] = torch.sigmoid(occlusion_feat)


        # ------------------- time b --------------------
        torch.cuda.synchronize()
        time_b = time.time()
        # -----------------------------------------------

        # query for deform
        # mask = (occlusion_mask > self.fast_deform_thres)
        mask = ~mask_outbbox
        k0 = torch.zeros(*occlusion_mask.shape, self.k0_dim).to(occlusion_mask)
        if not self.deformnet_full_implicit:
            gridRes = self.grid_sampler(rays_pts[mask], self.k0).to(rays_pts)
            k0[mask] = gridRes


        k0_view = k0

        # ------------------- time c --------------------
        torch.cuda.synchronize()
        time_c = time.time()
        # -----------------------------------------------

        if self.additionTimeEncoding != 0:
            timestep_emb = timestep
        else:
            timestep_emb = (timestep.unsqueeze(-1) * self.timefreq).flatten(-2)
            timestep_emb = torch.cat([timestep, timestep_emb.sin(), timestep_emb.cos()], -1)

        # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
        rays_xyz = rays_pts[mask]
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        k0_view_mask = k0_view[mask]
        k0_emb = (k0_view_mask.unsqueeze(-1) * self.timefreq).flatten(-2)
        k0_emb = torch.cat([k0_view_mask, k0_emb.sin(), k0_emb.cos()], -1)

        deform_feat = torch.cat([
            k0_emb,
            xyz_emb,
            # TODO: use `rearrange' to make it readable
            timestep_emb
        ], -1)

        deform_logit = torch.zeros(*occlusion_mask.shape, self.deformnet_output).to(occlusion_mask)

        if self.deformnet_output > 3:
            deform_logit[..., -1] = 100.0


        networkOutPut = self.deformnet(deform_feat).to(occlusion_mask)
        deform_logit[mask] = networkOutPut
        if self.deformnet_output > 3:
            deform = deform_logit[..., :-1].to(occlusion_mask)
            occlusion_mask = deform_logit[..., -1, None].to(occlusion_mask)
            occlusion_mask = torch.sigmoid(occlusion_mask)
        else:
            deform = deform_logit

        # ------------------- time d --------------------
        torch.cuda.synchronize()
        time_d = time.time()

        time_dict = {
            'query_occ': time_b - time_a,
            'query_k0': time_c - time_b,
            'query_dnet': time_d - time_c
        }
        # -----------------------------------------------

        return  occlusion_mask, deform, mask, time_dict

    def forward_simple(self, rays_pts, timestep, **render_kwargs):
        '''
            give occlusion mask and deformation according to given positions
        '''
    


        # query for occlusion mask
       # occlusion_mask = torch.zeros_like(rays_pts[...,0]).to(rays_pts)


        # query for deform
        # mask = (occlusion_mask > self.fast_deform_thres)
        k0 = torch.zeros(*rays_pts[...,0].shape, self.k0_dim).to(rays_pts)
        if not self.deformnet_full_implicit:
            gridRes = self.grid_sampler(rays_pts, self.k0).to(rays_pts)
            k0 = gridRes


        k0_view = k0
        # return k0


        if self.additionTimeEncoding != 0:
            timestep_emb = timestep
        else:
            timestep_emb = (timestep.unsqueeze(-1) * self.timefreq).flatten(-2)
            timestep_emb = torch.cat([timestep, timestep_emb.sin(), timestep_emb.cos()], -1)

        # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
        rays_xyz = rays_pts
        xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
        xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

        k0_view_mask = k0_view
        k0_emb = (k0_view_mask.unsqueeze(-1) * self.timefreq).flatten(-2)
        k0_emb = torch.cat([k0_view_mask, k0_emb.sin(), k0_emb.cos()], -1)
        if len(k0_emb.shape) == 1:
            k0_emb = k0_emb.unsqueeze(0)
        # print(k0_emb.shape)
        # print(xyz_emb.shape)
        # print(timestep_emb.shape)
        deform_feat = torch.cat([
            k0_emb,
            xyz_emb,
            # TODO: use `rearrange' to make it readable
            timestep_emb
        ], -1)

        # deform_logit = torch.zeros(*occlusion_mask.shape, self.deformnet_output).to(rays_pts)

        # if self.deformnet_output > 3:
        #     deform_logit[..., -1] = 100.0


        networkOutPut = self.deformnet(deform_feat).to(rays_pts)
        deform_logit = networkOutPut
        # if self.deformnet_output > 3:
        #     deform = deform_logit[..., :-1].to(rays_pts)
        #     occlusion_mask = deform_logit[..., -1, None].to(rays_pts)
        #     occlusion_mask = torch.sigmoid(rays_pts)
        # else:
        deform = deform_logit


        return  deform


class StyliziedDeformVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 deform_num_voxels=0, deform_num_voxels_base=0,
                 nearest=False, pre_act_density=False, in_act_density=False,
                  mask_cache_thres=1e-3,
                 fast_deform_thres=0,
                 deformnet_dim=0,
                 deformnet_depth=3, deformnet_width=128, deformnet_output=3,
                 posbase_pe=5, timebase_pe=5,use_time=True,
                 **kwargs):
        # deformnet_output = 4 overhere
        
        super(StyliziedDeformVoxGO, self).__init__()

        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

        # determine based grid resolution
        self.num_voxels_base = deform_num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
        self.use_time = use_time
        # # determine the density bias shift
        # self.alpha_init = alpha_init
        # self.act_shift = np.log(1/(1-alpha_init) - 1)
        # print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        # determin self.world_size here
        self._set_grid_resolution(deform_num_voxels)


        # init color representation
        self.deformnet_kwargs = {
            'deformnet_dim': deformnet_dim,
            'deformnet_depth': deformnet_depth, 'deformnet_width': deformnet_width,
            'posbase_pe': posbase_pe, 'timebase_pe': timebase_pe, 'deformnet_output': deformnet_output,
        }
        self.deformnet_output = deformnet_output

        self.k0_dim = deformnet_dim
        
        # decode from tensor
        self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        
        
        # torch.nn.init.xavier_normal_(self.k0)
        # self.k0 = torch.nn.Parameter(self.get_grid_worldcoords3().permute(3, 0, 1, 2).unsqueeze(0))
        self.register_buffer('posfreq', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('timefreq', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        
        if self.use_time:
            dim0 = (3+3*posbase_pe*2) + (1+timebase_pe*2)
            dim0 += self.k0_dim * (1+timebase_pe*2)
        else:
            dim0 = self.k0_dim 
            
        self.deformnet = nn.Sequential(
            nn.Linear(dim0, deformnet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(deformnet_width, deformnet_width), nn.ReLU(inplace=True))
                for _ in range(deformnet_depth-2)
            ],
            nn.Linear(deformnet_width, self.deformnet_output),
        )
        
        # nn.init.constant_(self.deformnet[-1].bias, 0)
        # self.deformnet[-1].weight.data *= 0.0

        # Using the coarse geometry if provided (used to determine known free space and unknown space)

        self.mask_cache = None
        self.nonempty_mask = None

    def k0_total_variation(self):
        v = self.k0
        return total_variation(v, self.nonempty_mask)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        # print('dvgo: voxel_size      ', self.voxel_size)
        # print('dvgo: world_size      ', self.world_size)
        # print('dvgo: voxel_size_base ', self.voxel_size_base)
        # print('dvgo: voxel_size_ratio', self.voxel_size_ratio)
        
    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.occlusion = torch.nn.Parameter(
            F.interpolate(self.occlusion.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def k0_total_variation(self):
        v = self.k0
        return total_variation(v, self.nonempty_mask)

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''

        mode = 'bilinear'
        
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)

        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays'''
        # 1. determine the maximum number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.k0.shape[2:])+1) / stepsize) + 1
        # 2. determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox

    # def getRoughMask(self,raybots):
        
    
    # this is only used for mask cache
    def get_grid_worldcoords(self,):
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.range(0, self.world_size[0]-1),
            torch.range(0, self.world_size[1]-1),
            torch.range(0, self.world_size[2]-1),
            indexing='ij'
        )
        grid_coord = torch.stack([grid_z, grid_y, grid_x], dim=-1) # grid_sample use pixel positions, inverse
        grid_coord = 2 * grid_coord / (self.world_size.flip((-1,)) - 1) - 1 # [-1 1]

        grid_coord = (grid_coord + 1) / 2
        grid_coord = grid_coord.flip((-1,))
        grid_coord = grid_coord * (self.xyz_max - self.xyz_min) + self.xyz_min

        return grid_coord

    # it is 3 in use
    # grid sample point
    def get_grid_worldcoords3(self,):
        grid_coord = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)

        return grid_coord
    
    # what is this occlusion_mask?
    # return occlusion_mask, deform, mask, time_dict
    def forward(self, rays_pts,timestep, mask_outbbox, **render_kwargs):
        '''
            give occlusion mask and deformation according to given positions
        '''

        # query for occlusion mask
        occlusion_mask = torch.zeros_like(rays_pts[...,0])

        mask = ~mask_outbbox
        
        k0_view = torch.zeros(*occlusion_mask.shape, self.k0_dim).to(occlusion_mask)
        
        k0_view[mask] = self.grid_sampler(rays_pts[mask], self.k0)


        if self.use_time:
            timestep_emb = (timestep.unsqueeze(-1) * self.timefreq).flatten(-2)
            timestep_emb = torch.cat([timestep, timestep_emb.sin(), timestep_emb.cos()], -1)

            # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
            rays_xyz = rays_pts[mask]
            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

            k0_view_mask = k0_view[mask]
            k0_emb = (k0_view_mask.unsqueeze(-1) * self.timefreq).flatten(-2)
            k0_emb = torch.cat([k0_view_mask, k0_emb.sin(), k0_emb.cos()], -1)

            deform_feat = torch.cat([
                k0_emb,
                xyz_emb,
                # TODO: use `rearrange' to make it readable
                timestep_emb.flatten(0,-2).unsqueeze(-2).repeat(1,occlusion_mask.shape[-1],1)[mask.flatten(0,-2)]
            ], -1)
        else:
            # rays_xyz = (rays_pts[mask] - self.xyz_min) / (self.xyz_max - self.xyz_min)
            rays_xyz = rays_pts[mask]

            xyz_emb = (rays_xyz.unsqueeze(-1) * self.posfreq).flatten(-2)
            xyz_emb = torch.cat([rays_xyz, xyz_emb.sin(), xyz_emb.cos()], -1)

            k0_view_mask = k0_view[mask]
            # print(k0_view_mask.shape)
            # k0_emb = (k0_view_mask.unsqueeze(-1) * self.timefreq).flatten(-2)
            # print(k0_emb.shape)
            # k0_emb = torch.cat([k0_view_mask, k0_emb.sin(), k0_emb.cos()], -1)
            k0_emb = k0_view_mask
            deform_feat = torch.cat([
                k0_emb,
                #xyz_emb,
            ], -1)

        deform_logit = torch.zeros(*occlusion_mask.shape, self.deformnet_output).to(occlusion_mask)
      
        # WHY NOT 3?
        # throw last?
        deform_logit[mask] = self.deformnet(deform_feat)
        # deform_logit[mask] = k0_view[mask]
        # 丢掉最后一个
        if self.deformnet_output > 3:
            deform = deform_logit[..., :-1]
        else:
            deform = deform_logit

        return  deform, mask



class StyliziedPureVoxelGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 deform_num_voxels=0, deform_num_voxels_base=0,
                H= 400,W = 400,focal = 1.0,c2w = None,far = 10.0,near = 1.0
 
                ,use_time=True,ndc_sampleMode= False,
                 **kwargs):
        # deformnet_output = 4 overhere
        
        super(StyliziedPureVoxelGO, self).__init__()


        self.H = H
        self.W = W
        self.focal = focal
        self.w2c = None
        if c2w!=None:
            self.w2c= torch.inverse(c2w)
        self.far = far
        self.near = near

        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))

        # determine based grid resolution
        self.num_voxels_base = deform_num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
        self.use_time = use_time
        # # determine the density bias shift
        # self.alpha_init = alpha_init
        # self.act_shift = np.log(1/(1-alpha_init) - 1)
        # print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        # determin self.world_size here
        self._set_grid_resolution(deform_num_voxels)

        self.k0_dim = 3
        # decode from tensor
        self.k0 = torch.nn.Parameter(0.1 * torch.zeros([1, self.k0_dim, *self.world_size]))
        
        
        # nn.init.constant_(self.deformnet[-1].bias, 0)
        # self.deformnet[-1].weight.data *= 0.0

        # Using the coarse geometry if provided (used to determine known free space and unknown space)

        self.mask_cache = None
        self.nonempty_mask = None
        
        self.ndc_sampleMode = ndc_sampleMode

    def k0_total_variation(self):
        v = self.k0
        return total_variation(v, self.nonempty_mask)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        # print('dvgo: voxel_size      ', self.voxel_size)
        # print('dvgo: world_size      ', self.world_size)
        # print('dvgo: voxel_size_base ', self.voxel_size_base)
        # print('dvgo: voxel_size_ratio', self.voxel_size_ratio)
        
    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

        self.occlusion = torch.nn.Parameter(
            F.interpolate(self.occlusion.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.k0_dim > 0:
            self.k0 = torch.nn.Parameter(
                F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        else:
            self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
        if self.mask_cache is not None:
            self._set_nonempty_mask()
        print('dvgo: scale_volume_grid finish')

    def k0_total_variation(self):
        v = self.k0
        return total_variation(v, self.nonempty_mask)

        
        
    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        
        
        ndc_sample = self.ndc_sampleMode
        
        '''Wrapper for the interp operation'''

        mode = 'bilinear'
        
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        # print(self.xyz_min)
        # print(self.xyz_max)
        if ndc_sample:
            tempxyz = xyz.squeeze(0)
            tempxyz = tempxyz.squeeze(0)
            tempxyz = tempxyz.squeeze(0)
            
            ind_norm  = projection(tempxyz,
                   self.w2c,
                   self.near,
                   self.far,
                   H=self.H,
                   W=self.W,
                   Fx=self.focal,
                   Fy=self.focal
                   )
            ind_norm = ind_norm.unsqueeze(0)
            ind_norm = ind_norm.unsqueeze(0)
            ind_norm = ind_norm.unsqueeze(0)

        else:
            ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        # ind_norm2 = abs(ind_norm)
        # print(torch.max(ind_norm))
        # print(torch.min(ind_norm))
        # print(torch.sum(ind_norm))
        # print((ind_norm.shape))
        # print(torch.max(ind_norm2))
        # print(torch.min(ind_norm2))
        # print(torch.sum(ind_norm2))
        # print((ind_norm2.shape))
        ret_lst = [
            # TODO: use `rearrange' to make it readable
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners,
                          ).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
            for grid in grids
        ]
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst
    
    # it is 3 in use
    # grid sample point
    def get_grid_worldcoords3(self,):
        grid_coord = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)

        return grid_coord
    
    # what is this occlusion_mask?
    # return occlusion_mask, deform, mask, time_dict
    def forward(self, rays_pts,timestep, mask_outbbox,k0overwrite = None, **render_kwargs):
        '''
            give occlusion mask and deformation according to given positions
        '''

        # query for occlusion mask
        occlusion_mask = torch.zeros_like(rays_pts[...,0])

        mask = ~mask_outbbox
        
        k0_view = torch.zeros(*occlusion_mask.shape, self.k0_dim).to(occlusion_mask)
        
        if k0overwrite == None:
            k0_view[mask] = self.grid_sampler(rays_pts[mask], self.k0).to(rays_pts)
        else:
            k0_view[mask] = self.grid_sampler(rays_pts[mask],k0overwrite).to(rays_pts)
        
        return  k0_view, mask
    
    def forward_simple(self, rays_pts,k0overwrite = None, **render_kwargs):
        '''
            give occlusion mask and deformation according to given positions
        '''

        # query for occlusion mask
        occlusion_mask = torch.zeros_like(rays_pts[...,0])

        
        k0_view = torch.zeros(*occlusion_mask.shape, self.k0_dim).to(occlusion_mask)
        if k0overwrite == None:
            k0_view = self.grid_sampler(rays_pts, self.k0).to(rays_pts)
        else:
            k0_view = self.grid_sampler(rays_pts,k0overwrite).to(rays_pts)
        
        return  k0_view


class VoxRendererDynamic(torch.nn.Module):
    def __init__(
        self,
        deformgrid,
        radiancegrid,
        **kwargs
    ):
        super(VoxRendererDynamic, self).__init__()
        self.deformgrid = deformgrid
        self.radiancegrid = radiancegrid

    def scale_volume_grid(self, factor):
        self.deformgrid.scale_volume_grid(self.deformgrid.num_voxels * factor)
        self.radiancegrid.scale_volume_grid(self.radiancegrid.num_voxels * factor)


    def get_deform_grid(self, time_step):
        grid_coord = self.deformgrid.get_grid_worldcoords3()
        grid_coord = grid_coord.reshape(1, -1, 3)
        num_grid = grid_coord.shape[1]

        timesstep = torch.ones(1, 1) * time_step
        mask_outbbox = torch.zeros(1, num_grid) > 0
        occ, deformation, _, _ = self.deformgrid(grid_coord, timesstep, mask_outbbox)

        can_mask = timesstep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deformation.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        deformation[can_mask] = 0.
        grid_coord += deformation

        return grid_coord, occ

    def get_deform_alpha_rgb(self, time_step):
        grid_coord, occ = self.get_deform_grid(time_step)
        densities = self.radiancegrid.grid_sampler(grid_coord.reshape(-1,3), self.radiancegrid.density)
        alpha = self.radiancegrid.activate_density(densities)
        alpha = alpha.reshape([1, 1, *self.deformgrid.world_size])
        occ = occ.reshape([1, 1, *self.deformgrid.world_size])

        k0 = self.radiancegrid.grid_sampler(grid_coord.reshape(-1,3), self.radiancegrid.k0)
        rgb = torch.sigmoid(k0).permute(1,0)
        rgb = rgb.reshape([1, 3, *self.deformgrid.world_size])
        return alpha, rgb, occ

    def forward(self, rays_o, rays_d, timestep, viewdirs, global_step=None, CalElastic_loss = True,**render_kwargs):
        '''Volume rendering'''
        ret_dict = {}

        # ------------------- time a --------------------
        torch.cuda.synchronize()
        time_a = time.time()
        # -----------------------------------------------

        # sample points on rays
        rays_pts, mask_outbbox = self.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.deformgrid.voxel_size_ratio

        # ------------------- time b --------------------
        torch.cuda.synchronize()
        time_b = time.time()
        # -----------------------------------------------

        # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)
        
        occlusion_mask, deform, mask_d, time_dict = self.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        # deform = deform * occlusion_mask.unsqueeze(-1)
        if 'bg_points_sel' in render_kwargs.keys():
            bg_mask_outbbox = torch.ones(render_kwargs['bg_points_sel'].shape[0], 1).to(rays_pts.device) > 0
            bg_time_step = timestep[:render_kwargs['bg_points_sel'].shape[0]]
            _, bg_points_deform, _, _ = self.deformgrid(render_kwargs['bg_points_sel'].unsqueeze(-2), bg_time_step, bg_mask_outbbox, **render_kwargs)
            ret_dict.update({'bg_points_delta': bg_points_deform})

        can_mask = timestep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        can_loss = torch.mean(torch.abs(deform[can_mask]))
        deform[can_mask] = 0.

        # # deform *= 0
        # rays_pts_ori = rays_pts
        Elastics_Loss =0.0
        # print(rays_pts.shape)
        # print(timestep.shape)
        # print(mask_d.shape)
        if CalElastic_loss:
            jac = self.deformgrid.getElastcJac(rays_pts, timestep, mask_outbbox,mask_d, **render_kwargs)
            Elastics_Loss = self.deformgrid.getElasticLoss(jac)
            
        
        rays_pts =rays_pts + deform

        # ------------------- time c --------------------
        torch.cuda.synchronize()
        time_c = time.time()
        # -----------------------------------------------

        # inference alpha, rgb
        # occlusion_mask 只是颜色而已
        if self.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None
        if OverWritedisableOcclusionMask:
            occ_input = None
        
        # alphainv_cum used for bg 
        alpha, alphainv_cum, rgb, weights, mask_r = self.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)

        # ------------------- time d --------------------
        torch.cuda.synchronize()
        time_d = time.time()
        # -----------------------------------------------

        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_loss': can_loss,
            'can_mask': can_mask,
            'elastics_loss':Elastics_Loss
        })

        # ------------------- time e --------------------
        torch.cuda.synchronize()
        time_e = time.time()

        time_dict.update({
            'sample_pts': time_b - time_a,
            'deform_forward': time_c - time_b,
            'rad_forward': time_d - time_c,
            'render': time_e - time_d
        })
        ret_dict.update({
            'time_dict': time_dict
        })
        # -----------------------------------------------

        return ret_dict


# opacity only?
# no gradient also needs
class StylizedDeformRender():
    def __init__(
        self,
       DynamicRender,
       StylizedDeformVol,
        **kwargs
    ):
        super(StylizedDeformRender, self).__init__()
        self.DynamicRender = DynamicRender
        self.StylizedDeformVol = StylizedDeformVol


    # @torch.enable_grad()
    # def compute_derived_normals(self, xyz_points):
    #     xyz_points.requires_grad_(True)
    #     # calculate sigma
    #     sigma_feature = self.compute_densityfeature_with_xyz_grad(xyz_locs)  # [..., 1]  detach() removed in the this function
    #     sigma = self.feature2density(sigma_feature)
    #     d_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)

    #     gradients = torch.autograd.grad(
    #                                 outputs=sigma,
    #                                 inputs=xyz_locs,
    #                                 grad_outputs=d_output,
    #                                 create_graph=True,
    #                                 retain_graph=True,
    #                                 only_inputs=True
    #                                 )[0]
    #     derived_normals = -safe_l2_normalize(gradients, dim=-1)
    #     derived_normals = derived_normals.view(-1, 3)
    #     return derived_normals




    def forward(self, rays_o, rays_d, timestep, viewdirs, global_step=None,Post_StyDeform = True, **render_kwargs):
        '''Volume rendering'''
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio


        # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)
        
        deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox, **render_kwargs)
        
        if not Post_StyDeform:
            rays_pts += deform_stylized
        
        occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox,EnableMask=True, **render_kwargs)
        
        # deform = deform * occlusion_mask.unsqueeze(-1)
        if 'bg_points_sel' in render_kwargs.keys():
            bg_mask_outbbox = torch.ones(render_kwargs['bg_points_sel'].shape[0], 1).to(rays_pts.device) > 0
            bg_time_step = timestep[:render_kwargs['bg_points_sel'].shape[0]]
            _, bg_points_deform, _, _ = self.DynamicRender.deformgrid(render_kwargs['bg_points_sel'].unsqueeze(-2), bg_time_step, bg_mask_outbbox, **render_kwargs)
            ret_dict.update({'bg_points_delta': bg_points_deform})

        can_mask = timestep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        can_loss = torch.mean(torch.abs(deform[can_mask]))
        # 把时间=0的部分的offset强制置于0
        deform[can_mask] = 0.

     
        mask_Valid = deform > 0.0
        mask_Valid = mask_d
        # rays_pts += deform_stylized *1.0
        
        rays_pts += deform 
        
        if Post_StyDeform:
            rays_pts[mask_Valid] += deform_stylized[mask_Valid] * 1.0
        # ------------------- time c --------------------
        torch.cuda.synchronize()
        time_c = time.time()
        # -----------------------------------------------
        
        # occlusion_mask 只是颜色而已
        # inference alpha, rgb
        if self.DynamicRender.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None

        # alphainv_cum used for bg 
        alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)


        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_loss': can_loss,
            'can_mask': can_mask
        })


        return ret_dict


class Stylized_WithTimeDeformRender_advect_velocity():
    def __init__(
        self,
       DynamicRender,
       StylizedDeformVol,
       StylizedDeformVol_prev,
        **kwargs
    ):
        super(Stylized_WithTimeDeformRender_advect_velocity, self).__init__()
        self.DynamicRender = DynamicRender
        self.StylizedDeformVol = StylizedDeformVol
        self.StylizedDeformVol_prev = StylizedDeformVol_prev

    # not opt in frame 0
    def forward_old(self, rays_o, rays_d, timestep, viewdirs, global_step=None,stylizied_deformed_Scale = 1.0,Post_StyDeform = True,DeltaTime_T =1.0/40.0, **render_kwargs):
        '''Volume rendering'''
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None,  **render_kwargs)
        interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio


        # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)
        rays_pts_ori = rays_pts
        
        _, deform_t_m1, _, _ = self.DynamicRender.deformgrid(rays_pts, timestep - DeltaTime_T, mask_outbbox, **render_kwargs)
         # occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        # # can_mask_prev = timestep == DeltaTime_T
        # # can_mask_prev = can_mask_prev.unsqueeze(-2)
        # # can_shape_prev = -1 * np.ones(len(can_mask_prev.shape), dtype=np.int64)
        # # can_shape_prev[-2] = deform_t_m1.shape[-2]
        # # can_mask_prev = can_mask_prev.expand(*can_shape_prev).squeeze(-1)
        # # # can_loss = torch.mean(torch.abs(deform[can_mask_prev]))
        # # # 把时间=0的部分的offset强制置于0
        # # deform_t_m1[can_mask_prev] = 0.
        
        # can_mask = timestep == 0.
        # can_mask = can_mask.unsqueeze(-2)
        # can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        # can_shape[-2] = deform.shape[-2]
        # can_mask = can_mask.expand(*can_shape).squeeze(-1)
        # can_loss = torch.mean(torch.abs(deform[can_mask]))
        # # 把时间=0的部分的offset强制置于0
        # deform[can_mask] = 0.       


        # 定义一个事情，风格化速度场的空间 是在形变后的
        # this is the base
        # rays_pts_Ut_m1 = rays_pts + deform_t_m1
        
        # Vt_m1 = deform - deform_t_m1
        
        # rays_pts_Ut_m1_2Sample = rays_pts_Ut_m1 + Vt_m1
        
        # if not Post_StyDeform:
        #     rays_pts += deform
        
        #TODO sample prev deform 
        #TODO 考虑要不要用 
        
        deform_stylized_prev, _ = self.StylizedDeformVol_prev(rays_pts, timestep,mask_outbbox, **render_kwargs)
        deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox, **render_kwargs)
        
        alpha = 0.5
        rays_pts += (deform_stylized_prev * alpha + deform_stylized * (1.0 - alpha)) * stylizied_deformed_Scale
        # rays_pts += deform_stylized * stylizied_deformed_Scale
        
        
        occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        
        can_mask = timestep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        can_loss = torch.mean(torch.abs(deform[can_mask]))
        # 把时间=0的部分的offset强制置于0
        deform[can_mask] = 0.
        
        # deform = deform * occlusion_mask.unsqueeze(-1)
        if Post_StyDeform:
            rays_pts += deform

     

        
        # ------------------- time c --------------------
        torch.cuda.synchronize()
        time_c = time.time()
        # -----------------------------------------------

        # inference alpha, rgb
        if self.DynamicRender.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None

        # alphainv_cum used for bg 
        alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)


        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_loss': can_loss,
            'can_mask': can_mask
        })


        return ret_dict
    def forward(self, rays_o, rays_d, timestep, viewdirs, global_step=None,stylizied_deformed_Scale = 1.0,Post_StyDeform = True,DeltaTime_T =1.0/40.0,cal_deform_Approaching_loss = False, **render_kwargs):
        '''Volume rendering'''
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None,  **render_kwargs)
        interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio


        # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)
        
        occlusion_mask, deform_t_m1, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep - DeltaTime_T, mask_outbbox, **render_kwargs)
        occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        
         # occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        # # can_mask_prev = timestep == DeltaTime_T
        # # can_mask_prev = can_mask_prev.unsqueeze(-2)
        # # can_shape_prev = -1 * np.ones(len(can_mask_prev.shape), dtype=np.int64)
        # # can_shape_prev[-2] = deform_t_m1.shape[-2]
        # # can_mask_prev = can_mask_prev.expand(*can_shape_prev).squeeze(-1)
        # # # can_loss = torch.mean(torch.abs(deform[can_mask_prev]))
        # # # 把时间=0的部分的offset强制置于0
        # # deform_t_m1[can_mask_prev] = 0.
        
        # can_mask = timestep == 0.
        # can_mask = can_mask.unsqueeze(-2)
        # can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        # can_shape[-2] = deform.shape[-2]
        # can_mask = can_mask.expand(*can_shape).squeeze(-1)
        # can_loss = torch.mean(torch.abs(deform[can_mask]))
        # # 把时间=0的部分的offset强制置于0
        # deform[can_mask] = 0.       


        # 定义一个事情，风格化速度场的空间 是在形变后的
        # this is the base
        # rays_pts_Ut_m1 = rays_pts + deform_t_m1
        
        # Vt_m1 = deform - deform_t_m1
        
        # rays_pts_Ut_m1_2Sample = rays_pts_Ut_m1 + Vt_m1
        
        # if not Post_StyDeform:
        #     rays_pts += deform
        
        #TODO sample prev deform 
        #TODO 考虑要不要用 
        
        rays_pts_prev =rays_pts + ( deform - deform_t_m1)
        
        test_usefuladvection = False
        if test_usefuladvection:
            # advection
            deform_stylized_prev, _ = self.StylizedDeformVol_prev(rays_pts_prev, timestep,mask_outbbox, **render_kwargs)
            # deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox, **render_kwargs)
            #TODO come back
            alpha = 0.5
            # Deform_delta = (deform_stylized_prev * alpha + deform_stylized * (1.0 - alpha)) * stylizied_deformed_Scale
            Deform_delta = deform_stylized_prev * stylizied_deformed_Scale
        else:
                 # advection
            deform_stylized_prev, _ = self.StylizedDeformVol_prev(rays_pts_prev, timestep,mask_outbbox, **render_kwargs)
            deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox, **render_kwargs)
            #TODO come back
            alpha = 0.5
            Deform_delta = (deform_stylized_prev * alpha + deform_stylized * (1.0 - alpha)) * stylizied_deformed_Scale
           
        
        rays_pts += Deform_delta
        if cal_deform_Approaching_loss:
            deform_Approaching = (deform_stylized_prev - Deform_delta.detach())
            deform_Approaching_loss = torch.mean(torch.abs(deform_Approaching))
        else:
            deform_Approaching_loss = 0.0
        # rays_pts += deform_stylized * stylizied_deformed_Scale
        
        
        occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        
        can_mask = timestep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)

        # 把时间=0的部分的offset强制置于0
        deform[can_mask] = 0.
        
        # deform = deform * occlusion_mask.unsqueeze(-1)
        if Post_StyDeform:
            rays_pts += deform

        # inference alpha, rgb
        if self.DynamicRender.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None

        # alphainv_cum used for bg 
        alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)


        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_mask': can_mask,
            'deform_loss':deform_Approaching_loss
        })


        return ret_dict
    
class Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth():
    def __init__(
        self,
       DynamicRender,
       StylizedDeformVol,
       StylizedDeformVol_prev,
        **kwargs
    ):
        super(Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth, self).__init__()
        self.DynamicRender = DynamicRender
        self.StylizedDeformVol = StylizedDeformVol
        self.StylizedDeformVol_prev = StylizedDeformVol_prev
        self.StylizedDeformVol_current_k0Bypass = None
        self.DynamicDeformVol_prev_qucikTrainUsed = None

    def prepare(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,prevDeformField = None,QucickTrainMode = False,alpha = 0.5,UseHalf = False):
        
        if QucickTrainMode == True:
            if prevDeformField:
                self.DynamicDeformVol_prev_qucikTrainUsed = prevDeformField
            self.StylizedDeformVol = self.StylizedDeformVol_prev
            return
        
        TestOverwrite = False
        # with torch.no_grad():
        sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
        if UseHalf:
            sample_points = sample_points.half()
        
        masks = torch.ones_like(sample_points) < 0.0
        masks = masks.any(dim=-1)

        
        timestep = timestep_ori.reshape(1,1,-1).expand(sample_points.shape[0], sample_points.shape[1],-1)  
        if prevDeformField:
            occlusion_mask, deform_t_m1, mask_d, time_dict = prevDeformField(sample_points, timestep - DeltaTime_T, masks,UseHalf = UseHalf)
        else:
            occlusion_mask, deform_t_m1, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points, timestep - DeltaTime_T, masks,UseHalf = UseHalf)
            
        occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points, timestep, masks,UseHalf = UseHalf)

        if abs(timestep_ori - DeltaTime_T)<0.001:
            deform_t_m1 = 0.0
    
        if TestOverwrite:
            sample_points_deformed = sample_points + deform 
        else:
            sample_points_deformed = sample_points + (deform - deform_t_m1)
        if OverWriteAdvection:
            sample_points_deformed = sample_points
            
        deform_stylized_prev, _ = self.StylizedDeformVol_prev(sample_points_deformed, timestep,masks)
        # becoming slicing

        deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
        deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

        test =  deform_stylized_prev * (1.0 - alpha) + self.StylizedDeformVol.k0 * ( alpha)
        if OverWriteAdvection:
            # import copy
            # test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
            test = self.StylizedDeformVol_prev.k0 + 0.0
            
        self.StylizedDeformVol_current_k0Bypass = test
        
        self.StylizedDeformVol.k0 = torch.nn.Parameter(test)

    # def prepare(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5):
    #     TestOverwrite = False
    #     # with torch.no_grad():
    #     sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
        
    #     masks = torch.ones_like(sample_points) < 0.0
    #     masks = masks.any(dim=-1)

        
    #     timestep = timestep_ori.reshape(1,1,-1).expand(sample_points.shape[0], sample_points.shape[1],-1)  
        
    #     occlusion_mask, deform_t_m1, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points, timestep - DeltaTime_T, masks)
    #     occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points, timestep, masks )
    #     print(timestep_ori)
    #     if (timestep_ori - DeltaTime_T)<0.001:
    #         deform_t_m1 = 0.0
    
    #     if TestOverwrite:
    #         sample_points_deformed = sample_points + deform 
    #     else:
    #         sample_points_deformed = sample_points + deform - deform_t_m1
    #     if OverWriteAdvection:
    #         sample_points_deformed = sample_points
        
    #     deform_stylized_prev, _ = self.StylizedDeformVol_prev(sample_points_deformed, timestep,masks)
    #     # becoming slicing

    #     deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
    #     deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

    #     test =  deform_stylized_prev * (1.0 - alpha) + self.StylizedDeformVol.k0 * ( alpha)
    #     if OverWriteAdvection:
    #         import copy
    #         test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
            
        
    #     self.StylizedDeformVol.k0 = torch.nn.Parameter(test)

    def forward_qucikTrain(self, rays_o, rays_d, timestep, viewdirs, global_step=None,stylizied_deformed_Scale = 1.0,Enablek0Bypass = False,Post_StyDeform = True,DeltaTime_T =1.0/40.0,cal_deform_Approaching_loss = False, **render_kwargs):
        '''Volume rendering'''
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None,  **render_kwargs)
        interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio

        

        occlusion_mask, deform_beforeSty, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        occlusion_mask, deform_beforeSty_prev, mask_d, time_dict = self.DynamicDeformVol_prev_qucikTrainUsed(rays_pts, timestep - DeltaTime_T, mask_outbbox, **render_kwargs)
        
        
        can_mask = timestep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform_beforeSty.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        # 把时间=0的部分的offset强制置于0
        deform_beforeSty[can_mask] = 0.

        delta = torch.ones_like(timestep) * DeltaTime_T
        
        can_mask = abs(timestep - delta) < 0.001
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform_beforeSty_prev.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        # 把时间=0的部分的offset强制置于0
        deform_beforeSty_prev[can_mask] = 0.
    
        rays_pts_advect = rays_pts + deform_beforeSty - deform_beforeSty_prev
        # advection

        
        # TODO add additonal sample to StylizedDeformVol_prev
        if stylizied_deformed_Scale != 0.0:

            deform_stylized, mask_d_stylizied = self.StylizedDeformVol_prev(rays_pts_advect,timestep, mask_outbbox, **render_kwargs)
            
            Deform_delta = deform_stylized * stylizied_deformed_Scale
        else:
            Deform_delta = 0.0
    
        deform = deform_beforeSty
        
        rays_pts += Deform_delta    
        occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        

        deform_Approaching_loss = 0.0
        # rays_pts += deform_stylized * 
        

        
        
        
        # deform = deform * occlusion_mask.unsqueeze(-1)
        if Post_StyDeform:
            rays_pts += deform


        # inference alpha, rgb
        if self.DynamicRender.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None

        # alphainv_cum used for bg 
        alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)


        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_mask': can_mask,
            'deform_loss':deform_Approaching_loss
        })


        return ret_dict        
            
    # Post_StyDeform = True  more make sense
    def forward(self, rays_o, rays_d, timestep, viewdirs, global_step=None,stylizied_deformed_Scale = 1.0,Enablek0Bypass = False,Post_StyDeform = True,DeltaTime_T =1.0/40.0,cal_deform_Approaching_loss = False, **render_kwargs):
        '''Volume rendering'''
        ret_dict = {}

        # sample points on rays
        rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None,  **render_kwargs)
        interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio
        rays_pts = rays_pts.to(rays_o)

        # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)

        if not Post_StyDeform:
            occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)



            can_mask = timestep == 0.
            can_mask = can_mask.unsqueeze(-2)
            can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
            can_shape[-2] = deform.shape[-2]
            can_mask = can_mask.expand(*can_shape).squeeze(-1)
            deform[can_mask] = 0.       


            # 定义一个事情，风格化速度场的空间 是在形变后的
            # this is the base
            # rays_pts_Ut_m1 = rays_pts + deform_t_m1

            # Vt_m1 = deform - deform_t_m1

            # rays_pts_Ut_m1_2Sample = rays_pts_Ut_m1 + Vt_m1

            rays_pts += deform
        
        #TODO sample prev deform 
        #TODO 考虑要不要用 
        
        
        

        # advection

        if stylizied_deformed_Scale != 0.0:
            if Enablek0Bypass:
                deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox,k0overwrite = self.StylizedDeformVol_current_k0Bypass, **render_kwargs)
            else:
                deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox, **render_kwargs)
            #TODO come back
            alpha = 0.5
            Deform_delta = deform_stylized * stylizied_deformed_Scale
            Deform_delta.to(rays_pts)
        else:
            Deform_delta = 0.0
            
        rays_pts += Deform_delta
        
          
        occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        

        deform_Approaching_loss = 0.0
        # rays_pts += deform_stylized * 
        
        can_mask = timestep == 0.
        can_mask = can_mask.unsqueeze(-2)
        can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        can_shape[-2] = deform.shape[-2]
        can_mask = can_mask.expand(*can_shape).squeeze(-1)
        # 把时间=0的部分的offset强制置于0
        deform[can_mask] = 0.
        
        
        
        # deform = deform * occlusion_mask.unsqueeze(-1)
        if Post_StyDeform:
            rays_pts += deform


        # inference alpha, rgb
        if self.DynamicRender.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None

        # alphainv_cum used for bg 
        alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)


        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_mask': can_mask,
            'deform_loss':deform_Approaching_loss
        })


        return ret_dict



def poc_fre(input_data,poc_buf):
    if OverwriteEmbedding:
        
        input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
        input_data_sin = input_data_emb.sin()
        input_data_cos = input_data_emb.sin()
        input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
        return input_data_emb
    else:
        input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
        input_data_sin = input_data_emb.sin()
        input_data_cos = input_data_emb.cos()
        input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
        return input_data_emb


# EnableAnotherModeDelta = True

class Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth():
    def __init__(
        self,
       DynamicRender,
       StylizedDeformVol,
       StylizedDeformVol_prev,
       currentTimeDeltaField,
       currentTimeDeltaField_prev,
       prevModel =None,
        **kwargs
    ):
        super(Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth, self).__init__()
        if EnableIndependentMode:
            self.DynamicRender = copy.deepcopy(DynamicRender) 
        else:
            self.DynamicRender = DynamicRender
        self.StylizedDeformVol = StylizedDeformVol
        self.StylizedDeformVol_prev = StylizedDeformVol_prev
        self.StylizedDeformVol_current_k0Bypass = None
        
        self.TimeDeltaField = currentTimeDeltaField
        self.TimeDeltaField_prev = currentTimeDeltaField_prev
        self.EnableRefMode = True
        self.DeltaMode2 = True
        self.StylizedRadianceField = None
        self.StylizedRadianceField_prev = None
        self.StylizedDeformVol_current_k0Bypass_stylzied = None
        
        

        # total Switch   
        # colors
        self.EnableStyRayField = True
        # self.AdaptStyRaidanceFieldMode_global = False
        self.EnableFeatureGridCopy_optimizeForColors = True
        
        self.AdaptStyRaidanceFieldMode = True
        if not self.EnableStyRayField:
            self.EnableFeatureGridCopy_optimizeForColors = False
            self.AdaptStyRaidanceFieldMode = False
        
        self.DeltaMode2EnableAdevection = True
        # core novel 2
        self.EnableNewAdvectionScehems= EnableNewAdvectionScehems
        self.NewAdvectionSchemeesExtrapolateDeformationFields = False
        
        
        if not EnableNewAdvectionScehems:
            self.NewAdvectionSchemeesExtrapolateDeformationFields = False
        self.NewAdvectionSchemeesExtrapolateDeformationFields_Cache = None
        
        
        self.EnableStyFieldSmooth = EnableStySmooth
        
        self.EnableAdvectionDeformDeltaFieldSmooth = False
        self.EnablePrevSmoothDeformDeltaField = False
        
        self.EnableNewAdvectionScehems_DeltaFieldInReferSpace = True
        
        self.EnableNewAdvectionScehems_enableAnotherSearch = True
        
        self.EnableStatic_styMode= False

        if self.EnableStatic_styMode:
            self.EnableNewAdvectionScehems = False

        self.EnablePureRK2 = True
        
        self.enableExtrapolate_velocityField = True
        
        if not self.EnableNewAdvectionScehems:
            self.EnableNewAdvectionScehems_DeltaFieldInReferSpace = False
            self.EnableNewAdvectionScehems_enableAnotherSearch = False
            self.NewAdvectionSchemeesExtrapolateDeformationFields = False
            
        if self.NewAdvectionSchemeesExtrapolateDeformationFields:
            self.enableExtrapolate_velocityField = True
        
        if self.EnableStyFieldSmooth:
            self.smoothing = GaussianSmoothing(3, 11, 1.0,3)
        if self.EnableAdvectionDeformDeltaFieldSmooth or self.EnablePrevSmoothDeformDeltaField:
            self.AdvectionDeformSmoothing = GaussianSmoothing(3, 11, 3.0,3)

        self.cachedPreparation = None
        self.EnableCachedPrepartion = True

        self.setColorField = False

        if prevModel!=None:
            
            self.cachedPreparation = copy.deepcopy(prevModel.cachedPreparation) 
            self.NewAdvectionSchemeesExtrapolateDeformationFields_Cache = copy.deepcopy(prevModel.NewAdvectionSchemeesExtrapolateDeformationFields_Cache) 
            del prevModel
        

        self.featureNetCopy = None
        
        self.feature_copy = None
        self.featurenet = None
        self.RGBnet = None

        if self.EnableFeatureGridCopy_optimizeForColors:
            if self.DynamicRender.featureGridCopy == None:
                self.DynamicRender.featureGridCopy = copy.deepcopy(self.DynamicRender.feature)
                self.DynamicRender.featureNetCopy = copy.deepcopy(self.DynamicRender.featurenet)
            self.feature_copy = self.DynamicRender.featureGridCopy
            self.featurenet = self.DynamicRender.featureNetCopy
        
        if self.EnableStyRayField:
            if self.DynamicRender.RGBNetCopy == None:
                self.DynamicRender.RGBNetCopy = copy.deepcopy(self.DynamicRender.rgbnet)
            self.rgbnet = self.DynamicRender.RGBNetCopy
        
    def setStyRadField(self,field,prevfield):
        self.StylizedRadianceField = field
        self.StylizedRadianceField_prev = prevfield
        self.setColorField = True
        self.AdaptStyRaidanceFieldMode = True
    
    def getRadianceField(self):
        return self.DynamicRender.rgbnet
    
    def setRadianceField(self,rgbnet):
        self.DynamicRender.rgbnet = rgbnet
    
    def clearGC(self):
        del self.StylizedDeformVol
        del self.StylizedDeformVol_prev
        del self.StylizedDeformVol_current_k0Bypass
        del self.TimeDeltaField

    def prepare(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5,UseHalf = False,**_kwargs):
        if self.EnableRefMode:
            return
        if self.AdaptStyRaidanceFieldMode and self.EnableStyRayField and False:
            return self.prepare_sty(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,**_kwargs)
        else:
            #return self.prepare_old(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,**_kwargs)
            
            if self.DeltaMode2:
                if self.EnablePureRK2:
                    self.prepare_old_nodelta_secondRK(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,
                                                      cuurent_field = self.StylizedDeformVol,
                                                      Prev_field = self.StylizedDeformVol_prev,
                                                      **_kwargs)
                elif self.enableExtrapolate_velocityField:
                    self.prepare_old_nodelta_with_extrapolate(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,**_kwargs)
                else:
                    self.prepare_old_nodelta(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,**_kwargs)
                
            else:
                self.prepare_old(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,**_kwargs) 
            
    def prepare_old(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5,UseHalf = False,**_kwargs):
        
        sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])
        #TODO sample_points ...3 
        if UseHalf:
            sample_points = sample_points.half()

        
        timestep = timestep_ori
            
        # occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points, timestep, masks,UseHalf = UseHalf)

        delta_deform = self.TimeDeltaField.forward_simple(sample_points)
        
    
        if NotenableSemiLagragin:
            sample_points_deformed = sample_points  - delta_deform
        else:
            sample_points_deformed = sample_points  + delta_deform
        if OverWriteAdvection:
            sample_points_deformed = sample_points 
            
        deform_stylized_prev = self.StylizedDeformVol_prev.forward_simple(sample_points_deformed)
        # # becoming slicing
        # if abs(timestep_ori - DeltaTime_T)<0.001:
        #     deform_t_m1 = 0.0
        
        deform_stylized_prev = deform_stylized_prev.reshape(oriShape)
        deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
        deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

        test =  deform_stylized_prev * (1.0 - alpha) + self.StylizedDeformVol.k0 * ( alpha)
        if OverWriteAdvection:
            # import copy
            # test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
            test = self.StylizedDeformVol_prev.k0 + 0.0
            
        self.StylizedDeformVol_current_k0Bypass = test
        del self.StylizedDeformVol.k0
        self.StylizedDeformVol.k0 = torch.nn.Parameter(test)

    
    def getDensity(self,target_pos,times):
        alphas =  self.DynamicRender.getAlphByTime_withFixedPos(target_pos,times)
        return alphas
    
    
    def extrapolateByDensityField(self,target_pos,target_Tensor_ori,time):
        density_current_mask = self.getDensity(target_pos,time) > 0.95
        density_current_mask = density_current_mask.squeeze(-1)
        counter_mask = torch.zeros_like(density_current_mask) * 0.0

        # init
        target_Tensor = target_Tensor_ori * 1.0
        target_Tensor[~density_current_mask.repeat(1,3,1,1,1)] = 0.0
        
        for __ in range(30):
        
            # loop
            # ori valid
            counter_mask[density_current_mask] = 1.0
            counter_mask[~density_current_mask] = 0.0
            
            left_mask = counter_mask[:,:,1:,:,:]
            left_mask = F.pad(left_mask, (0,0, 0,0, 1,0), mode='reflect')

            left_val = target_Tensor[:,:,1:,:,:]
            left_val = F.pad(left_val, (0,0, 0,0, 1,0), mode='reflect')
            
            right_mask = counter_mask[:,:,:-1,:,:]
            right_mask = F.pad(right_mask, (0,0, 0,0, 0,1), mode='reflect')

            right_val = target_Tensor[:,:,:-1,:,:]
            right_val = F.pad(right_val, (0,0, 0,0, 0,1), mode='reflect')
            
            front_mask = counter_mask[:,:,:,1:,:]
            front_mask = F.pad(front_mask, (0,0, 1,0, 0,0), mode='reflect')

            front_val = target_Tensor[:,:,:,1:,:]
            front_val = F.pad(front_val, (0,0, 1,0, 0,0), mode='reflect')
            
                
            behind_mask = counter_mask[:,:,:,:-1,:]
            behind_mask = F.pad(behind_mask, (0,0, 0,1, 0,0), mode='reflect')

            behind_val = target_Tensor[:,:,:,:-1,:]
            behind_val = F.pad(behind_val, (0,0, 0,1, 0,0), mode='reflect')
            
                
            up_mask = counter_mask[:,:,:,:,1:]
            up_mask = F.pad(up_mask, (1,0, 0,0, 0,0), mode='reflect')

            up_val = target_Tensor[:,:,:,:,1:]
            up_val = F.pad(up_val, (1,0, 0,0, 0,0), mode='reflect')
            
            
            down_mask = counter_mask[:,:,:,:,:-1]
            down_mask = F.pad(down_mask, (0,1, 0,0, 0,0), mode='reflect')
  
            down_val = target_Tensor[:,:,:,:,:-1]
            down_val = F.pad(down_val, (0,1, 0,0, 0,0), mode='reflect')     
            
            
            all_mask = left_mask + right_mask + front_mask + behind_mask + up_mask + down_mask
            all_val = left_val + right_val + front_val + behind_val + up_val + down_val
            all_average = all_val / all_mask
            update_pos = all_mask > 0.01
            update_pos = update_pos & (~density_current_mask)
            density_current_mask[update_pos] = True
            repeatted_index= update_pos.repeat(1,3,1,1,1)
            target_Tensor[repeatted_index] = all_average[repeatted_index]
        
        return target_Tensor
    
    
    def prepare_old_nodelta(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5,UseHalf = False,**_kwargs):
        
        Defrom_delta_scale = 1.0
        
        if  self.EnableNewAdvectionScehems:
            #return
            Defrom_delta_scale = 0.0
            # OverWriteAdvection = True
        
        if self.EnableStatic_styMode:
            OverWriteAdvection = True
        
        
        sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])
        #TODO sample_points ...3 

        

         
        deltas_coll = []
        TotalNum = sample_points.shape[0]
        targetSlice = targetSlice_base
        with torch.no_grad():
            for view_batch_start in range(0,TotalNum, targetSlice):
                UpperBound = min(TotalNum,view_batch_start + targetSlice)
                sample_points_slice = sample_points[view_batch_start:UpperBound,...]
                timestep = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)   
                timestep_prev = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)  - DeltaTime_T
                
                times_emb = poc_fre(timestep, self.DynamicRender.time_poc)   
                
                times_emb_prev = poc_fre(timestep_prev, self.DynamicRender.time_poc)  
                
                times_feature = self.DynamicRender.timenet(times_emb) 
                times_feature_prev = self.DynamicRender.timenet(times_emb_prev) 
                
                rays_pts_emb = poc_fre(sample_points_slice, self.DynamicRender.pos_poc)

                deform_t_m1 = self.DynamicRender.getDeformation(sample_points_slice, timestep - DeltaTime_T,
                                                                                                rays_pts_emb,times_feature_prev
                                                                                                )
                    
                deform = self.DynamicRender.getDeformation(sample_points_slice, timestep,
                                                                                            rays_pts_emb,times_feature
                                                                                            )


                Defrom_delta_slice =  deform - deform_t_m1
                # Defrom_delta_slice =  torch.ones_like(deform)
                deltas_coll.append(Defrom_delta_slice)
            
            Defrom_delta = torch.cat(deltas_coll) * Defrom_delta_scale

        
        
        # TODO make a smoothing here 
        # Defrom_delta = Defrom_delta.reshape(oriShape)
        # print(sample_points_deformed.shape)
        # print(sample_points.shape)
            if self.EnableAdvectionDeformDeltaFieldSmooth:
                print('deform advection smooth')
                
                Defrom_delta = Defrom_delta
                Defrom_delta = Defrom_delta.reshape(oriShape)
                Defrom_delta = Defrom_delta.unsqueeze(0)
                Defrom_delta = Defrom_delta.permute(0,4,1,2,3)
                # smooth delta_deform
                input = Defrom_delta
                input = F.pad(input, (5, 5, 5, 5,5,5), mode='reflect')
                output = self.AdvectionDeformSmoothing(input)
                Defrom_delta = (output) 
                Defrom_delta = Defrom_delta.permute(0,2,3,4,1)
                Defrom_delta = Defrom_delta.reshape([-1,3])
        
        
        delta_deform = self.TimeDeltaField.forward_simple(sample_points)
        if OverWriteAdvection:
            sample_points_deformed = sample_points
        else:
            if self.EnableNewAdvectionScehems:
                if NotenableSemiLagragin:
                    sample_points_deformed = sample_points - Defrom_delta 
                else:
                    sample_points_deformed = sample_points + Defrom_delta       
            else:
                
                if NotenableSemiLagragin:
                    sample_points_deformed = sample_points - Defrom_delta + delta_deform
                else:
                    sample_points_deformed = sample_points + Defrom_delta + delta_deform
        
        
            
        deform_stylized_prev = self.StylizedDeformVol_prev.forward_simple(sample_points_deformed)
        # # becoming slicing
        # if abs(timestep_ori - DeltaTime_T)<0.001:
        #     deform_t_m1 = 0.0
        
        deform_stylized_prev = deform_stylized_prev.reshape(oriShape)
        deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
        deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

        test =  deform_stylized_prev * (1.0 - alpha) + self.StylizedDeformVol.k0 * ( alpha)
        if NotenableSemiLagragin:
            test = sample_points_deformed + delta_deform
        
        
        if OverWriteAdvection:
            # import copy
            # test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
            test = self.StylizedDeformVol_prev.k0 + 0.0
            
        self.StylizedDeformVol_current_k0Bypass = test
        del self.StylizedDeformVol.k0
        self.StylizedDeformVol.k0 = torch.nn.Parameter(test)


    def prepare_old_nodelta_secondRK(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5,UseHalf = False,
                                     cuurent_field = None,Prev_field = None,
                                     **_kwargs):
        
        Defrom_delta_scale = 1.0

        if EnableIndependentMode:
            OverWriteAdvection = True
        
        if  self.EnableNewAdvectionScehems:
            #return
            Defrom_delta_scale = 0.0
            # OverWriteAdvection = True
        
        if self.EnableStatic_styMode:
            OverWriteAdvection = True
        
        
        sample_points = Prev_field.get_grid_worldcoords3()
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])
        #TODO sample_points ...3 

        

         
        deltas_coll = []
        TotalNum = sample_points.shape[0]
        targetSlice = targetSlice_base
        with torch.no_grad():
            for view_batch_start in range(0,TotalNum, targetSlice):
                UpperBound = min(TotalNum,view_batch_start + targetSlice)
                sample_points_slice = sample_points[view_batch_start:UpperBound,...]
                timestep = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)   
                timestep_prev = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)  - DeltaTime_T
                
                times_emb = poc_fre(timestep, self.DynamicRender.time_poc)   
                
                times_emb_prev = poc_fre(timestep_prev, self.DynamicRender.time_poc)  
                
                times_feature = self.DynamicRender.timenet(times_emb) 
                times_feature_prev = self.DynamicRender.timenet(times_emb_prev) 
                
                rays_pts_emb = poc_fre(sample_points_slice, self.DynamicRender.pos_poc)

                deform_t_m1 = self.DynamicRender.getDeformation(sample_points_slice, timestep - DeltaTime_T,
                                                                                                rays_pts_emb,times_feature_prev
                                                                                                )
                    
                deform = self.DynamicRender.getDeformation(sample_points_slice, timestep,
                                                                                            rays_pts_emb,times_feature
                                                                                            )


                Defrom_delta_slice =  deform - deform_t_m1
                # Defrom_delta_slice =  torch.ones_like(deform)
                deltas_coll.append(Defrom_delta_slice)
            
            Defrom_delta = torch.cat(deltas_coll) * Defrom_delta_scale

        
        sample_points_deformed = sample_points  + Defrom_delta * 0.5 
        
        
        deltas_coll = []
        TotalNum = sample_points.shape[0]
        targetSlice = targetSlice_base
        with torch.no_grad():
            for view_batch_start in range(0,TotalNum, targetSlice):
                UpperBound = min(TotalNum,view_batch_start + targetSlice)
                sample_points_slice = sample_points_deformed[view_batch_start:UpperBound,...]
                timestep = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)   
                timestep_prev = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)  - DeltaTime_T
                
                times_emb = poc_fre(timestep, self.DynamicRender.time_poc)   
                
                times_emb_prev = poc_fre(timestep_prev, self.DynamicRender.time_poc)  
                
                times_feature = self.DynamicRender.timenet(times_emb) 
                times_feature_prev = self.DynamicRender.timenet(times_emb_prev) 
                
                rays_pts_emb = poc_fre(sample_points_slice, self.DynamicRender.pos_poc)

                deform_t_m1 = self.DynamicRender.getDeformation(sample_points_slice, timestep - DeltaTime_T,
                                                                                                rays_pts_emb,times_feature_prev
                                                                                                )
                    
                deform = self.DynamicRender.getDeformation(sample_points_slice, timestep,
                                                                                            rays_pts_emb,times_feature
                                                                                            )


                Defrom_delta_slice =  deform - deform_t_m1
                # Defrom_delta_slice =  torch.ones_like(deform)
                deltas_coll.append(Defrom_delta_slice)
            
            Defrom_delta = torch.cat(deltas_coll) * Defrom_delta_scale
            
            
            
        self.TimeDeltaField = self.TimeDeltaField.cuda()
        
        delta_deform = self.TimeDeltaField.forward_simple(sample_points)
        if OverWriteAdvection:
            sample_points_deformed = sample_points
        else:
            if self.EnableNewAdvectionScehems:
                if NotenableSemiLagragin:
                    sample_points_deformed = sample_points - Defrom_delta 
                else:
                    sample_points_deformed = sample_points + Defrom_delta       
            else:
                
                if NotenableSemiLagragin:
                    sample_points_deformed = sample_points - Defrom_delta + delta_deform
                else:
                    sample_points_deformed = sample_points + Defrom_delta + delta_deform
        
        
            
        deform_stylized_prev = Prev_field.forward_simple(sample_points_deformed)
        # # becoming slicing
        # if abs(timestep_ori - DeltaTime_T)<0.001:
        #     deform_t_m1 = 0.0
        
        deform_stylized_prev = deform_stylized_prev.reshape(oriShape)
        deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
        deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

        test =  deform_stylized_prev * (1.0 - alpha) + cuurent_field.k0 * ( alpha)
        if NotenableSemiLagragin:
            test = sample_points_deformed + delta_deform
        
        
        if OverWriteAdvection:
            # import copy
            # test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
            test = cuurent_field.k0 + 0.0
            
        self.StylizedDeformVol_current_k0Bypass = test
        del cuurent_field.k0
        cuurent_field.k0 = torch.nn.Parameter(test)



    def prepare_old_nodelta_with_extrapolate(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5,UseHalf = False,**_kwargs):
        
        Defrom_delta_scale = 1.0
        
        
        
        if  self.EnableNewAdvectionScehems:
            #return
            Defrom_delta_scale = 0.0
            
            # OverWriteAdvection = True
        
        if self.EnableStatic_styMode:
            OverWriteAdvection = True
        
        
        sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
        # print('-----------')
        # print(sample_points.shape)
        sample_points_With_oriShape = sample_points * 1.0
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])
        #TODO sample_points ...3 

        

         
        deltas_coll = []
        deltas_ori = []
        delta_m1 = []
        TotalNum = sample_points.shape[0]
        targetSlice = targetSlice_base
        Defrom_delta = None
        if (not self.EnableCachedPrepartion) or (self.cachedPreparation == None):
        
            with torch.no_grad():
                for view_batch_start in range(0,TotalNum, targetSlice):
                    UpperBound = min(TotalNum,view_batch_start + targetSlice)
                    sample_points_slice = sample_points[view_batch_start:UpperBound,...]
                    timestep = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)   
                    timestep_prev = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)  - DeltaTime_T
                    
                    times_emb = poc_fre(timestep, self.DynamicRender.time_poc)   
                    
                    times_emb_prev = poc_fre(timestep_prev, self.DynamicRender.time_poc)  
                    
                    times_feature = self.DynamicRender.timenet(times_emb) 
                    times_feature_prev = self.DynamicRender.timenet(times_emb_prev) 
                    
                    rays_pts_emb = poc_fre(sample_points_slice, self.DynamicRender.pos_poc)

                    deform_t_m1 = self.DynamicRender.getDeformation(sample_points_slice, timestep - DeltaTime_T,
                                                                                                    rays_pts_emb,times_feature_prev
                                                                                                    )
                        
                    deform = self.DynamicRender.getDeformation(sample_points_slice, timestep,
                                                                                                rays_pts_emb,times_feature
                                                                                                )

                    deltas_ori.append(deform)
                    delta_m1.append(deform_t_m1)
                    # Defrom_delta_slice =  deform - deform_t_m1
                    # Defrom_delta_slice =  torch.ones_like(deform)
                    # deltas_coll.append(Defrom_delta_slice)
            
            
                deform_ori = torch.cat(deltas_ori) 
                deform_m1 = torch.cat(delta_m1) 
                
                if self.EnablePrevSmoothDeformDeltaField:
                    deform_ori = deform_ori
                    deform_ori = deform_ori.reshape(oriShape)
                    deform_ori = deform_ori.unsqueeze(0)
                    deform_ori = deform_ori.permute(0,4,1,2,3)
                    # smooth delta_deform
                    input = deform_ori
                    input = F.pad(input, (5, 5, 5, 5,5,5), mode='reflect')
                    output = self.AdvectionDeformSmoothing(input)
                    deform_ori = (output) 
                    deform_ori = deform_ori.permute(0,2,3,4,1)
                    deform_ori = deform_ori.reshape([-1,3])
    
                    deform_m1 = deform_m1
                    deform_m1 = deform_m1.reshape(oriShape)
                    deform_m1 = deform_m1.unsqueeze(0)
                    deform_m1 = deform_m1.permute(0,4,1,2,3)
                    # smooth delta_deform
                    input = deform_m1
                    input = F.pad(input, (5, 5, 5, 5,5,5), mode='reflect')
                    output = self.AdvectionDeformSmoothing(input)
                    deform_m1 = (output) 
                    deform_m1 = deform_m1.permute(0,2,3,4,1)
                    deform_m1 = deform_m1.reshape([-1,3])
            
                
                if True :
                    deform_ori = deform_ori
                    deform_ori = deform_ori.reshape(oriShape)

                    deform_ori = deform_ori.unsqueeze(0)
                    deform_ori = deform_ori.permute(0,4,1,2,3)

                    deform_ori = self.extrapolateByDensityField(sample_points_With_oriShape,deform_ori,timestep_ori)
                    
                    if self.NewAdvectionSchemeesExtrapolateDeformationFields:
                        self.NewAdvectionSchemeesExtrapolateDeformationFields_Cache = deform_ori
                    
                    deform_ori = deform_ori.permute(0,2,3,4,1)
                    deform_ori = deform_ori.reshape([-1,3])



                    deform_m1 = deform_m1
                    deform_m1 = deform_m1.reshape(oriShape)
                    deform_m1 = deform_m1.unsqueeze(0)
                    deform_m1 = deform_m1.permute(0,4,1,2,3)

                    deform_m1 = self.extrapolateByDensityField(sample_points_With_oriShape,deform_m1,timestep_ori - DeltaTime_T)
                    deform_m1 = deform_m1.permute(0,2,3,4,1)
                    deform_m1 = deform_m1.reshape([-1,3])
                Defrom_delta = (deform_ori - deform_m1) * Defrom_delta_scale
                
                # Defrom_delta = torch.cat(deltas_coll) * Defrom_delta_scale 

            

                if self.EnableAdvectionDeformDeltaFieldSmooth:
                    print('deform advection smooth')
                    
                    Defrom_delta = Defrom_delta
                    Defrom_delta = Defrom_delta.reshape(oriShape)
                    Defrom_delta = Defrom_delta.unsqueeze(0)
                    Defrom_delta = Defrom_delta.permute(0,4,1,2,3)
                    # smooth delta_deform
                    input = Defrom_delta
                    input = F.pad(input, (5, 5, 5, 5,5,5), mode='reflect')
                    output = self.AdvectionDeformSmoothing(input)
                    Defrom_delta = (output) 
                    Defrom_delta = Defrom_delta.permute(0,2,3,4,1)
                    Defrom_delta = Defrom_delta.reshape([-1,3])
                    
                if ( self.EnableCachedPrepartion):
                    self.cachedPreparation = Defrom_delta.cpu()
        else:
            Defrom_delta = self.cachedPreparation.cuda()
            
        delta_deform = self.TimeDeltaField.forward_simple(sample_points)
        if OverWriteAdvection:
            sample_points_deformed = sample_points
        else:
            if self.EnableNewAdvectionScehems:
                if NotenableSemiLagragin:
                    sample_points_deformed = sample_points - Defrom_delta 
                else:
                    sample_points_deformed = sample_points + Defrom_delta       
            else:
                
                if NotenableSemiLagragin:
                    sample_points_deformed = sample_points - Defrom_delta + delta_deform
                else:
                    sample_points_deformed = sample_points + Defrom_delta + delta_deform
        
        
            
        deform_stylized_prev = self.StylizedDeformVol_prev.forward_simple(sample_points_deformed)
        # # becoming slicing
        # if abs(timestep_ori - DeltaTime_T)<0.001:
        #     deform_t_m1 = 0.0
        
        deform_stylized_prev = deform_stylized_prev.reshape(oriShape)
        deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
        deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

        test =  deform_stylized_prev * (1.0 - alpha) + self.StylizedDeformVol.k0 * ( alpha)
        if NotenableSemiLagragin:
            test = sample_points_deformed + delta_deform
        
        
        if OverWriteAdvection:
            # import copy
            # test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
            test = self.StylizedDeformVol_prev.k0 + 0.0
            
        self.StylizedDeformVol_current_k0Bypass = test
        del self.StylizedDeformVol.k0
        self.StylizedDeformVol.k0 = torch.nn.Parameter(test)



 
    def prepare_sty(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5,UseHalf = False,**_kwargs):
        
        if self.DeltaMode2:
            self.prepare_old_nodelta(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,**_kwargs)
        else:
            self.prepare_old(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = OverWriteAdvection,alpha = alpha,UseHalf = UseHalf,**_kwargs)
        #print('should not hit')



        sample_points = self.StylizedRadianceField_prev.get_grid_worldcoords3()
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])

        delta_deform = self.TimeDeltaField.forward_simple(sample_points)
        
    
        if NotenableSemiLagragin:
            sample_points_deformed = sample_points  - delta_deform
        else:
            sample_points_deformed = sample_points  + delta_deform
        
        if OverWriteAdvection:
            sample_points_deformed = sample_points 
            
        deform_stylized_prev = self.StylizedRadianceField_prev.forward_grid(sample_points_deformed)
        # # becoming slicing
        # if abs(timestep_ori - DeltaTime_T)<0.001:
        #     deform_t_m1 = 0.0
        
        deform_stylized_prev = deform_stylized_prev.reshape([oriShape[0],oriShape[1],oriShape[2],self.StylizedRadianceField_prev.k0_dim] )
        deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
        deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

        test =  deform_stylized_prev * (1.0 - alpha) + self.StylizedRadianceField.k0 * ( alpha)
        if OverWriteAdvection:
            # import copy
            # test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
            test = self.StylizedRadianceField_prev.k0 + 0.0
            
        self.StylizedDeformVol_current_k0Bypass_stylzied = test
        del self.StylizedRadianceField.k0
        self.StylizedRadianceField.k0 = torch.nn.Parameter(test)
        # print('prepare_sty hit')

    def Init_TimeDeltaField_deltaMode2(self,timestep_ori, DeltaTime_T =1.0/40.0,UseHalf = False):
        
        self.prepare_old_nodelta_secondRK(timestep_ori, DeltaTime_T =DeltaTime_T,OverWriteAdvection = False,alpha = 0.0001,UseHalf = UseHalf,
                                    cuurent_field = self.TimeDeltaField,
                                    Prev_field = self.TimeDeltaField_prev)
        return
           

    def Init_TimeDeltaField(self,timestep_ori, DeltaTime_T =1.0/40.0,UseHalf = False):
        
        if self.DeltaMode2:
            if self.DeltaMode2EnableAdevection:
                self.Init_TimeDeltaField_deltaMode2(timestep_ori,DeltaTime_T)
            return
        
        TestOverwrite = False
        # with torch.no_grad():
        sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
        
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])
         
        deltas_coll = []
        TotalNum = sample_points.shape[0]
        targetSlice = targetSlice_base
        
        for view_batch_start in range(0,TotalNum, targetSlice):
            UpperBound = min(TotalNum,view_batch_start + targetSlice)
            sample_points_slice = sample_points[view_batch_start:UpperBound,...]
            timestep = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)   
            timestep_prev = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)  - DeltaTime_T
            
            times_emb = poc_fre(timestep, self.DynamicRender.time_poc)   
            
            times_emb_prev = poc_fre(timestep_prev, self.DynamicRender.time_poc)  
            
            times_feature = self.DynamicRender.timenet(times_emb) 
            times_feature_prev = self.DynamicRender.timenet(times_emb_prev) 
            
            rays_pts_emb = poc_fre(sample_points_slice, self.DynamicRender.pos_poc)

            deform_t_m1 = self.DynamicRender.getDeformation(sample_points_slice, timestep - DeltaTime_T,
                                                                                               rays_pts_emb,times_feature_prev
                                                                                               )
                
            deform = self.DynamicRender.getDeformation(sample_points_slice, timestep,
                                                                                          rays_pts_emb,times_feature
                                                                                          )


            Defrom_delta_slice =  deform - deform_t_m1
            # Defrom_delta_slice =  torch.ones_like(deform)
            deltas_coll.append(Defrom_delta_slice)
        
        Defrom_delta = torch.cat(deltas_coll) 
        Defrom_delta = Defrom_delta.reshape(oriShape)
        
        
        
        # becoming slicing

        Defrom_delta = Defrom_delta.unsqueeze(0)
        Defrom_delta = Defrom_delta.permute(0,4,1,2,3)
        del self.TimeDeltaField.k0
        self.TimeDeltaField.k0 = torch.nn.Parameter(Defrom_delta)

    def stySmoothe(self):
        if not self.EnableStyFieldSmooth:
            return
        # self.StylizedDeformVol.k0 = torch.nn.Parameter(test)
        with torch.no_grad():
            print('----')
            input = self.StylizedDeformVol.k0.grad
            mask = abs(input) < 0.001
            smalls = input[mask]
            input = F.pad(input, (5, 5, 5, 5,5,5), mode='reflect')
            output = self.smoothing(input)
            output[mask] = smalls
            self.StylizedDeformVol.k0.grad = (output) 
 
 
            
    # Post_StyDeform = True  more make sense
    def forward(self, rays_o, rays_d, viewdirs,times_sel, cam_sel=None,bg_points_sel=None,global_step=None,stylizied_deformed_Scale = 1.0,
                Enablek0Bypass = False,Post_StyDeform = True,DeltaTime_T =1.0/40.0,cal_deform_Approaching_loss = False, renderStyImgaeWithoutGrad = False,
                FrameNum = 2, **render_kwargs):
        
        # def forward(self, rays_o, rays_d, viewdirs,times_sel, cam_sel=None,bg_points_sel=None,global_step=None,
        #         using_stylizedMode = False,
        #         **render_kwargs):
        
        if DisableGeoChanges:
            stylizied_deformed_Scale = 0.0
            
        # print('this is in sty reached')
        result = self.DynamicRender.forward_stylizied(rays_o, rays_d, viewdirs,times_sel, cam_sel,bg_points_sel,global_step,                        
                           using_stylizedMode = True,deform_stylizied_models = self,
                        stylizied_deformed_Scale = stylizied_deformed_Scale, Enablek0Bypass = Enablek0Bypass,DeltaTime_T = DeltaTime_T, FrameNum = FrameNum,renderStyImgaeWithoutGrad = renderStyImgaeWithoutGrad,
                        **render_kwargs
                           )
        return result
        '''Volume rendering'''
        ret_dict = {}
        printDtype = False
        # sample points on rays
        rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None,  **render_kwargs)
        interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio
        rays_pts = rays_pts.to(rays_o)

        # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)

        # if not Post_StyDeform:
        #     occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)



        #     can_mask = timestep == 0.
        #     can_mask = can_mask.unsqueeze(-2)
        #     can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
        #     can_shape[-2] = deform.shape[-2]
        #     can_mask = can_mask.expand(*can_shape).squeeze(-1)
        #     deform[can_mask] = 0.       
        #     rays_pts += deform
        
        #TODO sample prev deform 
        #TODO 考虑要不要用 
        
        
        

        # advection

        if stylizied_deformed_Scale >= 0.01:
            if Enablek0Bypass:
                deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox,k0overwrite = self.StylizedDeformVol_current_k0Bypass, **render_kwargs)
            else:
                deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox, **render_kwargs)
            #TODO come back
            if printDtype:
                print(deform_stylized.dtype)
            sty_Deform_delta = deform_stylized * stylizied_deformed_Scale
            sty_Deform_delta.to(rays_pts)
        else:
            sty_Deform_delta = 0.0
            
        rays_pts += sty_Deform_delta
        
        can_mask = None
        deform_Approaching_loss = 0.0
        if self.EnableRefMode:
            occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
            occlusion_mask = occlusion_mask.to(rays_o)

            rays_pts += deform * 1.0
        else:
        # TODO add deform
            if FrameNum == 0:
                # print(timestep)
                occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
                occlusion_mask = occlusion_mask.to(rays_o)

                rays_pts += deform * 0.0
            else:
         
                occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep - DeltaTime_T * 1.0, mask_outbbox, **render_kwargs)
                occlusion_mask = occlusion_mask.to(rays_o)



                
                # rays_pts += deform_stylized * 
                
                can_mask = timestep <= DeltaTime_T
                
                can_mask = can_mask.unsqueeze(-2)
                can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
                can_shape[-2] = deform.shape[-2]
                can_mask = can_mask.expand(*can_shape).squeeze(-1)
                # 把时间=0的部分的offset强制置于0
                deform[can_mask] = 0.
                
                
                delta_deform, mask_d = self.TimeDeltaField(rays_pts, timestep, mask_outbbox)
                deform += delta_deform 
                

                rays_pts += deform




        # inference alpha, rgb
        if self.DynamicRender.deformgrid.deformnet_output > 3:
            occ_input = occlusion_mask
        else:
            occ_input = None
        # occ_input = None
        # if OverWritedisableOcclusionMask:
        #     occ_input = None
        # alphainv_cum used for bg 
        alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
            rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)
        if printDtype:
            print(rgb.dtype)

        # Ray marching
        rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
        rgb_marched = rgb_marched.clamp(0, 1)
        depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
        depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']

        disp = 1 / depth
        ret_dict.update({
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'depth': depth,
            'disp': disp,
            'mask': mask_r,
            'deformation': deform,
            'occlusion': occlusion_mask[mask_d],
            'can_mask': can_mask,
            'deform_loss':deform_Approaching_loss
        })


        return ret_dict



''' Module for the searched coarse geometry
It supports query for the known free space and unknown space.
'''
class MaskCache(nn.Module):
    def __init__(self, path, mask_cache_thres, ks=3):
        super().__init__()
        st = torch.load(path)
        self.mask_cache_thres = mask_cache_thres
        self.register_buffer('xyz_min', torch.FloatTensor(st['MaskCache_kwargs']['xyz_min']))
        self.register_buffer('xyz_max', torch.FloatTensor(st['MaskCache_kwargs']['xyz_max']))
        if 'radiancegrid.density' in st['model_state_dict'].keys():
            den_key = 'radiancegrid.density'
        else:
            den_key = 'density'
        self.register_buffer('density', F.max_pool3d(
            st['model_state_dict'][den_key], kernel_size=ks, padding=ks//2, stride=1))
        self.act_shift = st['MaskCache_kwargs']['act_shift']
        self.voxel_size_ratio = st['MaskCache_kwargs']['voxel_size_ratio']
        self.nearest = st['MaskCache_kwargs'].get('nearest', False)
        self.pre_act_density = st['MaskCache_kwargs'].get('pre_act_density', False)
        self.in_act_density = st['MaskCache_kwargs'].get('in_act_density', False)

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if self.nearest:
            density = F.grid_sample(self.density, ind_norm, align_corners=True, mode='nearest')
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        elif self.pre_act_density:
            alpha = 1 - torch.exp(-F.softplus(self.density + self.act_shift) * self.voxel_size_ratio)
            alpha = F.grid_sample(self.density, ind_norm, align_corners=True)
        elif self.in_act_density:
            density = F.grid_sample(F.softplus(self.density + self.act_shift), ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-density * self.voxel_size_ratio)
        else:
            density = F.grid_sample(self.density, ind_norm, align_corners=True)
            alpha = 1 - torch.exp(-F.softplus(density + self.act_shift) * self.voxel_size_ratio)
        alpha = alpha.reshape(*shape)
        return (alpha >= self.mask_cache_thres)


class MaskCacheDeform(nn.Module):
    def __init__(self, path, mask_cache_thres, train_times, ks=3):
        super().__init__()
        print('dvgo: making cache, start')
        self.mask_cache_thres = mask_cache_thres
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = utils.load_model_dynamic(
            [VoxRendererDynamic, DeformVoxGO, DirectVoxGO],
            path
            ).to(device)
        self.xyz_max = model.deformgrid.xyz_max
        self.xyz_min = model.deformgrid.xyz_min

        alphas =[]
        with torch.no_grad():
            for i in range(0, len(train_times), 1):
                # print('dvgo:  making cache, processing time step: ', train_times[i])
                ti = train_times[i]
                alpha, _, _ = model.get_deform_alpha_rgb(ti)
                alphas.append(alpha)
        self.alpha, _ = torch.max(torch.stack(alphas, dim=-1), dim=-1)
        self.alpha = F.max_pool3d(
            self.alpha, kernel_size=ks, padding=ks//2, stride=1)
        del model
        print('dvgo:  making cache, finished')

    @torch.no_grad()
    def forward(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        al = F.grid_sample(self.alpha, ind_norm, align_corners=True)
        al = al.reshape(*shape)
        return (al >= self.mask_cache_thres)


''' Misc
'''
def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[...,[0]]), p.clamp_min(1e-10).cumprod(-1)], -1)

def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1-alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum

def total_variation4(v, mask=None):
    mask = None
    tv2 = v.diff(dim=2, n=1).abs()
    tv3 = v.diff(dim=3, n=1).abs()
    tv4 = v.diff(dim=4, n=1).abs()
    if mask is not None:
        tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
        tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
        tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

def total_variation(v, mask=None):
    mask = None
    tv2 = v.diff(dim=2, n=2).abs().mean(dim=1, keepdim=True)
    tv3 = v.diff(dim=3, n=2).abs().mean(dim=1, keepdim=True)
    tv4 = v.diff(dim=4, n=2).abs().mean(dim=1, keepdim=True)
    if mask is not None:
        maska = mask[:,:,:-1] & mask[:,:,1:]
        tv2 = tv2[maska[:,:,:-1] & maska[:,:,1:]]
        maskb = mask[:,:,:,:-1] & mask[:,:,:,1:]
        tv3 = tv3[maskb[:,:,:,:-1] & maskb[:,:,:,1:]]
        maskc = mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]
        tv4 = tv4[maskc[:,:,:,:,:-1] & maskc[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

def total_variation2(v, mask=None):
    tv2 = torch.square(v.diff(dim=2, n=2))
    tv3 = torch.square(v.diff(dim=3, n=2))
    tv4 = torch.square(v.diff(dim=4, n=2))
    if mask is not None:
        tv2 = tv2[mask[:,:,:-1] & mask[:,:,1:]]
        tv3 = tv3[mask[:,:,:,:-1] & mask[:,:,:,1:]]
        tv4 = tv4[mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]]
    return (tv2.mean() + tv3.mean() + tv4.mean()) / 3


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def get_training_rays_dynamic(rgb_tr, times_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    times_tr = times_tr.reshape(-1,1,1,1).expand(-1, rgb_tr.shape[1], rgb_tr.shape[2], -1)
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = model.sample_ray(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i+CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling_dynamic(rgb_tr_ori, times_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    times_tr_ori = times_tr_ori.reshape(-1,1,1,1).expand(-1, rgb_tr_ori.shape[1], rgb_tr_ori.shape[2], -1)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    times_tr = torch.zeros([N,1], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, tim, (H, W), K in zip(train_poses, rgb_tr_ori, times_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.ones(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            rays_pts, mask_outbbox = model.radiancegrid.sample_ray(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
            mask_outbbox[~mask_outbbox] |= (~model.radiancegrid.mask_cache(rays_pts[~mask_outbbox]))
            mask[i:i+CHUNK] &= (~mask_outbbox).any(-1).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        times_tr[top:top+n].copy_(tim[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    times_tr = times_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS





import torch
import numpy as np
import math
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def projection(pts : torch.Tensor, w2c, near, far, H = 400, W = 400, Fx = 1.0, Fy = 1.0):
    '''
    pts.shape = (n, 3) in w
    z_sign 1.0和世界坐标系手性相同，-1相反

    return:
    pts in NDC 
    '''
    IsDnerf = True
    if IsDnerf:
        FovY = focal2fov(Fy, H)
        FovX = focal2fov(Fx, W)
        tanHalfFovY = np.tan((FovY / 2))
        tanHalfFovX = np.tan((FovX / 2))

        top = tanHalfFovY * near
        bottom = -top
        right = tanHalfFovX * near
        left = -right

        proj = torch.zeros(4, 4, dtype=torch.double)
        z_sign = -1.0 

        proj[0, 0] = 2.0 * near / (right - left)
        proj[1, 1] = 2.0 * near / (top - bottom)
        proj[0, 2] = (right + left) / (right - left)
        proj[1, 2] = (top + bottom) / (top - bottom)
        proj[3, 2] = z_sign
        proj[2, 2] = z_sign * far / (far - near)
        proj[2, 3] = -(far * near) / (far - near)

        
        
        # print(w2c)
        # print(proj)
        # print('---')
        # print(pts[0])
        
        proj = proj.to(pts)
        t = torch.cat((pts, torch.ones(pts.shape[0], 1)), dim=1).T
        t =  torch.matmul(w2c, t)
        clip_pts = torch.matmul(proj, t).T 

        res=  clip_pts[:, :3]  / clip_pts[:, 3].reshape(-1, 1)
   
        res[...,2] = res[...,2] * 2.0 - 1.0
        return res
    if not IsDnerf:
        pass
