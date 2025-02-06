import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch_scatter import segment_coo
from lib import VoxelDeformation 
from lib.deformRegular import *
# from lib.cuda_gridsample import *
# from grid_sample import cuda_gridsample as cu

import DataDefines

# 'cuda' 'ori' 'torch'
Deform_Grid_type = 'torch'

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
        verbose=True)

class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_views=3, input_ch_time=9, skips=[],):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self._time, self._time_out = self.create_net()

    def create_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 2):
            layer = nn.Linear
            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch
            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        return net_final(h)

        

    def forward(self, input_pts, ts):
        dx = self.query_time(input_pts, ts, self._time, self._time_out)
        input_pts_orig = input_pts[:, :3]
        out=input_pts_orig + dx
        return out
enableDelatemode2 = VoxelDeformation.EnableNewAdvectionScehems

# ss
# Model
# overwrite views
class RGBNet(nn.Module):
    def __init__(self, D=3, W=256, h_ch=256, views_ch=33, pts_ch=27, times_ch=17, output_ch=3):
        """ 
        """
        super(RGBNet, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = h_ch
        self.input_ch_views = views_ch
        self.input_ch_pts = pts_ch
        self.input_ch_times = times_ch
        self.output_ch = output_ch
        self.feature_linears = nn.Linear(self.input_ch, W)
        self.views_linears = nn.Sequential(nn.Linear(W+self.input_ch_views, W//2),nn.ReLU(),nn.Linear(W//2, self.output_ch))
        #self.views_linears = nn.Sequential(nn.Linear(W, W//2),nn.ReLU(),nn.Linear(W//2, self.output_ch))
    def forward(self, input_h, input_views):
        feature = self.feature_linears(input_h)
        feature_views = torch.cat([feature, input_views],dim=-1)
        outputs = self.views_linears(feature_views)
        return outputs





# UseDeformationGridOverWriteMLP = True

# may be need use this UseDeformationGridTimeFeature
UseDeformationGridTimeFeature = False

OverWriteMulFeatureGrid = False

UseDitheredRaySampling = False
MarkZeroTimeFeature = False
StaticVersion = False
EnableDeformationZero = False



#EnableNewAdvectionScehems = True

rgb_scale = 1.0

class TiNeuVox(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0, 
                 deform_num_voxels = 0,deform_num_voxels_base= 0,
                 add_cam=False,
                 alpha_init=None, fast_color_thres=0,
                 voxel_dim=0, defor_depth=3, net_width=128,
                 posbase_pe=10, viewbase_pe=4, timebase_pe=8, gridbase_pe=2,loadfromckpt = True,UseDeformationGridOverWriteMLP = True,
                 UsePureVoxelDensity = False, OverWriteFeatureNetInputWithGridSamplesOnly = False,AdditionalPtsEmbedding = False,
                 **kwargs):
        
        super(TiNeuVox, self).__init__()
        
        # timebase_pe = 0
        # gridbase_pe = 0
        # viewbase_pe = 0
        # posbase_pe = 5
        
        global rgb_scale
        if UsePureVoxelDensity:
            rgb_scale = 0.001
            
        
        self.AdditionalPtsEmbedding = AdditionalPtsEmbedding
        self.OverWriteFeatureNetInputWithGridSamplesOnly = OverWriteFeatureNetInputWithGridSamplesOnly
        self.UsePureVoxelDensity = UsePureVoxelDensity
        self.UseDeformationGridOverWriteMLP = UseDeformationGridOverWriteMLP
        
        if UseDeformationGridOverWriteMLP:
            print('Using UseDeformationGridOverWriteMLP')
        else:
            print('Not Using UseDeformationGridOverWriteMLP')

        # if loadfromckpt:
        #     deform_num_voxels = deform_num_voxels_base
        self.deform_num_voxels = deform_num_voxels
        self.deform_num_voxels_base = deform_num_voxels_base
        self.add_cam = add_cam
        self.voxel_dim = voxel_dim
        self.defor_depth = defor_depth
        self.net_width = net_width
        self.posbase_pe = posbase_pe
        self.viewbase_pe = viewbase_pe
        self.timebase_pe = timebase_pe
        self.gridbase_pe = gridbase_pe
        times_ch = 2*timebase_pe+1
        views_ch = 3+3*viewbase_pe*2
        pts_ch = 3+3*posbase_pe*2,
        
        print('init model xyz_min xyz_min')
        print(xyz_min)
        print(xyz_max)
        
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
        
        print('reading voxels')
        print(num_voxels)
        print(num_voxels_base)
        
        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1/(1-alpha_init) - 1)
        print('TiNeuVox: set density bias shift to', self.act_shift)

        timenet_width = net_width
        timenet_depth = 1
        timenet_output = voxel_dim+voxel_dim*2*gridbase_pe
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
        nn.Linear(timenet_width, timenet_output))
        if self.add_cam == True:
            views_ch = 3+3*viewbase_pe*2+timenet_output
            self.camnet = nn.Sequential(
            nn.Linear(times_ch, timenet_width), nn.ReLU(inplace=True),
            nn.Linear(timenet_width, timenet_output))
            print('TiNeuVox: camnet', self.camnet)

        featurenet_width = net_width
        featurenet_depth = 1
        grid_dim = voxel_dim*3+voxel_dim*3*2*gridbase_pe
        if self.OverWriteFeatureNetInputWithGridSamplesOnly:
            input_dim = grid_dim
        else:
            input_dim = grid_dim+timenet_output+0+0+3+3*posbase_pe*2
        self.featurenet = nn.Sequential(
            nn.Linear(input_dim, featurenet_width), nn.ReLU(inplace=True),
            *[
                nn.Sequential(nn.Linear(featurenet_width, featurenet_width), nn.ReLU(inplace=True))
                for _ in range(featurenet_depth-1)
            ],
            )
        self.featurenet_width = featurenet_width
        self._set_grid_resolution(num_voxels)
        self.deformationVeoxl_net = None
        self.deformation_net = None
        
        print('test reached')
        print(deform_num_voxels)
        print(deform_num_voxels_base)
        
        if self.UseDeformationGridOverWriteMLP:
            timeFeature = 0
            if UseDeformationGridTimeFeature:
                timeFeature = timenet_output
            self.deformationVeoxl_net = VoxelDeformation.DeformVoxGO(xyz_min,xyz_max,
                                                                     additionTimeEncoding = timeFeature,
                                                                     deform_num_voxels=deform_num_voxels, deform_num_voxels_base=deform_num_voxels_base)
            
        #else:
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=3+3*posbase_pe*2, input_ch_time=timenet_output)
        
        input_dim = featurenet_width
        self.densitynet = nn.Linear(input_dim, 1)

        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('grid_poc', torch.FloatTensor([(2**i) for i in range(gridbase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('view_poc', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))

        self.voxel_dim = voxel_dim
        self.feature= torch.nn.Parameter(torch.zeros([1, self.voxel_dim, *self.world_size],dtype=torch.float32))
        self.densityGrid = None
        
        if self.UsePureVoxelDensity:
            self.densityGrid = torch.nn.Parameter(torch.zeros([1, 1, *self.world_size],dtype=torch.float32 ))
        
        self.rgbnet = RGBNet(W=net_width, h_ch=featurenet_width, views_ch=views_ch, pts_ch=pts_ch, times_ch=times_ch)

        self.UseStyliziedRadianceField = True
        
        self.featureGridCopy = None
        self.featureNetCopy = None
        self.RGBNetCopy = None
        
        print('TiNeuVox: feature voxel grid', self.feature.shape)
        print('TiNeuVox: timenet mlp', self.timenet)
        # print('TiNeuVox: deformation_net mlp', self.deformation_net)
        print('TiNeuVox: densitynet mlp', self.densitynet)
        print('TiNeuVox: featurenet mlp', self.featurenet)
        print('TiNeuVox: rgbnet mlp', self.rgbnet)


    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('TiNeuVox: voxel_size      ', self.voxel_size)
        print('TiNeuVox: world_size      ', self.world_size)
        print('TiNeuVox: voxel_size_base ', self.voxel_size_base)
        print('TiNeuVox: voxel_size_ratio', self.voxel_size_ratio)


    def get_kwargs(self):
        if self.deformationVeoxl_net != None:
            deform_voxel = self.deformationVeoxl_net.num_voxels
        else:
            deform_voxel = 0
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'AdditionalPtsEmbedding' : self.AdditionalPtsEmbedding,
            'UseDeformationGridOverWriteMLP':self.UseDeformationGridOverWriteMLP,
            'OverWriteFeatureNetInputWithGridSamplesOnly' : self.OverWriteFeatureNetInputWithGridSamplesOnly,
            'num_voxels': self.num_voxels,
            'deform_num_voxels': deform_voxel,
            'num_voxels_base': self.num_voxels_base,
            'deform_num_voxels_base':deform_voxel,
            'alpha_init': self.alpha_init,
            'act_shift': self.act_shift,
            'voxel_size_ratio': self.voxel_size_ratio,
            'fast_color_thres': self.fast_color_thres,
            'voxel_dim':self.voxel_dim,
            'defor_depth':self.defor_depth,
            'net_width':self.net_width,
            'posbase_pe':self.posbase_pe,
            'viewbase_pe':self.viewbase_pe,
            'timebase_pe':self.timebase_pe,
            'gridbase_pe':self.gridbase_pe,
            'add_cam': self.add_cam,
        }


    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('TiNeuVox: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('num_voxels')
        print(num_voxels)
        if self.deformationVeoxl_net!= None:
            self.deformationVeoxl_net.scale_volume_grid( self.deformationVeoxl_net.num_voxels*2)
        print('TiNeuVox: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.feature = torch.nn.Parameter(
            F.interpolate(self.feature.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
        if self.densityGrid!=None:
            self.densityGrid = torch.nn.Parameter(
            F.interpolate(self.densityGrid.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
    
      
    def feature_total_variation_add_grad(self, weight, dense_mode):
        weight = weight * self.world_size.max() / 128
        total_variation_cuda.total_variation_add_grad(
            self.feature.float(), self.feature.grad.float(), weight, weight, weight, dense_mode)


    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True,debug=False):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        if debug:
            print(ind_norm)
        ret_lst = [
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1])
            for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst


    def mult_dist_interp(self, ray_pts_delta,overwrite = None):
        
        target = self.feature
        if overwrite!=None:
            target= overwrite
        
        x_pad = math.ceil((target.shape[2]-1)/4.0)*4-target.shape[2]+1
        y_pad = math.ceil((target.shape[3]-1)/4.0)*4-target.shape[3]+1
        z_pad = math.ceil((target.shape[4]-1)/4.0)*4-target.shape[4]+1
        grid = F.pad(target.float(),(0,z_pad,0,y_pad,0,x_pad))
        # three 
        if OverWriteMulFeatureGrid:
            vox_l = self.grid_sampler(ray_pts_delta, grid)
            vox_m = vox_l
            vox_s = vox_l
        else:
            vox_l = self.grid_sampler(ray_pts_delta, grid)
            vox_m = self.grid_sampler(ray_pts_delta, grid[:,:,::2,::2,::2])
            vox_s = self.grid_sampler(ray_pts_delta, grid[:,:,::4,::4,::4])
        vox_feature = torch.cat((vox_l,vox_m,vox_s),-1)

        if len(vox_feature.shape)==1:
            vox_feature_flatten = vox_feature.unsqueeze(0)
        else:
            vox_feature_flatten = vox_feature
        
        return vox_feature_flatten


    def activate_density(self, density, interval=None): 
        interval = interval if interval is not None else self.voxel_size_ratio 
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval) 


    def get_mask(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the geometry or not'''
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox]] = 1
        return hit.reshape(shape)


    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        
        stepdist = stepsize * self.voxel_size

        
        
        if UseDitheredRaySampling:
            step_diff =  torch.rand_like(rays_o[:,0]).unsqueeze(-1).repeat([1,3])
            rays_o_dithered = rays_o +  step_diff* rays_d
        else:
            rays_o_dithered = rays_o
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o_dithered, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id,mask_inbbox


    def sample_ray_new(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
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
        # if is_train:
        #     rng = rng.repeat(rays_d.shape[-2],1)
        #     rng += torch.rand_like(rng[:,[0]])
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[...,None] | ((self.xyz_min>rays_pts) | (rays_pts>self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox


    def getDeformation(self,ray_pts, times_sel,rays_pts_emb,times_feature,ray_id = ...):
        delta = self.getDeformation_delta(ray_pts, times_sel,rays_pts_emb,times_feature,ray_id)
        
        return delta + ray_pts


    def getDeformation_delta(self,ray_pts, times_sel,rays_pts_emb,times_feature,ray_id = ...):
        if self.UseDeformationGridOverWriteMLP:
            # mask = torch.ones_like(ray_pts[...,0]) < 0.0
            if UseDeformationGridTimeFeature:
               delta = self.deformationVeoxl_net.forward_simple(ray_pts, times_feature[ray_id]) 
            else:
                delta = self.deformationVeoxl_net.forward_simple(ray_pts, times_sel[ray_id]) 
            if MarkZeroTimeFeature :
                can_mask = times_sel[ray_id] < 0.001
                can_mask = can_mask.squeeze(-1)
                #can_mask = can_mask.unsqueeze(-2)
                #print(can_mask.shape)
                delta[can_mask] = 0.
            if EnableDeformationZero:      
                ray_pts = delta * 0.0 
            else:
                ray_pts = delta   
            return ray_pts
        
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id]) - ray_pts
        return ray_pts_delta




    def getDeformation_withElasticsLoss_old(self,ray_pts, times_sel,rays_pts_emb,times_feature,ray_id = ...):
        if self.UseDeformationGridOverWriteMLP:
            # mask = torch.ones_like(ray_pts[...,0]) < 0.0
            if UseDeformationGridTimeFeature:
               delta = self.deformationVeoxl_net.forward_simple(ray_pts, times_feature[ray_id]) 
            else:
                delta = self.deformationVeoxl_net.forward_simple(ray_pts, times_sel[ray_id]) 
            ray_pts = delta + ray_pts
        # TODO not correct
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id])
        ray_pts_delta_jac = self.deformation_net.getElastcJac(ray_pts,self.pos_poc , times_feature[ray_id])
        ray_pts_delta_Loss = self.deformation_net.getElasticLoss(ray_pts_delta_jac)
        
        
        times_emb_zeros = poc_fre(times_sel * 0.0, self.time_poc)
        times_feature_zeros = self.timenet(times_emb_zeros)
          
        ray_pts_delta_base = self.deformation_net(rays_pts_emb, times_feature_zeros[ray_id])
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id]) - ray_pts_delta_base
        
        return ray_pts_delta,ray_pts_delta_Loss

        
    def getDeformation_withElasticsLoss(self,ray_pts, times_sel,rays_pts_emb,times_feature,ray_id = ...):
        if self.UseDeformationGridOverWriteMLP:
            # mask = torch.ones_like(ray_pts[...,0]) < 0.0
            if UseDeformationGridTimeFeature:
               delta = self.deformationVeoxl_net.forward_simple(ray_pts, times_feature[ray_id]) 
            else:
                delta = self.deformationVeoxl_net.forward_simple(ray_pts, times_sel[ray_id]) 
            ray_pts = delta + ray_pts
        # TODO not correct
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id])
        ray_pts_delta_jac = self.deformation_net.getElastcJac(ray_pts,self.pos_poc , times_feature[ray_id])
        ray_pts_delta_Loss = self.deformation_net.getElasticLoss(ray_pts_delta_jac)
        
        
        times_emb_zeros = poc_fre(times_sel * 0.0, self.time_poc)
        times_feature_zeros = self.timenet(times_emb_zeros)
          
        ray_pts_delta_base = self.deformation_net(rays_pts_emb, times_feature_zeros[ray_id])
        ray_pts_delta = self.deformation_net(rays_pts_emb, times_feature[ray_id]) - ray_pts_delta_base
        
        return ray_pts_delta,ray_pts_delta_Loss
    
    
    def get_grid_worldcoords3(self,):
        grid_coord = torch.stack(torch.meshgrid(
            torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
            torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
            torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
        ), -1)

        return grid_coord
    
    
    def get_deform_grid(self, time_step):
        if self.densityGrid!=None:
            grid_coord = self.get_grid_worldcoords3()
        else:
            grid_coord = self.get_grid_worldcoords3()
        # grid_coord = grid_coord.reshape(1, -1, 3)
        # num_grid = grid_coord.shape[1]
        orishape = grid_coord.shape
        grid_coord = grid_coord.reshape([-1,3])
        timesstep = torch.ones(1, 1) * time_step

        timestep = time_step.reshape(1,-1).expand(grid_coord.shape[0],-1)  
        
        times_emb = poc_fre(timestep, self.time_poc)   
        times_feature = self.timenet(times_emb) 
        
        
        rays_pts_emb = poc_fre(grid_coord, self.pos_poc)
        
        newpos = self.getDeformation(grid_coord, timestep,rays_pts_emb,times_feature)
        
        global enableDelatemode2
        
        if not enableDelatemode2:
            grid_coord = newpos
        grid_coord = grid_coord.reshape(orishape)
        return grid_coord

    def get_deform_grid_by_pos(self,target_pos, time_step):
        # if self.densityGrid!=None:
        #     grid_coord = self.get_grid_worldcoords3()
        # else:
        grid_coord = target_pos
        # grid_coord = grid_coord.reshape(1, -1, 3)
        # num_grid = grid_coord.shape[1]
        orishape = grid_coord.shape
        grid_coord = grid_coord.reshape([-1,3])
        timesstep = torch.ones(1, 1) * time_step

        timestep = time_step.reshape(1,-1).expand(grid_coord.shape[0],-1)  
        
        times_emb = poc_fre(timestep, self.time_poc)   
        times_feature = self.timenet(times_emb) 
        
        
        rays_pts_emb = poc_fre(grid_coord, self.pos_poc)
        
        newpos = self.getDeformation(grid_coord, timestep,rays_pts_emb,times_feature)
        
        global enableDelatemode2
        
        if not enableDelatemode2:
            grid_coord = newpos
        grid_coord = grid_coord.reshape(orishape)
        return grid_coord

    def get_deform_alpha(self, time_step,interval):
        
        grid_coord = self.get_deform_grid(time_step)
        
        orishape = grid_coord.shape
        grid_coord = grid_coord.reshape([-1,3])
        
        vox_feature_flatten=self.mult_dist_interp(grid_coord)
        
        timestep = time_step.reshape(1,-1).expand(grid_coord.shape[0],-1)  
        
        times_emb = poc_fre(timestep, self.time_poc)   
        times_feature = self.timenet(times_emb) 

        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        rays_pts_emb = poc_fre(grid_coord, self.pos_poc)
        
        if self.OverWriteFeatureNetInputWithGridSamplesOnly:
            h_feature = self.featurenet(vox_feature_flatten_emb)
        else:
            h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        # h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        density_result = self.getDensity(h_feature,grid_coord)
        


        alpha = self.activate_density(density_result,interval)
        
        alpha = alpha.reshape(orishape[0],orishape[1],orishape[2],1)
        
        alpha = alpha.reshape([1, 1, *alpha.shape])

        return alpha
  

    def get_deform_alpha_byPos(self, target_pos, time_step,interval):
        
        grid_coord = self.get_deform_grid_by_pos(target_pos,time_step)
        
        orishape = grid_coord.shape
        grid_coord = grid_coord.reshape([-1,3])
        
        vox_feature_flatten=self.mult_dist_interp(grid_coord)
        
        timestep = time_step.reshape(1,-1).expand(grid_coord.shape[0],-1)  
        
        times_emb = poc_fre(timestep, self.time_poc)   
        times_feature = self.timenet(times_emb) 

        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        rays_pts_emb = poc_fre(grid_coord, self.pos_poc)
        
        if self.OverWriteFeatureNetInputWithGridSamplesOnly:
            h_feature = self.featurenet(vox_feature_flatten_emb)
        else:
            h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        # h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        density_result = self.getDensity(h_feature,grid_coord)
        


        alpha = self.activate_density(density_result,interval)
        
        alpha = alpha.reshape(orishape[0],orishape[1],orishape[2],1)
        
        alpha = alpha.reshape([1, 1, *alpha.shape])

        return alpha
  
    def getAlphByTime_withFixedPos(self,target_pos, time):
        stepsize = 0.5
        interval = stepsize * self.voxel_size_ratio
        alpha = self.get_deform_alpha_byPos(target_pos,time,interval)
        return alpha
    #TODO give thres
    @torch.no_grad()
    def compute_bbox_by_coarse_geo_deform(self, thres, times_one,stepsize,BBX_Time ):
        print('compute_bbox_by_coarse_geo: start')
        eps_time = time.time()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xyz_min = []
        xyz_max = []

        interval = stepsize * self.voxel_size_ratio
        if self.densityGrid!=None:
            dense_xyz = self.get_grid_worldcoords3()
        else:
            dense_xyz = self.get_grid_worldcoords3()
            
        with torch.no_grad():
            for i in range(0, times_one.shape[0], 1):
                ti = times_one[i]
                # if i ==27:
                #    continue
                alpha = self.get_deform_alpha(ti,interval)
                
                mask = (alpha.squeeze() > thres)
                active_xyz = dense_xyz[mask]
                if active_xyz.shape[0] > 0:
                    currentmin = active_xyz.amin(0)
                    currentmax = active_xyz.amax(0)
                    min_curtime,max_curtime = BBX_Time[i]
                    print(currentmin)
                    print(currentmax)
                    print( BBX_Time[i])
                    currentmin = torch.max(currentmin,min_curtime)
                    currentmax = torch.min(currentmax,max_curtime)
                    xyz_min.append(currentmin)
                    xyz_max.append(currentmax)
                    print('compute_bbox_by_coarse_geo: processed deform time ', ti,
                        ' ', currentmin, ' ', currentmax)

        xyz_min = torch.stack(xyz_min, dim=0)
        xyz_max = torch.stack(xyz_max, dim=0)
        xyz_min = xyz_min.amin(0)
        xyz_max = xyz_max.amax(0)

        print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
        print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
        eps_time = time.time() - eps_time
        print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
        
        # thre plus 1.1 or voxel * 0.5
        return xyz_min, xyz_max
    
    
    def getDensity(self,h_feature,ray_pts_delta):
        if not self.UsePureVoxelDensity:
            #print(h_feature.shape)
            res = self.densitynet(h_feature)
            #print(res.shape)
            return res
        else:
            #print(ray_pts_delta.shape)
            res = self.grid_sampler(ray_pts_delta, self.densityGrid,debug=False)  # .to(ray_pts_delta)
            res = res.unsqueeze(-1) 
            #print(res.shape)
            return res 


    def BFECCReg(self,times,DeltaTime_T):
        if times==0:
            return
        sample_points = self.get_grid_worldcoords3()
        # sample_points = self.TimeDeltaField.get_grid_worldcoords3()
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])
        #TODO sample_points ...3 

        
        timestep_ori = torch.tensor(times).cuda()
        
        targetSlice_base = 2048 * 2
        
        deltas_coll = []
        TotalNum = sample_points.shape[0]
        targetSlice = targetSlice_base
        
        for view_batch_start in range(0,TotalNum, targetSlice):
            UpperBound = min(TotalNum,view_batch_start + targetSlice)
            sample_points_slice = sample_points[view_batch_start:UpperBound,...]
            timestep = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)   
            timestep_prev = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)  - DeltaTime_T
            
            times_emb = poc_fre(timestep, self.time_poc)   
            
            times_emb_prev = poc_fre(timestep_prev, self.time_poc)  
            
            times_feature = self.timenet(times_emb) 
            times_feature_prev = self.timenet(times_emb_prev) 
            
            rays_pts_emb = poc_fre(sample_points_slice, self.pos_poc)

            # deform_t_m1 = self.getDeformation(sample_points_slice, timestep - DeltaTime_T,
            #                                                                                    rays_pts_emb,times_feature_prev
            #                                                                                    )
                
            deform = self.getDeformation(sample_points_slice, timestep,
                                                                                          rays_pts_emb,times_feature
                                                                                          )


            Defrom_delta_slice =  deform #- deform_t_m1
            # Defrom_delta_slice =  torch.ones_like(deform)
            deltas_coll.append(Defrom_delta_slice)
        
        Defrom_delta = torch.cat(deltas_coll) 

        sample_points_deformed = sample_points + Defrom_delta
        res_den = self.grid_sampler(sample_points_deformed, self.densityGrid,debug=False)# .to(ray_pts_delta)
        res = res_den.unsqueeze(-1)
        
        res = res.reshape([*oriShape[:3],1])
        res = res.unsqueeze(0)
        resGrid = res.permute(0,4,1,2,3)
        
        sample_points_deformed = sample_points - Defrom_delta 

        res_density = self.grid_sampler(sample_points_deformed, resGrid, debug=False)# .to(ray_pts_delta)
        
        oriDensity = self.densityGrid.reshape([-1])
        res_density = res_density.reshape([-1])
        res = (res_density -oriDensity).mean()
        
        # res2 = (res_density -oriDensity).sum()
        # print(res2.item())
        # print(oriDensity.sum().item())
        # print(res_density.sum().item())
        return res


    def NormLoss(self,times,DeltaTime_T):
        if times==0:
            return
        sample_points = self.get_grid_worldcoords3()
        # sample_points = self.TimeDeltaField.get_grid_worldcoords3()
        oriShape = sample_points.shape
        sample_points = sample_points.reshape([-1,3])
        #TODO sample_points ...3 

        
        timestep_ori = torch.tensor(times).cuda()
        
        targetSlice_base = 2048 * 2
        
        deltas_coll = []
        TotalNum = sample_points.shape[0]
        targetSlice = targetSlice_base
        
        for view_batch_start in range(0,TotalNum, targetSlice):
            UpperBound = min(TotalNum,view_batch_start + targetSlice)
            sample_points_slice = sample_points[view_batch_start:UpperBound,...]
            timestep = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)   
            timestep_prev = timestep_ori.reshape(1,-1).expand(sample_points_slice.shape[0],-1)  - DeltaTime_T
            
            times_emb = poc_fre(timestep, self.time_poc)   
            
            times_emb_prev = poc_fre(timestep_prev, self.time_poc)  
            
            times_feature = self.timenet(times_emb) 
            times_feature_prev = self.timenet(times_emb_prev) 
            
            rays_pts_emb = poc_fre(sample_points_slice, self.pos_poc)

            deform_t_m1 = self.getDeformation(sample_points_slice, timestep - DeltaTime_T,
                                                                                               rays_pts_emb,times_feature_prev
                                                                                               )
                
            deform = self.getDeformation(sample_points_slice, timestep,
                                                                                          rays_pts_emb,times_feature
                                                                                          )


            Defrom_delta_slice =  deform - deform_t_m1
            # Defrom_delta_slice =  torch.ones_like(deform)
            deltas_coll.append(Defrom_delta_slice)
        
        Defrom_delta = torch.cat(deltas_coll) 

        res = torch.abs(Defrom_delta).sum(dim=-1).mean()
        return res


    def getElasticLoss(self,jacobian):
        elastics_loss,residual = compute_elastic_loss(jacobian)
        return elastics_loss

    
    def getElastcJac(self,rays_pts, timestep, DeltaTimeT = 1.0/40.0,ray_id = ...):
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
            times_emb = poc_fre(timestep, self.time_poc)
            times_feature = self.timenet(times_emb)
            
            times_emb = poc_fre(timestep - DeltaTimeT, self.time_poc)
            times_feature_prev = self.timenet(times_emb)

        
        EnableAnotherElaMode = False
        def elafunc2(rays_pts_in,times_feature,times_feature_prev):
            
            deform = self.getDeformationByMLPPure(rays_pts_in, times_feature)
            deform_prev = self.getDeformationByMLPPure(rays_pts_in, times_feature_prev)
            return deform - deform_prev
        def elafunc(rays_pts_in,times_feature):
            
            deform = self.getDeformationByMLPPure(rays_pts_in, times_feature)
            
            return deform 
        if EnableAnotherElaMode:
            jacobian = vmap(jacfwd(elafunc2,argnums=0))(rays_pts, times_feature[ray_id],times_feature_prev[ray_id])
        else:
            jacobian = vmap(jacfwd(elafunc,argnums=0))(rays_pts,times_feature[ray_id])

        jacobian = jacobian.squeeze(1)
        return jacobian
         

    def getDeformationByMLPPure(self,rays_pts, time_feature):
        rays_pts_emb = poc_fre(rays_pts, self.pos_poc)

        rays_pts_emb = rays_pts_emb.unsqueeze(0)
        time_feature = time_feature.unsqueeze(0)
        
        ray_pts_delta = self.deformation_net(rays_pts_emb, time_feature)
        return ray_pts_delta

    def forward(self, rays_o, rays_d, viewdirs,times_sel_ori, cam_sel=None,bg_points_sel=None,global_step=None,
                calcuateEalstsicsloss = False,DeltaTime = 1.0/40.0,
                **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        if StaticVersion:
            times_sel = times_sel_ori * 0.0
        else:
            times_sel = times_sel_ori
        ret_dict = {}
        N = len(rays_o)
        times_emb = poc_fre(times_sel, self.time_poc)
        times_feature = self.timenet(times_emb)
        viewdirs_emb = poc_fre(viewdirs, self.view_poc)
        
        if self.add_cam==True:
            cam_emb= poc_fre(cam_sel, self.time_poc)
            cams_feature=self.camnet(cam_emb)
        # sample points on rays
        ray_pts, ray_id, step_id, mask_inbbox= self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio
        #if ray_id.dim()!=1:
        # print('000')
        # print(rays_o.shape)
        # print(ray_pts.shape)
        #     # return 
        
        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
        etlasticsloss = 0.0
        # if calcuateEalstsicsloss:
        #     ray_pts_delta,etlasticsloss = self.getDeformation_withElasticsLoss(ray_pts, times_sel,rays_pts_emb,times_feature,ray_id)
        # else:
        ray_pts_delta_offset = self.getDeformation_delta(ray_pts, times_sel,rays_pts_emb,times_feature,ray_id)
        
        delta_delta = 0.0
        
        # ray_pts_delta_offset_prev = self.getDeformation_delta(ray_pts, times_sel - DeltaTime,rays_pts_emb,times_feature,ray_id)    
        
        # delta_delta = ray_pts_delta_offset - ray_pts_delta_offset_prev
        
        ray_pts_delta = ray_pts + ray_pts_delta_offset
        
        # computer bg_points_delta
        # if bg_points_sel is not None:
            
        #     # TODO make correct

        #     bg_points_sel_emb = poc_fre(bg_points_sel, self.pos_poc)
        #     bg_points_sel_delta = self.getDeformation(bg_points_sel, times_sel[:(bg_points_sel.shape[0])],bg_points_sel_emb,times_feature[:(bg_points_sel.shape[0])],...)
        #     ret_dict.update({'bg_points_delta': bg_points_sel_delta})
        # voxel query interp
        if self.AdditionalPtsEmbedding:
            rays_pts_emb = poc_fre(ray_pts_delta, self.pos_poc)
        
        vox_feature_flatten=self.mult_dist_interp(ray_pts_delta)

        times_feature = times_feature[ray_id]
        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        if self.OverWriteFeatureNetInputWithGridSamplesOnly:
            h_feature = self.featurenet(vox_feature_flatten_emb)
        else:
            h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))
        
        density_result = self.getDensity(h_feature,ray_pts_delta)
        
        # alpha = nn.Softplus()(density_result+self.act_shift)
        alpha = self.activate_density(density_result,interval)
        # print(alpha.shape)
        if alpha.shape[0] != 1 or len(alpha.shape) != 1:
            alpha=alpha.squeeze(-1)

        if calcuateEalstsicsloss:
            mask = (alpha > self.fast_color_thres)
            if self.UseDeformationGridOverWriteMLP:
                jacs = self.deformationVeoxl_net.getElastcJac(ray_pts,times_sel[ray_id],None,mask,DeltaTimeT=DeltaTime)
                etlasticsloss = self.deformationVeoxl_net.getElasticLoss(jacs)
                # print(etlasticsloss)
            else:
                jacs = self.getElastcJac(ray_pts,times_sel,DeltaTimeT=DeltaTime,ray_id=ray_id)
                etlasticsloss = self.getElasticLoss(jacs)
                
                
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            h_feature=h_feature[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        
        
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            h_feature=h_feature[mask]

        viewdirs_emb_reshape = viewdirs_emb[ray_id]
        if self.add_cam == True:
            viewdirs_emb_reshape=torch.cat((viewdirs_emb_reshape, cams_feature[ray_id]), -1)
        rgb_logit = self.rgbnet(h_feature, viewdirs_emb_reshape)
        
        
        # rgb_logit = torch.clamp(rgb_logit, min=-100.0, max=5.0)
        rgb = torch.sigmoid( rgb_logit*  rgb_scale)
        # print('----')
        # print(rgb_logit[3000])
        # print(rgb[3000])
        
        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
            'elastics_loss':etlasticsloss,
            'deltas':ray_pts_delta_offset,
            'delta_delta':delta_delta
        })

        with torch.no_grad():
            depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N]),
                    reduce='sum')
        ret_dict.update({'depth': depth})
        return ret_dict


    def forward_stylizied(self, rays_o, rays_d, viewdirs,times_sel_ori, cam_sel=None,bg_points_sel=None,global_step=None,
                using_stylizedMode = False,deform_stylizied_models = None,
                 stylizied_deformed_Scale = 1.0, Enablek0Bypass = False,DeltaTime_T =1.0/40.0,renderStyImgaeWithoutGrad = False,
                  FrameNum = 2,  
                **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        if StaticVersion:
            times_sel = times_sel_ori * 0.0
        else:
            times_sel = times_sel_ori
        
        ret_dict = {}
        N = len(rays_o)

        
        viewdirs_emb = poc_fre(viewdirs, self.view_poc)
        if self.add_cam==True:
            cam_emb= poc_fre(cam_sel, self.time_poc)
            cams_feature=self.camnet(cam_emb)
        # sample points on rays
        ray_pts, ray_id, step_id, mask_inbbox= self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio


        if not deform_stylizied_models.EnableNewAdvectionScehems:
            #TODO add sth here
            # if using_stylizedMode:
            # if stylizied_deformed_Scale >= 0.01:
            if Enablek0Bypass:
                deform_stylized = deform_stylizied_models.StylizedDeformVol.forward_simple(ray_pts, k0overwrite = deform_stylizied_models.StylizedDeformVol_current_k0Bypass)
            else:
                deform_stylized = deform_stylizied_models.StylizedDeformVol.forward_simple(ray_pts)
            #TODO come back

            sty_Deform_delta = deform_stylized * stylizied_deformed_Scale 
            sty_Deform_delta.to(ray_pts)

            
            ray_pts = sty_Deform_delta + ray_pts 
        
        

        
        # pts deformation 
        

        rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
    
        times_emb = poc_fre(times_sel, self.time_poc)   
        times_feature = self.timenet(times_emb) 
        
        if deform_stylizied_models.EnableRefMode or FrameNum == 0 or deform_stylizied_models.DeltaMode2 :
            
            ray_pts_delta = self.getDeformation(ray_pts, times_sel,rays_pts_emb,times_feature,ray_id)  
            
           
            #print('simple hit')
        else:
            times_sel_new = times_sel - DeltaTime_T
            times_emb_new = poc_fre(times_sel_new, self.time_poc)   
            times_feature_new = self.timenet(times_emb_new)
            # print('unnormal simple hit')          
            ray_pts_delta = self.getDeformation(ray_pts, times_sel_new,rays_pts_emb,times_feature_new,ray_id)  
            # ray_pts_delta = ray_pts
        
        if (not deform_stylizied_models.EnableNewAdvectionScehems) or (not deform_stylizied_models.EnableNewAdvectionScehems_DeltaFieldInReferSpace) :
            timedelta = deform_stylizied_models.TimeDeltaField.forward_simple(ray_pts)
            ray_pts_delta = ray_pts_delta + timedelta
        
        # computer bg_points_delta
        if bg_points_sel is not None:
             
            bg_points_sel_emb = poc_fre(bg_points_sel, self.pos_poc)
            bg_points_sel_delta = self.getDeformation(bg_points_sel, times_sel[:(bg_points_sel.shape[0])],bg_points_sel_emb,times_feature[:(bg_points_sel.shape[0])],ray_id = ...)
            ret_dict.update({'bg_points_delta': bg_points_sel_delta})
        # voxel query interp
        
        TotalDelta = 0.0
        
        
        if deform_stylizied_models.EnableNewAdvectionScehems:
            
            if deform_stylizied_models.NewAdvectionSchemeesExtrapolateDeformationFields:
                
                timedelta = deform_stylizied_models.TimeDeltaField.forward_simple(ray_pts_delta)
                
                ray_pts_delta_altered = deform_stylizied_models.StylizedDeformVol.forward_simple(ray_pts, k0overwrite = deform_stylizied_models.NewAdvectionSchemeesExtrapolateDeformationFields_Cache)
                
                if Enablek0Bypass:
                    deform_stylized = deform_stylizied_models.StylizedDeformVol.forward_simple(ray_pts_delta_altered, k0overwrite = deform_stylizied_models.StylizedDeformVol_current_k0Bypass)
                else:
                    deform_stylized = deform_stylizied_models.StylizedDeformVol.forward_simple(ray_pts_delta_altered)
                #TODO come back

                sty_Deform_delta = deform_stylized * stylizied_deformed_Scale 
                sty_Deform_delta.to(ray_pts_delta)

                if deform_stylizied_models.EnableNewAdvectionScehems_enableAnotherSearch:
                    TotalDelta = sty_Deform_delta + timedelta
                else:
                    if deform_stylizied_models.EnableNewAdvectionScehems_DeltaFieldInReferSpace:
                        ray_pts_delta = ray_pts_delta + timedelta
                    ray_pts_delta = sty_Deform_delta + ray_pts_delta 
            else:
                timedelta = deform_stylizied_models.TimeDeltaField.forward_simple(ray_pts_delta)
                #TODO add sth here
                # if using_stylizedMode:
                # if stylizied_deformed_Scale >= 0.01:
                if Enablek0Bypass:
                    deform_stylized = deform_stylizied_models.StylizedDeformVol.forward_simple(ray_pts_delta, k0overwrite = deform_stylizied_models.StylizedDeformVol_current_k0Bypass)
                else:
                    deform_stylized = deform_stylizied_models.StylizedDeformVol.forward_simple(ray_pts_delta)
                #TODO come back

                sty_Deform_delta = deform_stylized * stylizied_deformed_Scale 
                sty_Deform_delta.to(ray_pts_delta)

                if deform_stylizied_models.EnableNewAdvectionScehems_enableAnotherSearch:
                    TotalDelta = sty_Deform_delta + timedelta
                else:
                    if deform_stylizied_models.EnableNewAdvectionScehems_DeltaFieldInReferSpace:
                        ray_pts_delta = ray_pts_delta + timedelta
                    ray_pts_delta = sty_Deform_delta + ray_pts_delta 

        
        if deform_stylizied_models.EnableNewAdvectionScehems_enableAnotherSearch:
            
            ray_pts = ray_pts + TotalDelta
            rays_pts_emb = poc_fre(ray_pts, self.pos_poc)
            if deform_stylizied_models.EnableRefMode or FrameNum == 0 or deform_stylizied_models.DeltaMode2 :
                
                ray_pts_delta = self.getDeformation(ray_pts, times_sel,rays_pts_emb,times_feature,ray_id)  
                #print('simple hit')
            else:
                times_sel_new = times_sel - DeltaTime_T
                times_emb_new = poc_fre(times_sel_new, self.time_poc)   
                times_feature_new = self.timenet(times_emb_new)
                # print('unnormal simple hit')          
                ray_pts_delta = self.getDeformation(ray_pts, times_sel_new,rays_pts_emb,times_feature_new,ray_id)  
                # ray_pts_delta = ray_pts
            
        
        
        if self.AdditionalPtsEmbedding:
            rays_pts_emb = poc_fre(ray_pts_delta, self.pos_poc)
        
        vox_feature_flatten=self.mult_dist_interp(ray_pts_delta)

        times_feature = times_feature[ray_id]
        vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
        
        if self.OverWriteFeatureNetInputWithGridSamplesOnly:
            h_feature = self.featurenet(vox_feature_flatten_emb)
        else:
            h_feature = self.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))

        density_result = self.getDensity(h_feature,ray_pts_delta)

        h_feature_copy = None
        
        adpatSty = deform_stylizied_models.EnableStyRayField and deform_stylizied_models.AdaptStyRaidanceFieldMode
        
        if deform_stylizied_models.EnableFeatureGridCopy_optimizeForColors and adpatSty:
        
            vox_feature_flatten=self.mult_dist_interp(ray_pts_delta,deform_stylizied_models.feature_copy)

            times_feature = times_feature[ray_id]
            vox_feature_flatten_emb = poc_fre(vox_feature_flatten, self.grid_poc)
            
            if self.OverWriteFeatureNetInputWithGridSamplesOnly:
                h_feature_copy = deform_stylizied_models.featurenet(vox_feature_flatten_emb)
            else:
                h_feature_copy = deform_stylizied_models.featurenet(torch.cat((vox_feature_flatten_emb, rays_pts_emb, times_feature), -1))

        # alpha = nn.Softplus()(density_result+self.act_shift)
        alpha = self.activate_density(density_result,interval)
        alpha=alpha.squeeze(-1)
        
        # viewdirs_emb_reshape_ori = viewdirs_emb[ray_id]
        ray_pts_ori = ray_pts_delta
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            alpha = alpha[mask]
            h_feature=h_feature[mask]
            if deform_stylizied_models.EnableFeatureGridCopy_optimizeForColors and adpatSty:
                h_feature_copy=h_feature_copy[mask]
            ray_pts_ori = ray_pts_ori[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            h_feature=h_feature[mask]
            if deform_stylizied_models.EnableFeatureGridCopy_optimizeForColors and adpatSty:
                h_feature_copy=h_feature_copy[mask]
            ray_pts_ori = ray_pts_ori[mask]

        viewdirs_emb_reshape = viewdirs_emb[ray_id]
        if self.add_cam == True:
            viewdirs_emb_reshape=torch.cat((viewdirs_emb_reshape, cams_feature[ray_id]), -1)




        
        rgb = 0.5
        # if deform_stylizied_models.setColorField and deform_stylizied_models.EnableStyRayField:
        #     if Enablek0Bypass:
        #         rgb = deform_stylizied_models.StylizedRadianceField(ray_pts_ori,
        #                                                     viewdirs_emb_reshape,k0bypass = deform_stylizied_models.StylizedDeformVol_current_k0Bypass_stylzied,
        #                                                     featurevecEmbedding = h_feature
        #                                                     )
                
        #     else:
        #         rgb = deform_stylizied_models.StylizedRadianceField(ray_pts_ori,
        #                                                     viewdirs_emb_reshape,
        #                                                     featurevecEmbedding = h_feature)
            
        # # else:
        if (not deform_stylizied_models.EnableFeatureGridCopy_optimizeForColors) or (not deform_stylizied_models.AdaptStyRaidanceFieldMode):
            h_feature_copy = h_feature

        
        if deform_stylizied_models.AdaptStyRaidanceFieldMode:
            rgb_logit = deform_stylizied_models.rgbnet(h_feature_copy, viewdirs_emb_reshape)
            rgb = torch.sigmoid(rgb_logit *  rgb_scale) # + (rgb - 0.5) *2.0
        else:
            rgb_logit = self.rgbnet(h_feature_copy, viewdirs_emb_reshape)
            rgb = torch.sigmoid(rgb_logit *  rgb_scale) # + (rgb - 0.5) *2.0       
        # Ray marching
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3]),
                reduce='sum')
        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'] *DataDefines.bgColor  )
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        with torch.no_grad():
            depth = segment_coo(
                    src=(weights * step_id),
                    index=ray_id,
                    out=torch.zeros([N]),
                    reduce='sum')
        ret_dict.update({'depth': depth})
        return ret_dict



class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y=False, flip_x=False ,flip_y=False, mode='center'):
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

def get_rays_of_a_view(H, W, K, c2w, ndc, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs

@torch.no_grad()
def get_training_rays(rgb_tr, times,train_poses, HW, Ks, ndc):
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
    times_tr = torch.ones([len(rgb_tr), H, W, 1], device=rgb_tr.device)

    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        times_tr[i] = times_tr[i]*times[i]
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, times,train_poses, HW, Ks, ndc):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    times_tr=torch.ones([N,1], device=DEVICE)
    times=times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        n = H * W
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n
    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, times,train_poses, HW, Ks, ndc, model, render_kwargs):
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
    times_tr = torch.ones([N,1], device=DEVICE)
    times = times.unsqueeze(-1)
    imsz = []
    top = 0
    for c2w, img, (H, W), K ,time_one in zip(train_poses, rgb_tr_ori, HW, Ks,times):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.get_mask(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        times_tr[top:top+n]=times_tr[top:top+n]*time_one
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
    return rgb_tr, times_tr,rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS

OverwriteEmbedding = False

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
