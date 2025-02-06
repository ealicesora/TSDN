# import torch
# import os
# import time
# import math
# import functools
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from lib import utils
# from lib import tineuvox

# def total_variation(v, mask=None):
#     mask = None
#     tv2 = v.diff(dim=2, n=2).abs().mean(dim=1, keepdim=True)
#     tv3 = v.diff(dim=3, n=2).abs().mean(dim=1, keepdim=True)
#     tv4 = v.diff(dim=4, n=2).abs().mean(dim=1, keepdim=True)
#     if mask is not None:
#         maska = mask[:,:,:-1] & mask[:,:,1:]
#         tv2 = tv2[maska[:,:,:-1] & maska[:,:,1:]]
#         maskb = mask[:,:,:,:-1] & mask[:,:,:,1:]
#         tv3 = tv3[maskb[:,:,:,:-1] & maskb[:,:,:,1:]]
#         maskc = mask[:,:,:,:,:-1] & mask[:,:,:,:,1:]
#         tv4 = tv4[maskc[:,:,:,:,:-1] & maskc[:,:,:,:,1:]]
#     return (tv2.mean() + tv3.mean() + tv4.mean()) / 3

# class StyliziedPureVoxelGO(torch.nn.Module):
#     def __init__(self, xyz_min, xyz_max,
#                  deform_num_voxels=0, deform_num_voxels_base=0,
#                  nearest=False, pre_act_density=False, in_act_density=False,
#                   mask_cache_thres=1e-3,
#                  fast_deform_thres=0,
#                  deformnet_dim=0,
#                  deformnet_depth=3, deformnet_width=128, deformnet_output=4,
#                  posbase_pe=5, timebase_pe=5,use_time=True,
#                  **kwargs):
#         # deformnet_output = 4 overhere
        
#         super(StyliziedPureVoxelGO, self).__init__()

#         self.register_buffer('xyz_min', torch.Tensor(xyz_min))
#         self.register_buffer('xyz_max', torch.Tensor(xyz_max))

#         # determine based grid resolution
#         self.num_voxels_base = deform_num_voxels_base
#         self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)
#         self.use_time = use_time
#         # # determine the density bias shift
#         # self.alpha_init = alpha_init
#         # self.act_shift = np.log(1/(1-alpha_init) - 1)
#         # print('dvgo: set density bias shift to', self.act_shift)

#         # determine init grid resolution
#         # determin self.world_size here
#         self._set_grid_resolution(deform_num_voxels)

#         self.k0_dim = 3
#         # decode from tensor
#         self.k0 = torch.nn.Parameter(torch.randn([1, self.k0_dim, *self.world_size]))
        
        
#         # nn.init.constant_(self.deformnet[-1].bias, 0)
#         # self.deformnet[-1].weight.data *= 0.0

#         # Using the coarse geometry if provided (used to determine known free space and unknown space)

#         self.mask_cache = None
#         self.nonempty_mask = None

#     def k0_total_variation(self):
#         v = self.k0
#         return total_variation(v, self.nonempty_mask)

#     def _set_grid_resolution(self, num_voxels):
#         # Determine grid resolution
#         self.num_voxels = num_voxels
#         self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
#         self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
#         self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
#         # print('dvgo: voxel_size      ', self.voxel_size)
#         # print('dvgo: world_size      ', self.world_size)
#         # print('dvgo: voxel_size_base ', self.voxel_size_base)
#         # print('dvgo: voxel_size_ratio', self.voxel_size_ratio)
        
#     @torch.no_grad()
#     def scale_volume_grid(self, num_voxels):
#         print('dvgo: scale_volume_grid start')
#         ori_world_size = self.world_size
#         self._set_grid_resolution(num_voxels)
#         print('dvgo: scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)

#         self.occlusion = torch.nn.Parameter(
#             F.interpolate(self.occlusion.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
#         if self.k0_dim > 0:
#             self.k0 = torch.nn.Parameter(
#                 F.interpolate(self.k0.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))
#         else:
#             self.k0 = torch.nn.Parameter(torch.zeros([1, self.k0_dim, *self.world_size]))
#         if self.mask_cache is not None:
#             self._set_nonempty_mask()
#         print('dvgo: scale_volume_grid finish')

#     def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
#         '''Wrapper for the interp operation'''

#         mode = 'bilinear'
        
#         shape = xyz.shape[:-1]
#         xyz = xyz.reshape(1,1,1,-1,3)

#         ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
#         ret_lst = [
#             # TODO: use `rearrange' to make it readable
#             F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners).reshape(grid.shape[1],-1).T.reshape(*shape,grid.shape[1]).squeeze()
#             for grid in grids
#         ]
#         if len(ret_lst) == 1:
#             return ret_lst[0]
#         return ret_lst
    
#     # it is 3 in use
#     # grid sample point
#     def get_grid_worldcoords3(self,):
#         grid_coord = torch.stack(torch.meshgrid(
#             torch.linspace(self.xyz_min[0], self.xyz_max[0], self.world_size[0]),
#             torch.linspace(self.xyz_min[1], self.xyz_max[1], self.world_size[1]),
#             torch.linspace(self.xyz_min[2], self.xyz_max[2], self.world_size[2]),
#         ), -1)

#         return grid_coord
    
#     # what is this occlusion_mask?
#     # return occlusion_mask, deform, mask, time_dict
#     def forward(self, rays_pts,timestep, mask_outbbox,k0overwrite = None, **render_kwargs):
#         '''
#             give occlusion mask and deformation according to given positions
#         '''

#         # query for occlusion mask
#         occlusion_mask = torch.zeros_like(rays_pts[...,0])

#         mask = ~mask_outbbox
        
#         k0_view = torch.zeros(*occlusion_mask.shape, self.k0_dim).to(occlusion_mask)
        
#         if k0overwrite == None:
#             k0_view[mask] = self.grid_sampler(rays_pts[mask], self.k0).to(rays_pts)
#         else:
#             k0_view[mask] = self.grid_sampler(rays_pts[mask],k0overwrite).to(rays_pts)
        
#         return  k0_view, mask



# class Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth():
#     def __init__(
#         self,
#        DynamicRender,
#        StylizedDeformVol,
#        StylizedDeformVol_prev,
#        currentTimeDeltaField,
#         **kwargs
#     ):
#         super(Stylized_WithTimeDeformRender_advect_velocity_NSTSmooth_DeltaSmooth, self).__init__()
#         self.DynamicRender = DynamicRender
#         self.StylizedDeformVol = StylizedDeformVol
#         self.StylizedDeformVol_prev = StylizedDeformVol_prev
#         self.StylizedDeformVol_current_k0Bypass = None
#         self.TimeDeltaField = currentTimeDeltaField
#         self.EnableRefMode = False

#     def getRadianceField(self):
#         return self.DynamicRender.radiancegrid
    
#     def setRadianceField(self,radiancegrid):
#         self.DynamicRender.radiancegrid = radiancegrid
    
#     def clearGC(self):
#         del self.StylizedDeformVol
#         del self.StylizedDeformVol_prev
#         del self.StylizedDeformVol_current_k0Bypass
#         del self.TimeDeltaField

#     def prepare(self,timestep_ori, DeltaTime_T =1.0/40.0,OverWriteAdvection = False,alpha = 0.5,UseHalf = False,**_kwargs):
        
#         sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
#         if UseHalf:
#             sample_points = sample_points.half()
        
#         masks = torch.ones_like(sample_points) < 0.0
#         masks = masks.any(dim=-1)

        
#         timestep = timestep_ori.reshape(1,1,-1).expand(sample_points.shape[0], sample_points.shape[1],-1)  
            
#         # occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points, timestep, masks,UseHalf = UseHalf)

#         delta_deform, mask_d = self.TimeDeltaField(sample_points, timestep, masks,UseHalf = UseHalf)
        
    
#         sample_points_deformed = sample_points + delta_deform
        
#         if OverWriteAdvection:
#             sample_points_deformed = sample_points 
            
#         deform_stylized_prev, _ = self.StylizedDeformVol_prev(sample_points_deformed, timestep,masks)
#         # becoming slicing
#         if abs(timestep_ori - DeltaTime_T)<0.001:
#             deform_t_m1 = 0.0
#         deform_stylized_prev = deform_stylized_prev.unsqueeze(0)
#         deform_stylized_prev = deform_stylized_prev.permute(0,4,1,2,3)

#         test =  deform_stylized_prev * (1.0 - alpha) + self.StylizedDeformVol.k0 * ( alpha)
#         if OverWriteAdvection:
#             # import copy
#             # test = copy.deepcopy(self.StylizedDeformVol_prev.k0) 
#             test = self.StylizedDeformVol_prev.k0 + 0.0
            
#         self.StylizedDeformVol_current_k0Bypass = test
#         del self.StylizedDeformVol.k0
#         self.StylizedDeformVol.k0 = torch.nn.Parameter(test)
        

#     def Init_TimeDeltaField(self,timestep_ori, DeltaTime_T =1.0/40.0,UseHalf = False):
        
#         TestOverwrite = False
#         # with torch.no_grad():
#         sample_points = self.StylizedDeformVol_prev.get_grid_worldcoords3()
#         if UseHalf:
#             sample_points = sample_points.half()
        


        
#         deltas_coll = []
#         TotalNum = sample_points.shape[0]
#         targetSlice = 40
        
#         for view_batch_start in range(0,TotalNum, targetSlice):
#             UpperBound = min(TotalNum,view_batch_start + targetSlice)
#             sample_points_slice = sample_points[view_batch_start:UpperBound,...]
#             timestep = timestep_ori.reshape(1,1,-1).expand(sample_points_slice.shape[0], sample_points_slice.shape[1],-1)          
#             masks = torch.ones_like(sample_points_slice) < 0.0
#             masks = masks.any(dim=-1)

#             occlusion_mask, deform_t_m1, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points_slice, timestep - DeltaTime_T, masks,UseHalf = UseHalf)
                
#             occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(sample_points_slice, timestep, masks,UseHalf = UseHalf)

            
#             if (timestep_ori - DeltaTime_T)<0.001:
#                 print(' 1 hits!!!!')
#                 deform_t_m1 = 0.0
        

#             Defrom_delta_slice =  deform - deform_t_m1
#             deltas_coll.append(Defrom_delta_slice)
        
#         Defrom_delta = torch.cat(deltas_coll) 
#         # becoming slicing

#         Defrom_delta = Defrom_delta.unsqueeze(0)
#         Defrom_delta = Defrom_delta.permute(0,4,1,2,3)
#         del self.TimeDeltaField.k0
#         self.TimeDeltaField.k0 = torch.nn.Parameter(Defrom_delta)


            
#     # Post_StyDeform = True  more make sense
#     def forward(self, rays_o, rays_d, timestep, viewdirs, global_step=None,stylizied_deformed_Scale = 1.0,Enablek0Bypass = False,Post_StyDeform = True,DeltaTime_T =1.0/40.0,cal_deform_Approaching_loss = False, 
#                 FrameNum = 2, **render_kwargs):
#         '''Volume rendering'''
#         ret_dict = {}
#         printDtype = False
#         # sample points on rays
#         rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
#                 rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None,  **render_kwargs)
#         interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio
#         rays_pts = rays_pts.to(rays_o)

#         # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)

#         # if not Post_StyDeform:
#         #     occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)



#         #     can_mask = timestep == 0.
#         #     can_mask = can_mask.unsqueeze(-2)
#         #     can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
#         #     can_shape[-2] = deform.shape[-2]
#         #     can_mask = can_mask.expand(*can_shape).squeeze(-1)
#         #     deform[can_mask] = 0.       
#         #     rays_pts += deform
        
#         #TODO sample prev deform 
#         #TODO 考虑要不要用 
        
        
        

#         # advection

#         if stylizied_deformed_Scale >= 0.01:
#             if Enablek0Bypass:
#                 deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox,k0overwrite = self.StylizedDeformVol_current_k0Bypass, **render_kwargs)
#             else:
#                 deform_stylized, mask_d_stylizied = self.StylizedDeformVol(rays_pts,timestep, mask_outbbox, **render_kwargs)
#             #TODO come back
#             if printDtype:
#                 print(deform_stylized.dtype)
#             sty_Deform_delta = deform_stylized * stylizied_deformed_Scale
#             sty_Deform_delta.to(rays_pts)
#         else:
#             sty_Deform_delta = 0.0
            
#         rays_pts += sty_Deform_delta
        
#         can_mask = None
#         deform_Approaching_loss = 0.0
#         if self.EnableRefMode:
#             occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
#             occlusion_mask = occlusion_mask.to(rays_o)

#             rays_pts += deform * 1.0
#         else:
#         # TODO add deform
#             if FrameNum == 0:
#                 # print(timestep)
#                 occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
#                 occlusion_mask = occlusion_mask.to(rays_o)

#                 rays_pts += deform * 0.0
#             else:
         
#                 occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep - DeltaTime_T * 1.0, mask_outbbox, **render_kwargs)
#                 occlusion_mask = occlusion_mask.to(rays_o)



                
#                 # rays_pts += deform_stylized * 
                
#                 can_mask = timestep <= DeltaTime_T
                
#                 can_mask = can_mask.unsqueeze(-2)
#                 can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
#                 can_shape[-2] = deform.shape[-2]
#                 can_mask = can_mask.expand(*can_shape).squeeze(-1)
#                 # 把时间=0的部分的offset强制置于0
#                 deform[can_mask] = 0.
                
                
#                 delta_deform, mask_d = self.TimeDeltaField(rays_pts, timestep, mask_outbbox)
#                 deform += delta_deform 
                

#                 rays_pts += deform




#         # inference alpha, rgb
#         if self.DynamicRender.deformgrid.deformnet_output > 3:
#             occ_input = occlusion_mask
#         else:
#             occ_input = None
#         # occ_input = None
#         # if OverWritedisableOcclusionMask:
#         #     occ_input = None
#         # alphainv_cum used for bg 
#         alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
#             rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)
#         if printDtype:
#             print(rgb.dtype)

#         # Ray marching
#         rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
#         rgb_marched = rgb_marched.clamp(0, 1)
#         depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
#         depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']

#         disp = 1 / depth
#         ret_dict.update({
#             'alphainv_cum': alphainv_cum,
#             'weights': weights,
#             'rgb_marched': rgb_marched,
#             'raw_alpha': alpha,
#             'raw_rgb': rgb,
#             'depth': depth,
#             'disp': disp,
#             'mask': mask_r,
#             'deformation': deform,
#             'occlusion': occlusion_mask[mask_d],
#             'can_mask': can_mask,
#             'deform_loss':deform_Approaching_loss
#         })


#         return ret_dict


#     # Post_StyDeform = True  more make sense
#     def forward_ref(self, rays_o, rays_d, timestep, viewdirs, global_step=None,stylizied_deformed_Scale = 1.0,Enablek0Bypass = False,Post_StyDeform = True,DeltaTime_T =1.0/40.0,cal_deform_Approaching_loss = False, **render_kwargs):
#         '''Volume rendering'''
#         ret_dict = {}

#         # sample points on rays
#         rays_pts, mask_outbbox = self.DynamicRender.deformgrid.sample_ray(
#                 rays_o=rays_o, rays_d=rays_d, is_train=global_step is not None,  **render_kwargs)
#         interval = render_kwargs['stepsize'] * self.DynamicRender.deformgrid.voxel_size_ratio
#         rays_pts = rays_pts.to(rays_o)

#         # occlusion_mask is another radiacne grid, which accounts for shading (direct opt object to rgb loss)

#         # if not Post_StyDeform:
#         #     occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)



#         #     can_mask = timestep == 0.
#         #     can_mask = can_mask.unsqueeze(-2)
#         #     can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
#         #     can_shape[-2] = deform.shape[-2]
#         #     can_mask = can_mask.expand(*can_shape).squeeze(-1)
#         #     deform[can_mask] = 0.       
#         #     rays_pts += deform
        
#         #TODO sample prev deform 
#         #TODO 考虑要不要用 
        
        
        

#         # advection

        
#         # TODO add deform 
#         occlusion_mask, deform, mask_d, time_dict = self.DynamicRender.deformgrid(rays_pts, timestep, mask_outbbox, **render_kwargs)
        



#         deform_Approaching_loss = 0.0
#         # rays_pts += deform_stylized * 
        
#         can_mask = timestep == 0.0
#         can_mask = can_mask.unsqueeze(-2)
#         can_shape = -1 * np.ones(len(can_mask.shape), dtype=np.int64)
#         can_shape[-2] = deform.shape[-2]
#         can_mask = can_mask.expand(*can_shape).squeeze(-1)
#         # 把时间=0的部分的offset强制置于0
#         deform[can_mask] = 0.
        

#         rays_pts += deform




#         # inference alpha, rgb
#         if self.DynamicRender.deformgrid.deformnet_output > 3:
#             occ_input = occlusion_mask
#         else:
#             occ_input = None
#         # alphainv_cum used for bg 
#         alpha, alphainv_cum, rgb, weights, mask_r = self.DynamicRender.radiancegrid(
#             rays_pts, mask_outbbox, interval, viewdirs, occ_input, timestep, **render_kwargs)


#         # Ray marching
#         rgb_marched = (weights[...,None] * rgb).sum(-2) + alphainv_cum[...,[-1]] * render_kwargs['bg']
#         rgb_marched = rgb_marched.clamp(0, 1)
#         depth = (rays_o[...,None,:] - rays_pts).norm(dim=-1)
#         depth = (weights * depth).sum(-1) + alphainv_cum[...,-1] * render_kwargs['far']
#         disp = 1 / depth
#         ret_dict.update({
#             'alphainv_cum': alphainv_cum,
#             'weights': weights,
#             'rgb_marched': rgb_marched,
#             'raw_alpha': alpha,
#             'raw_rgb': rgb,
#             'depth': depth,
#             'disp': disp,
#             'mask': mask_r,
#             'deformation': deform,
#             'occlusion': occlusion_mask[mask_d],
#             'can_mask': can_mask,
#             'deform_loss':deform_Approaching_loss
#         })


#         return ret_dict

