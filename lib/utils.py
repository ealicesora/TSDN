import copy
import json
import math
import os
import pathlib
from typing import Any, Callable, List, Optional, Text, Tuple, Union

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .masked_adam import MaskedAdam

focalScale = 1.0
EnableRotationMatrixAlignRobustDynf = True


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any
Activation = Callable[[Array], Array]
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]
Normalizer = Callable[[], Callable[[Array], Array]]
PathType = Union[Text, pathlib.PurePosixPath]

from pathlib import PurePosixPath as GPath


def _compute_residual_and_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Auxiliary function of radial_and_tangential_undistort()."""

  r = x * x + y * y
  d = 1.0 + r * (k1 + r * (k2 + k3 * r))

  fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
  fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

  # Compute derivative of d over [x, y]
  d_r = (k1 + r * (2.0 * k2 + 3.0 * k3 * r))
  d_x = 2.0 * x * d_r
  d_y = 2.0 * y * d_r

  # Compute derivative of fx over x and y.
  fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
  fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

  # Compute derivative of fy over x and y.
  fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
  fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

  return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
    xd: np.ndarray,
    yd: np.ndarray,
    k1: float = 0,
    k2: float = 0,
    k3: float = 0,
    p1: float = 0,
    p2: float = 0,
    eps: float = 1e-9,
    max_iterations=10) -> Tuple[np.ndarray, np.ndarray]:
  """Computes undistorted (x, y) from (xd, yd)."""
  # Initialize from the distorted point.
  x = xd.copy()
  y = yd.copy()

  for _ in range(max_iterations):
    fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
        x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    denominator = fy_x * fx_y - fx_x * fy_y
    x_numerator = fx * fy_y - fy * fx_y
    y_numerator = fy * fx_x - fx * fy_x
    step_x = np.where(
        np.abs(denominator) > eps, x_numerator / denominator,
        np.zeros_like(denominator))
    step_y = np.where(
        np.abs(denominator) > eps, y_numerator / denominator,
        np.zeros_like(denominator))

    x = x + step_x
    y = y + step_y

  return x, y


class Camera:
  """Class to handle camera geometry."""

  def __init__(self,
               orientation: np.ndarray,
               position: np.ndarray,
               focal_length: Union[np.ndarray, float],
               principal_point: np.ndarray,
               image_size: np.ndarray,
               skew: Union[np.ndarray, float] = 0.0,
               pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
               radial_distortion: Optional[np.ndarray] = None,
               tangential_distortion: Optional[np.ndarray] = None,
               dtype=np.float32,
               isColMap = False
               ):
    """Constructor for camera class."""
    if radial_distortion is None:
      radial_distortion = np.array([0.0, 0.0, 0.0], dtype)
    if tangential_distortion is None:
      tangential_distortion = np.array([0.0, 0.0], dtype)

    if not isColMap:
        
        orientation = orientation.transpose(1,0)
        # print(orientation.shape)
        orientation_Sl1 = orientation[0:1,:]
        orientation_Sl2 = -orientation[1:2,:]
        orientation_Sl3 = orientation[2:3,:]
        orientation = np.concatenate([orientation_Sl2,orientation_Sl1,orientation_Sl3], 0) #Cw2 in OpenCV
        
        
        # print(orientation.shape)
        # orientation = orientation.transpose(1,0) #W2C in OpenCV
        
        
        
        # orientation = orientation.transpose(1,0)
        # # print(orientation.shape)
        # orientation_Sl1 = -orientation[0:1,:]
        # orientation_Sl2 = orientation[1:2,:]
        # orientation_Sl3 = -orientation[2:3,:]
        # # orientation = np.concatenate([orientation_Sl2,orientation_Sl1,orientation_Sl3], 0) #Cw2 in OpenCV
        
        # orientation = np.concatenate([orientation_Sl1,orientation_Sl2,orientation_Sl3], 0) #Cw2 in OpenCV
        # orientation = orientation.transpose(1,0) #W2C in OpenCV       


    self.orientation = np.array(orientation, dtype)
    self.position = np.array(position, dtype)
    self.focal_length = np.array(focal_length, dtype)
    self.principal_point = np.array(principal_point, dtype)
    self.skew = np.array(skew, dtype)
    self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, dtype)
    self.radial_distortion = np.array(radial_distortion, dtype)
    self.tangential_distortion = np.array(tangential_distortion, dtype)
    self.image_size = np.array(image_size, np.uint32)
    self.dtype = dtype

  @classmethod
  def from_json(cls, path: PathType,isColMap = False):
    """Loads a JSON camera into memory."""
    
    # self.isColMap = isColMap
    
    path = GPath(path)
    # with path.open('r') as fp:
    with open(path, 'r') as fp:
      camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
      camera_json['tangential_distortion'] = camera_json['tangential']

    return cls(
        orientation=np.asarray(camera_json['orientation']),
        position=np.asarray(camera_json['position']),
        focal_length=camera_json['focal_length'],
        principal_point=np.asarray(camera_json['principal_point']),
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.asarray(camera_json['radial_distortion']),
        tangential_distortion=np.asarray(camera_json['tangential_distortion']),
        image_size=np.asarray(camera_json['image_size']),
        isColMap = isColMap
    )

  def to_json(self):
    return {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in self.get_parameters().items()
    }

  def get_parameters(self):
    return {
        'orientation': self.orientation,
        'position': self.position,
        'focal_length': self.focal_length,
        'principal_point': self.principal_point,
        'skew': self.skew,
        'pixel_aspect_ratio': self.pixel_aspect_ratio,
        'radial_distortion': self.radial_distortion,
        'tangential_distortion': self.tangential_distortion,
        'image_size': self.image_size,
    }

  @property
  def scale_factor_x(self):
    return self.focal_length

  @property
  def scale_factor_y(self):
    # print('self.focal_length')  
    # print(self.focal_length)
    return self.focal_length * self.pixel_aspect_ratio

  @property
  def principal_point_x(self):
    return self.principal_point[0]

  @property
  def principal_point_y(self):
    return self.principal_point[1]

  @property
  def has_tangential_distortion(self):
    return any(self.tangential_distortion != 0.0)

  @property
  def has_radial_distortion(self):
    return any(self.radial_distortion != 0.0)

  @property
  def image_size_y(self):
    return self.image_size[1]

  @property
  def image_size_x(self):
    return self.image_size[0]

  @property
  def image_shape(self):
    return self.image_size_y, self.image_size_x

  @property
  def optical_axis(self):
    return self.orientation[2, :]

  @property
  def translation(self):
    return -np.matmul(self.orientation, self.position)

  def pixel_to_local_rays(self, pixels: np.ndarray):
    """Returns the local ray directions for the provided pixels."""
    # print(self.principal_point_x)
    # print(self.principal_point_y)
    y = ((pixels[..., 1] - self.principal_point_y) / self.scale_factor_y)
    x = ((pixels[..., 0] - self.principal_point_x - y * self.skew) /
         self.scale_factor_x)

    if self.has_radial_distortion or self.has_tangential_distortion:
      x, y = _radial_and_tangential_undistort(
          x,
          y,
          k1=self.radial_distortion[0],
          k2=self.radial_distortion[1],
          k3=self.radial_distortion[2],
          p1=self.tangential_distortion[0],
          p2=self.tangential_distortion[1])
    if EnableRotationMatrixAlignRobustDynf:
        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    else:
        dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

  def pixels_to_rays(self, pixels: np.ndarray) -> np.ndarray:
    """Returns the rays for the provided pixels.

    Args:
      pixels: [A1, ..., An, 2] tensor or np.array containing 2d pixel positions.

    Returns:
        An array containing the normalized ray directions in world coordinates.
    """
    if pixels.shape[-1] != 2:
      raise ValueError('The last dimension of pixels must be 2.')
    if pixels.dtype != self.dtype:
      raise ValueError(f'pixels dtype ({pixels.dtype!r}) must match camera '
                       f'dtype ({self.dtype!r})')

    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))

    local_rays_dir = self.pixel_to_local_rays(pixels)
    if EnableRotationMatrixAlignRobustDynf:
        # rays_dir = np.matmul(local_rays_dir[..., np.newaxis] ,self.orientation)
        # print(local_rays_dir[:, np.newaxis,:].shape)
        # print(self.orientation.shape)
        
        torchOri =  torch.tensor(np.transpose(self.orientation[:3, :3])[np.newaxis,:,:]).repeat(local_rays_dir.shape[0],1,1).cpu()
        torchDir = torch.tensor(local_rays_dir[:,np.newaxis, :]).cpu()
        rays_dir = torch.bmm(torchDir, torchOri).numpy()
        
    else:
        rays_dir = np.matmul(self.orientation.T, local_rays_dir[..., np.newaxis])
        rays_dir = np.squeeze(rays_dir, axis=-1)

    # Normalize rays.
    rays_dir /= np.linalg.norm(rays_dir, axis=-1, keepdims=True)
    rays_dir = rays_dir.reshape((*batch_shape, 3))
    return rays_dir

  def pixels_to_points(self, pixels: np.ndarray, depth: np.ndarray):
    rays_through_pixels = self.pixels_to_rays(pixels)
    cosa = np.matmul(rays_through_pixels, self.optical_axis)
    points = (
        rays_through_pixels * depth[..., np.newaxis] / cosa[..., np.newaxis] +
        self.position)
    return points

  def points_to_local_points(self, points: np.ndarray):
    translated_points = points - self.position
    local_points = (np.matmul(self.orientation, translated_points.T)).T
    return local_points

  def project(self, points: np.ndarray):
    """Projects a 3D point (x,y,z) to a pixel position (x,y)."""
    batch_shape = points.shape[:-1]
    points = points.reshape((-1, 3))
    local_points = self.points_to_local_points(points)

    # Get normalized local pixel positions.
    x = local_points[..., 0] / local_points[..., 2]
    y = local_points[..., 1] / local_points[..., 2]
    r2 = x**2 + y**2

    # Apply radial distortion.
    distortion = 1.0 + r2 * (
        self.radial_distortion[0] + r2 *
        (self.radial_distortion[1] + self.radial_distortion[2] * r2))

    # Apply tangential distortion.
    x_times_y = x * y
    x = (
        x * distortion + 2.0 * self.tangential_distortion[0] * x_times_y +
        self.tangential_distortion[1] * (r2 + 2.0 * x**2))
    y = (
        y * distortion + 2.0 * self.tangential_distortion[1] * x_times_y +
        self.tangential_distortion[0] * (r2 + 2.0 * y**2))

    # Map the distorted ray to the image plane and return the depth.
    pixel_x = self.focal_length * x + self.skew * y + self.principal_point_x
    pixel_y = (self.focal_length * self.pixel_aspect_ratio * y
               + self.principal_point_y)

    pixels = np.stack([pixel_x, pixel_y], axis=-1)
    return pixels.reshape((*batch_shape, 2))

  def get_pixel_centers(self):
    """Returns the pixel centers."""
    xx, yy = np.meshgrid(np.arange(self.image_size_x, dtype=self.dtype),
                         np.arange(self.image_size_y, dtype=self.dtype))
    return np.stack([xx, yy], axis=-1) + 0.5

  def scale(self, scale: float):
    """Scales the camera."""
    if scale <= 0:
      raise ValueError('scale needs to be positive.')

    new_camera = Camera(
        orientation=self.orientation.copy(),
        position=self.position.copy(),
        focal_length=self.focal_length * scale * focalScale,
        principal_point=self.principal_point.copy() * scale,
        skew=self.skew,
        pixel_aspect_ratio=self.pixel_aspect_ratio,
        radial_distortion=self.radial_distortion.copy(),
        tangential_distortion=self.tangential_distortion.copy(),
        image_size=np.array((int(np.round(self.image_size[0] * scale)),
                             int(np.round(self.image_size[1] * scale)))),
    )
    return new_camera

  def look_at(self, position, look_at, up, eps=1e-6):
    """Creates a copy of the camera which looks at a given point.

    Copies the provided vision_sfm camera and returns a new camera that is
    positioned at `camera_position` while looking at `look_at_position`.
    Camera intrinsics are copied by this method. A common value for the
    up_vector is (0, 1, 0).

    Args:
      position: A (3,) numpy array representing the position of the camera.
      look_at: A (3,) numpy array representing the location the camera
        looks at.
      up: A (3,) numpy array representing the up direction, whose
        projection is parallel to the y-axis of the image plane.
      eps: a small number to prevent divides by zero.

    Returns:
      A new camera that is copied from the original but is positioned and
        looks at the provided coordinates.

    Raises:
      ValueError: If the camera position and look at position are very close
        to each other or if the up-vector is parallel to the requested optical
        axis.
    """

    look_at_camera = self.copy()
    optical_axis = look_at - position
    norm = np.linalg.norm(optical_axis)
    if norm < eps:
      raise ValueError('The camera center and look at position are too close.')
    optical_axis /= norm

    right_vector = np.cross(optical_axis, up)
    norm = np.linalg.norm(right_vector)
    if norm < eps:
      raise ValueError('The up-vector is parallel to the optical axis.')
    right_vector /= norm

    # The three directions here are orthogonal to each other and form a right
    # handed coordinate system.
    camera_rotation = np.identity(3)
    camera_rotation[0, :] = right_vector
    camera_rotation[1, :] = np.cross(optical_axis, right_vector)
    camera_rotation[2, :] = optical_axis

    look_at_camera.position = position
    look_at_camera.orientation = camera_rotation
    return look_at_camera

  def crop_image_domain(
      self, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0):
    """Returns a copy of the camera with adjusted image bounds.

    Args:
      left: number of pixels by which to reduce (or augment, if negative) the
        image domain at the associated boundary.
      right: likewise.
      top: likewise.
      bottom: likewise.

    The crop parameters may not cause the camera image domain dimensions to
    become non-positive.

    Returns:
      A camera with adjusted image dimensions.  The focal length is unchanged,
      and the principal point is updated to preserve the original principal
      axis.
    """

    crop_left_top = np.array([left, top])
    crop_right_bottom = np.array([right, bottom])
    new_resolution = self.image_size - crop_left_top - crop_right_bottom
    new_principal_point = self.principal_point - crop_left_top
    if np.any(new_resolution <= 0):
      raise ValueError('Crop would result in non-positive image dimensions.')

    new_camera = self.copy()
    new_camera.image_size = np.array([int(new_resolution[0]),
                                      int(new_resolution[1])])
    new_camera.principal_point = np.array([new_principal_point[0],
                                           new_principal_point[1]])
    return new_camera

  def copy(self):
    return copy.deepcopy(self)


''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
            
    for k in cfg_train.keys():
        modelshad = getattr(model, 'deformationVeoxl_net')
        if modelshad is None:
            break
          
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model.deformationVeoxl_net, k):
            continue

        param = getattr(model.deformationVeoxl_net, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
                 
    return MaskedAdam(param_group)




def create_optimizer_for_styliziedDefromationGrid_transportBased(model, cfg_train, global_step,lrscale = 1.0,lrate_decay = -1.0,opt_radiance_filed = False,opt_firstField = True, opt_prevField = True):
    if lrate_decay < 0.0:  
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (global_step/decay_steps)
    else:
        decay_steps = lrate_decay * 1000
        decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    
    
    if opt_prevField:
        model.StylizedDeformVol_prev.train()
        model.StylizedDeformVol_prev.requires_grad_(True) 
        for k in cfg_train.keys():
            if (not hasattr(model, 'StylizedDeformVol_prev')) or (not opt_prevField):
                break
            if not k.startswith('lrate_deform_'):
                continue
            
            k = k[len('lrate_deform_'):]

            if not hasattr(model.StylizedDeformVol_prev, k):
                continue

            param = getattr(model.StylizedDeformVol_prev, k)
            if param is None:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol_prev: param lrate_deform_{k} not exist')
                continue

            lr = getattr(cfg_train, f'lrate_deform_{k}') * decay_factor * lrscale
            if lr > 0:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol_prev: param  lrate_deform_{k} lr {lr}')
                if isinstance(param, nn.Module):
                    param = param.parameters()
                param_group.append({'params': param, 'lr': lr})
            else:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol_prev: param lrate_deform_{k} freeze')
                param.requires_grad = False
    else:
        model.StylizedDeformVol_prev.eval()
        model.StylizedDeformVol_prev.requires_grad_(False)   
        
    if opt_firstField:
        model.StylizedDeformVol.train()
        model.StylizedDeformVol.requires_grad_(True) 
        for k in cfg_train.keys():
            if not k.startswith('lrate_deform_'):
                continue
            
            k = k[len('lrate_deform_'):]

            if not hasattr(model.StylizedDeformVol, k):
                continue

            param = getattr(model.StylizedDeformVol, k)
            if param is None:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol: param lrate_deform_{k} not exist')
                continue

            lr = getattr(cfg_train, f'lrate_deform_{k}') * decay_factor * lrscale
            if lr > 0:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol: param lrate_deform_{k} lr {lr}')
                if isinstance(param, nn.Module):
                    param = param.parameters()
                param_group.append({'params': param, 'lr': lr})
            else:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol: param lrate_deform_{k} freeze')
                param.requires_grad = False
    else:
        model.StylizedDeformVol.eval()
        model.StylizedDeformVol.requires_grad_(False)   
        

    
    if model.StylizedRadianceField != None:
      if opt_radiance_filed:
          model.StylizedRadianceField.train()
          model.StylizedRadianceField.requires_grad_(True)      
          for k in cfg_train.keys():
              if not k.startswith('lrate_'):
                  continue
              k = k[len('lrate_'):]

              if not hasattr(model.StylizedRadianceField, k):
                  continue

              param = getattr(model.StylizedRadianceField, k)
              if param is None:
                  print(f'create_optimizer_or_freeze_model  StylizedRadianceField: param {k} not exist')
                  continue

              lr = getattr(cfg_train, f'lrate_{k}') * decay_factor * lrscale
              if lr > 0:
                  if k=='density':
                      continue
                  print(f'create_optimizer_or_freeze_model  StylizedRadianceField: param {k} lr {lr}')
                  if isinstance(param, nn.Module):
                      param = param.parameters()
                  param_group.append({'params': param, 'lr': lr})
              else:
                  print(f'create_optimizer_or_freeze_model  StylizedRadianceField: param {k} freeze')
                  param.requires_grad = False             
    if opt_radiance_filed:
        model.DynamicRender.rgbnet.train()
        model.DynamicRender.rgbnet.requires_grad_(True)      
        for k in cfg_train.keys():
            if not k.startswith('Sty_lrate_rgbnet'):
                continue
            k = k[len('Sty_lrate_'):]

            if not hasattr(model.DynamicRender, k):
                continue

            param = getattr(model.DynamicRender, k)
            if param is None:
                print(f'create_optimizer_or_freeze_model radiancegrid: param {k} not exist')
                continue

            lr = getattr(cfg_train, f'Sty_lrate_{k}') * decay_factor * lrscale
            if lr > 0:
                print(f'create_optimizer_or_freeze_model radiancegrid: param {k} lr {lr}')
                if isinstance(param, nn.Module):
                    param = param.parameters()
                param_group.append({'params': param, 'lr': lr})
            else:
                print(f'create_optimizer_or_freeze_model radiancegrid: param {k} freeze')
                param.requires_grad = False
            
    
    return torch.optim.Adam(param_group, betas=(0.9,0.99))




def create_optimizer_for_styliziedDefromationGrid_transportBased_withAdditionalRadField(model, cfg_train, global_step,lrscale = 1.0,
                                                                                        lrate_decay = -1.0,opt_radiance_filed = False,opt_firstField = True, opt_prevField = True,
                                                                                        opt_radNet = True,opt_OriNeRFModelColors = False,coloroptScale = 1.0
                                                                                        ):
    if lrate_decay < 0.0:  
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (global_step/decay_steps)
    else:
        decay_steps = lrate_decay * 1000
        decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    
    
    if opt_prevField:
        model.StylizedDeformVol_prev.train()
        model.StylizedDeformVol_prev.requires_grad_(True) 
        for k in cfg_train.keys():
            if (not hasattr(model, 'StylizedDeformVol_prev')) or (not opt_prevField):
                break
            if not k.startswith('lrate_deform_'):
                continue
            
            k = k[len('lrate_deform_'):]

            if not hasattr(model.StylizedDeformVol_prev, k):
                continue

            param = getattr(model.StylizedDeformVol_prev, k)
            if param is None:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol_prev: param lrate_deform_{k} not exist')
                continue

            lr = getattr(cfg_train, f'lrate_deform_{k}') * decay_factor * lrscale
            if lr > 0:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol_prev: param lrate_deform_{k} lr {lr}')
                if isinstance(param, nn.Module):
                    param = param.parameters()
                param_group.append({'params': param, 'lr': lr})
            else:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol_prev: param lrate_deform_{k} freeze')
                param.requires_grad = False
    else:
        model.StylizedDeformVol_prev.eval()
        model.StylizedDeformVol_prev.requires_grad_(False)   
        
    if opt_firstField:
        model.StylizedDeformVol.train()
        model.StylizedDeformVol.requires_grad_(True) 
        for k in cfg_train.keys():
            if not k.startswith('lrate_deform_'):
                continue
            
            k = k[len('lrate_deform_'):]

            if not hasattr(model.StylizedDeformVol, k):
                continue

            param = getattr(model.StylizedDeformVol, k)
            if param is None:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol: param lrate_deform_{k} not exist')
                continue

            lr = getattr(cfg_train, f'lrate_deform_{k}') * decay_factor * lrscale
            if lr > 0:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol: param lrate_deform_{k} lr {lr}')
                if isinstance(param, nn.Module):
                    param = param.parameters()
                param_group.append({'params': param, 'lr': lr})
            else:
                print(f'create_optimizer_for_styliziedDefromationGrid StylizedDeformVol: param lrate_deform_{k} freeze')
                param.requires_grad = False
    else:
        model.StylizedDeformVol.eval()
        model.StylizedDeformVol.requires_grad_(False)   
        
    if opt_OriNeRFModelColors:
        opt_radiance_filed = False
    
    if opt_OriNeRFModelColors:
        #   model.StylizedRadianceField.train()
        #   model.StylizedRadianceField.requires_grad_(True)      
          for k in cfg_train.keys():
              if not k.startswith('Sty_lrate_Ori_'):
                  continue
              k = k[len('Sty_lrate_Ori_'):]

              if not hasattr(model.DynamicRender, k):
                  continue

              param = getattr(model.DynamicRender, k)
              if param is None:
                  print(f'create_optimizer_or_freeze_model  DynamicRender: param {k} not exist')
                  continue

              lr = getattr(cfg_train, f'Sty_lrate_Ori_{k}') * decay_factor * lrscale * coloroptScale
              if lr > 0:
                  if k=='density':
                      continue
                  # param.requires_grad = True
                  print(f'create_optimizer_or_freeze_model  DynamicRender: param {k} lr {lr}')
                  param.requires_grad_(True) 
                  if isinstance(param, nn.Module):
                      param = param.parameters()
                    
                  param_group.append({'params': param, 'lr': lr})
              else:
                  print(f'create_optimizer_or_freeze_model  DynamicRender: param {k} freeze')
                  param.requires_grad = False     
          for k in cfg_train.keys():
              if not k.startswith('Sty_lrate_Ori_Model_'):
                  continue
              k = k[len('Sty_lrate_Ori_Model_'):]

              if not hasattr(model, k):
                  continue

              param = getattr(model, k)
              if param is None:
                  print(f'create_optimizer_or_freeze_model  DynamicRender: param {k} not exist')
                  continue

              lr = getattr(cfg_train, f'Sty_lrate_Ori_Model_{k}') * decay_factor * lrscale * coloroptScale
              if lr > 0:
                  if k=='density':
                      continue
                  # param.requires_grad = True
                  param.requires_grad_(True)  
                  print(f'create_optimizer_or_freeze_model  DynamicRender: param {k} lr {lr}')
                  if isinstance(param, nn.Module):
                      param = param.parameters()
                   
                  param_group.append({'params': param, 'lr': lr})
              else:
                  print(f'create_optimizer_or_freeze_model  DynamicRender: param {k} freeze')
                  param.requires_grad = False     
    
    
    
    if model.StylizedRadianceField != None:
      if opt_radiance_filed:
          model.StylizedRadianceField.train()
          model.StylizedRadianceField.requires_grad_(True)      
          for k in cfg_train.keys():
              if not opt_radNet:
                if k.startswith('Sty_lrate_rgbnet'):
                  continue
              if not k.startswith('Sty_lrate_'):
                  continue
              k = k[len('Sty_lrate_'):]

              if not hasattr(model.StylizedRadianceField, k):
                  continue

              param = getattr(model.StylizedRadianceField, k)
              if param is None:
                  print(f'create_optimizer_or_freeze_model  StylizedRadianceField: param {k} not exist')
                  continue

              lr = getattr(cfg_train, f'Sty_lrate_{k}') * decay_factor * lrscale * coloroptScale
              if lr > 0:
                  if k=='density':
                      continue
                  print(f'create_optimizer_or_freeze_model  StylizedRadianceField: param {k} lr {lr}')
                  if isinstance(param, nn.Module):
                      param = param.parameters()
                  # param.requires_grad_(False)   
                  param_group.append({'params': param, 'lr': lr})
              else:
                  print(f'create_optimizer_or_freeze_model  StylizedRadianceField: param {k} freeze')
                  param.requires_grad = False     
                    
   #  print(param_group)
    opt = torch.optim.Adam(param_group, betas=(0.9,0.99))
    # print(opt)
    return opt

LerpScale = 1.0

def match_colors_for_image_set(image_set_ori, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """

    import copy
    style_img = copy.deepcopy(style_img)
    # print(image_set.shape)
    image_set = image_set_ori.squeeze(0)#.permute(1,2,0)
    #print(image_set.shape)
    style_img = style_img.squeeze(0)
    #print(image_set.shape)
    
    image_set = image_set.permute(1, 2, 0)
    style_img = style_img.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
    
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)

    
    if LerpScale != 1.0:
        image_set = image_set * LerpScale + (1.0 - image_set) * image_set_ori
    
    return image_set, color_tf



def match_colors_for_image_set_newShape(image_set, style_img):
    """
    image_set: [3,H, W ]
    style_img: [H, W, 3]
    """
    

    import copy
    style_img = copy.deepcopy(style_img)

    image_set = image_set.squeeze(0)#.permute(1,2,0)
    #print(image_set.shape)
    style_img = style_img.squeeze(0)
    
    image_set = image_set.permute(1, 2, 0)
    style_img = style_img.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
    
    sh = image_set.shape
    image_set = image_set.reshape(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)#.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)
    

    return image_set


def match_colors_for_image_set_dNeRF(image_set, style_img, image_set_target_ori):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    image_set_target [N,3]
    """

    import copy
    style_img = copy.deepcopy(style_img)
    # print(image_set.shape)
    
    #print(image_set.shape)
    
    #print(image_set.shape)
    image_set = image_set.squeeze(0)#.permute(1,2,0)
    image_set = image_set.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
   
    # image_set_target = image_set_target.squeeze(0)#.permute(1,2,0)
    # image_set_target = image_set_target.permute(1, 2, 0)
    # image_set_target = image_set_target.unsqueeze(0)
    
    style_img = style_img.squeeze(0)
    style_img = style_img.permute(1, 2, 0)
    
    
    sh = image_set.shape
    image_set_target = image_set_target_ori.clone()
    image_set_target = image_set_target.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set_target.device)

    mu_c = image_set_target.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set_target - mu_c).transpose(1, 0), image_set_target - mu_c) / float(image_set_target.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)

    return image_set, color_tf



def match_colors_for_image_set_newShape_dNeRF(image_set, style_img,image_set_target):
    """
    image_set: [3,H, W ]
    style_img: [H, W, 3]
    """
    

    import copy
    style_img = copy.deepcopy(style_img)

    image_set = image_set.squeeze(0)#.permute(1,2,0)
    #print(image_set.shape)
    style_img = style_img.squeeze(0)
    
    image_set = image_set.permute(1, 2, 0)
    style_img = style_img.permute(1, 2, 0)
    image_set = image_set.unsqueeze(0)
    
    sh = image_set.shape
    image_set_target = image_set_target.reshape(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set_target.device)

    mu_c = image_set_target.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set_target - mu_c).transpose(1, 0), image_set_target - mu_c) / float(image_set_target.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    
    image_set = image_set.squeeze(0)
    image_set = image_set.permute(2, 0, 1)#.permute(2, 0, 1)
    image_set = image_set.unsqueeze(0)
    

    return image_set


def apply_CT(image_set,colorTransfer):
    tmp_mat = colorTransfer[:3, :3]
    tmp_vec = colorTransfer[:3, 3:4] 
    
    image_set = image_set.squeeze(0).permute(1,2,0)
    
    sh = image_set.shape
    
    image_set = image_set.view(-1, 3)
    
    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().view(sh)
    image_set = image_set.permute(2,0,1).unsqueeze(0)

    return image_set


def create_optimizer_for_deltaSmooth(model, cfg_train, global_step,lrscale = 1.0,lrate_decay = -1.0,opt_radiance_filed = False,opt_firstField = True, opt_prevField = True):
    if lrate_decay < 0.0:  
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (global_step/decay_steps)
    else:
        decay_steps = lrate_decay * 1000
        decay_factor = 0.1 ** (global_step/decay_steps)


    param_group = []

    for k in cfg_train.keys():
        if not k.startswith('lrate_sty_'):
            continue
        
        k = k[len('lrate_sty_'):]

        if not hasattr(model.TimeDeltaField, k):
            continue

        param = getattr(model.TimeDeltaField, k)
        if param is None:
            print(f'create_optimizer_for_styliziedDefromationGrid TimeDeltaField: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_sty_{k}') * decay_factor * lrscale
        if lr > 0:
            print(f'create_optimizer_for_styliziedDefromationGrid TimeDeltaField: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr})
        else:
            print(f'create_optimizer_for_styliziedDefromationGrid TimeDeltaField: param {k} freeze')
            param.requires_grad = False
            

            
    
    return torch.optim.Adam(param_group, betas=(0.9,0.99))



''' Checkpoint utils
'''




def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'] ,loadfromckpt = True)
    model.load_state_dict(ckpt['model_state_dict'])
    return model


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

