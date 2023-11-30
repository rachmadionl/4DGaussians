#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from typing import Tuple

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


# def raw2outputs(
#   rgb: torch.Tensor,
#   opacity: torch.Tensor,
#   z_vals: torch.Tensor,
#   rays_d: torch.Tensor,
#   raw_noise_std: float = 0.0,
#   white_bkgd: bool = False
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#   r"""
#   Convert the raw NeRF output into RGB and other maps.
#   """

#   # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
#   dists = z_vals[..., 1:] - z_vals[..., :-1]
#   dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

#   # Multiply each distance by the norm of its corresponding direction ray
#   # to convert to real world distance (accounts for non-unit directions).
#   dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

#   # Add noise to model's predictions for density. Can be used to 
#   # regularize network during training (prevents floater artifacts).
#   noise = 0.
#   if raw_noise_std > 0.:
#     noise = torch.randn(raw[..., 3].shape) * raw_noise_std

#   # Predict density of each sample along each ray. Higher values imply
#   # higher likelihood of being absorbed at this point. [n_rays, n_samples]
#   alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

#   # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
#   # The higher the alpha, the lower subsequent weights are driven.
#   weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

#   # Compute weighted RGB map.
#   rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
#   rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

#   # Estimated depth map is predicted distance.
#   depth_map = torch.sum(weights * z_vals, dim=-1)

#   # Disparity map is inverse depth.
#   disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
#                             depth_map / torch.sum(weights, -1))

#   # Sum of weights along each ray. In [0, 1] up to numerical error.
#   acc_map = torch.sum(weights, dim=-1)

#   # To composite onto a white background, use the accumulated alpha map.
#   if white_bkgd:
#     rgb_map = rgb_map + (1. - acc_map[..., None])

#   return rgb_map, depth_map, acc_map, weights


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine"):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation
    means3D = pc.get_xyz
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    means2D = screenspace_points
    opacity = pc._opacity
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        # ORIGINAL CODE
        means3D_deform, scales_deform, rotations_deform, opacity_deform, rgb_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
                                                                                        rotations[deformation_point], opacity[deformation_point],
                                                                                        time[deformation_point])
        
        # EXPERIMENTAL CODE
            # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D, scales, 
            #                                                                                 rotations, opacity,
            #                                                                                 time)
    # print(time.max())

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    
    # EXPERIMENTAL CODE
    # means3D_final =  means3D_deform
    # rotations_final =  rotations_deform
    # scales_final =  scales_deform
    # opacity_final = opacity_deform

    # ORIGINAL CODE
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            if stage == 'fine':
                colors_precomp[deformation_point] = colors_precomp[deformation_point] + rgb_deform
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth,
            "deformation_point": deformation_point,
            "means3D": means3D,
            "means3D_deform": means3D_deform
            }

