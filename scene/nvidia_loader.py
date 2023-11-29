import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import copy
import itertools
import numpy as np
import torch
import cv2
import imageio
from PIL import Image
import math
from tqdm import tqdm
from scene.utils import Camera
from typing import NamedTuple, List, Optional
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
# from scene.dataset_readers import 
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import copy


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float


def format_nvidia_info(dataset_class: Dataset):
    data_idx = len(dataset_class)
    cam_infos = []
    for uid, index in tqdm(enumerate(range(data_idx))):
        time = dataset_class.all_time[index]
        pose = dataset_class.poses[index]
        R = pose[:3,:3]
        R = -R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)

        FovY = focal2fov(dataset_class.focal, dataset_class.height)
        FovX = focal2fov(dataset_class.focal, dataset_class.width)

        image_path = "/".join(dataset_class.i_train_files[index].split("/")[:-1])
        image_name = dataset_class.i_train_files[index].split("/")[-1]
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None, mask=None,
                              image_path=image_path, image_name=image_name, width=int(dataset_class.width),
                              height=int(dataset_class.height), time=time,
                              )
        cam_infos.append(cam_info)

    return cam_infos


class LoadNvidiaData(Dataset):
    def __init__(self,
                 basedir: str,
                 ratio: float = 0.5,
                 final_height: int = 288,
                 start_frame: int = 0,
                 end_frame: int = 24,
                 split: str = 'train'
                 ) -> None:
        super().__init__()
        images, poses, bds, imgfiles = load_llff_data(basedir, start_frame, end_frame, ratio,
                                            final_height=final_height)
        # Define Camera Params
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        height, width, focal = hwf
        height, width = int(height), int(width)
        self.poses = poses
        self.height = height
        self.width = width
        self.focal = focal

        # Define time
        all_time = [i / (end_frame - 1) for i in range(start_frame, end_frame)]
        max_time = end_frame - 1
        self.max_time = max_time
        self.all_time = all_time

        # Define Training data
        images = np.transpose(images, (0, 3, 1, 2))
        self.i_train = images
        self.i_train_files = imgfiles

        # Define Test data
        times = range(0, 24)
        views = range(0, 12)
        self.i_test = [(t, v) for t in times for v in views if (t % 12 != v)]
        test_dir = os.path.join(basedir, 'mv_images')
        self.i_test_files = [os.path.join(test_dir, f'{time:05d}', f'cam{view + 1:02d}.jpg') for time, view in self.i_test]
        print(f'TEST DATA LEN {len(self.i_test)}')

        self.map_train = {}
        self.map_test = {}
        self.split = split

    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            return len(self.i_test)

    def __getitem__(self, index):
        if self.split == "train":
            return self.load_train(index)
        elif self.split == "test":
            return self.load_test(index)
        elif self.split == "video":
            return self.load_test(index)

    def load_train(self, idx):
        if idx in self.map_train.keys():
            return self.map_train[idx]
        
        image = self.i_train[idx]
        image = torch.Tensor(image)
        image = image.to(torch.float32)

        time = self.all_time[idx]

        pose = np.array(self.poses[idx % 12])
        R = pose[:3,:3]
        R = -R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)

        FovY = focal2fov(self.focal, self.height)
        FovX = focal2fov(self.focal, self.width)

        image_path = "/".join(self.i_train_files[idx].split("/")[:-1])
        image_name = self.i_train_files[idx].split("/")[-1]

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=self.width, height=self.height, time=time,
                              )
        self.map_train[idx] = caminfo
        return caminfo
    
    def load_test(self, idx):
        if idx in self.map_test.keys():
            return self.map_test[idx]

        image_path = self.i_test_files[idx]
        image = Image.open(image_path)
        image = PILtoTorch(image,(self.width, self.height)).to(torch.float32)
        assert self.i_train.shape[-2] == image.shape[-2]
        assert self.i_train.shape[-1] == image.shape[-1]

        time, view = self.i_test[idx]
        time /= self.max_time

        assert view >= 0 and view <= 11
        pose = np.array(self.poses[view])

        R = pose[:3, :3]
        R = -R
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)

        FovY = focal2fov(self.focal, self.height)
        FovX = focal2fov(self.focal, self.width)

        image_path = "/".join(self.i_test_files[idx].split("/")[:-1])
        image_name = self.i_test_files[idx].split("/")[-1]

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=self.width, height=self.height, time=time,
                              )
        self.map_test[idx] = caminfo
        return caminfo


class LoadNvidiaDVSData(Dataset):
    def __init__(self,
                 basedir: str,
                 ratio: float = 0.5,
                 final_height: Optional[int] = None,
                 start_frame: int = 0,
                 end_frame: int = 12,
                 split: str = 'train'
                 ) -> None:
        super().__init__()
        images, poses, bds, imgfiles, scale_factor = load_llff_data(basedir, start_frame, end_frame,
                                                      factor=int(1/ratio),
                                                      final_height=final_height)
        # Define Camera Params
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print(f'POSES SHAPE {poses.shape}')
        height, width, focal = hwf
        height, width = int(height), int(width)
        self.poses = poses
        self.height = height
        self.width = width
        self.focal = focal
        self.scale_factor = scale_factor

        # Define time
        all_time = [i / (end_frame - 1) for i in range(start_frame, end_frame)]
        max_time = end_frame - 1
        self.max_time = max_time
        self.all_time = all_time

        # Define Training data
        images = np.transpose(images, (0, 3, 1, 2))
        self.i_train = images
        self.i_train_files = imgfiles
        mask_images = range(0, 12)
        mask_dir = os.path.join(basedir, 'background_mask')
        mask_path = [os.path.join(mask_dir, f'{img:03d}.jpg.png') for img in mask_images]
        self.mask_path = mask_path
        assert len(mask_path) == len(self.i_train_files)

        # Define Test data
        times = range(start_frame, end_frame)
        self.i_test = list(times)
        test_dir = os.path.join(basedir, 'gt')
        self.i_test_files = [os.path.join(test_dir, f'v000_t{time:03d}.png') for time in self.i_test]
        print(f'TEST DATA LEN {len(self.i_test)}')

        self.map_train = {}
        self.map_test = {}
        self.split = split

    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            return len(self.i_test)

    def __getitem__(self, index):
        if self.split == "train":
            return self.load_train(index)
        elif self.split == "test":
            return self.load_test(index)
        elif self.split == "video":
            return self.load_test(index)

    def load_train(self, idx):
        if idx in self.map_train.keys():
            return self.map_train[idx]
        
        image = self.i_train[idx]
        image = torch.Tensor(image)
        image = image.to(torch.float32)

        mask = self.mask_path[idx]
        mask = Image.open(mask)
        mask = PILtoTorch(mask, (self.width, self.height))
        mask = torch.tensor(mask, requires_grad=True)
        assert self.i_train.shape[-2] == mask.shape[-2]
        assert self.i_train.shape[-1] == mask.shape[-1]

        time = self.all_time[idx]

        pose = np.array(self.poses[idx])
        R = pose[:3,:3]
        R = -R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)

        FovY = focal2fov(self.focal, self.height)
        FovX = focal2fov(self.focal, self.width)

        image_path = "/".join(self.i_train_files[idx].split("/")[:-1])
        image_name = self.i_train_files[idx].split("/")[-1]

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mask,
                              image_path=image_path, image_name=image_name, width=self.width, height=self.height, time=time,
                              )
        self.map_train[idx] = caminfo
        return caminfo
    
    def load_test(self, idx):
        if idx in self.map_test.keys():
            return self.map_test[idx]

        image_path = self.i_test_files[idx]
        image = Image.open(image_path)
        image = PILtoTorch(image, None).to(torch.float32)
        assert self.i_train.shape[-2] == image.shape[-2]
        assert self.i_train.shape[-1] == image.shape[-1]
        time = self.i_test[idx]
        time /= self.max_time

        pose = np.array(self.poses[0])

        R = pose[:3, :3]
        R = -R
        R[:, 0] = -R[:, 0]
        T = -pose[:3, 3].dot(R)

        FovY = focal2fov(self.focal, self.height)
        FovX = focal2fov(self.focal, self.width)

        image_path = "/".join(self.i_test_files[idx].split("/")[:-1])
        image_name = self.i_test_files[idx].split("/")[-1]

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=None,
                              image_path=image_path, image_name=image_name, width=self.width, height=self.height, time=time,
                              )
        self.map_test[idx] = caminfo
        return caminfo


def _load_data(basedir, start_frame, end_frame, 
               factor=None, width=None, height=None, 
               load_imgs=True, evaluation=False):
    print('factor ', factor)
    print(f'final height {height}')
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses_arr = poses_arr[start_frame:end_frame, ...]

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        # _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(round(sh[1] / factor))
        # width = int((sh[1] / factor))
        # _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
        print(f'GETTING THIS IMAGE {sfx}')
    elif width is not None:
        factor = sh[1] / float(width)
        width = int(round(sh[0] / factor))
        # _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgfiles = imgfiles[start_frame:end_frame]

    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), 
                                                                poses.shape[-1]) )
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    if evaluation:
        return poses, bds, imgs,
    
    print(imgs.shape)
    return poses, bds, imgs, imgfiles


def load_llff_data(basedir, start_frame, end_frame, 
                   factor=8, target_idx=10, 
                   recenter=True, bd_factor=.75, 
                   spherify=False, path_zflat=False, 
                   final_height=288):
    
    poses, bds, imgs, imgfiles = _load_data(basedir, 
                                  start_frame, end_frame,
                                  factor=factor,
                                  height=final_height,
                                  evaluation=False)
    print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], 
                            -poses[:, 0:1, :], 
                            poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(np.percentile(bds[:, 0], 5) * bd_factor)
    # sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)

    poses[:,:3,3] *= sc

    bds *= sc
    
    if recenter:
        poses = recenter_poses(poses)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, imgfiles, sc


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def render_wander_path(c2w):
    hwf = c2w[:,4:5]
    num_frames = 60
    max_disp = 48.0 # 64 , 48

    max_trans = max_disp / hwf[2][0] #self.targets['K_src'][0, 0, 0]  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0 #* 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)#[np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        # print('render_pose ', render_pose.shape)
        # sys.exit()
        output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
    
    return output_poses