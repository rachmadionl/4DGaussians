import os
import glob
import argparse
import numpy as np
from plyfile import PlyData, PlyElement
from colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary, read_points3D_binary_static


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def generate_points(realdir):

    # camerasfile = os.path.join(realdir, 'sparse_filtered/0/cameras.bin')
    # camdata = read_cameras_binary(camerasfile)

    # list_of_keys = list(camdata.keys())
    # cam = camdata[list_of_keys[0]]
    # print( 'Cameras', len(cam))

    # h, w, f = cam.height, cam.width, cam.params[0]
    # # w, h, f = factor * w, factor * h, factor * f
    # hwf = np.array([h,w,f]).reshape([3,1])

    # imagesfile = os.path.join(realdir, 'sparse_filtered/0/images.bin')
    # imdata = read_images_binary(imagesfile)

    # w2c_mats = []
    # bottom = np.array([0,0,0,1.]).reshape([1,4])

    # names = [imdata[k].name for k in imdata]
    # img_keys = [k for k in imdata]

    # print('Images #', len(names))
    # perm = np.argsort(names)

    points3dfile = os.path.join(realdir, 'sparse_filtered/0/points3D.bin')
    xyz, rgb, _ = read_points3D_binary_static(points3dfile)
    ply_path = os.path.join(realdir, 'points.npy')
    storePly(ply_path, xyz, rgb)

    # bounds_mats = []

    # for i in perm[0:len(img_keys)]:

    #     im = imdata[img_keys[i]]
    #     print(im.name)
    #     R = im.qvec2rotmat()
    #     t = im.tvec.reshape([3,1])
    #     m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
    #     w2c_mats.append(m)

    #     pts_3d_idx = im.point3D_ids
    #     pts_3d_vis_idx = pts_3d_idx[pts_3d_idx >= 0]

    #     #
    #     depth_list = []
    #     for k in range(len(pts_3d_vis_idx)):
    #       point_info = pts3d[pts_3d_vis_idx[k]]

    #       P_g = point_info.xyz
    #       P_c = np.dot(R, P_g.reshape(3, 1)) + t.reshape(3, 1)
    #       depth_list.append(P_c[2])

    #     zs = np.array(depth_list)
    #     close_depth, inf_depth = np.percentile(zs, 5), np.percentile(zs, 95)
    #     bounds = np.array([close_depth, inf_depth])
    #     bounds_mats.append(bounds)

    # w2c_mats = np.stack(w2c_mats, 0)
    # c2w_mats = np.linalg.inv(w2c_mats)

    # poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    # poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis],
    #                                     [1,1,poses.shape[-1]])], 1)

    # # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    # poses = np.concatenate([poses[:, 1:2, :],
    #                         poses[:, 0:1, :],
    #                        -poses[:, 2:3, :],
    #                         poses[:, 3:4, :],
    #                         poses[:, 4:5, :]], 1)

    # save_arr = []

    # for i in range((poses.shape[2])):
    #     save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mats[i]], 0))

    # save_arr = np.array(save_arr)
    # print(save_arr.shape)

    # # Use all frames to calculate COLMAP camera poses.
    # if os.path.isdir(os.path.join(realdir, 'images_colmap')):
    #     num_colmap_frames = len(glob.glob(os.path.join(realdir, 'images_colmap', '*.jpg')))
    #     num_data_frames = len(glob.glob(os.path.join(realdir, 'images', '*.png')))

    #     assert num_colmap_frames == save_arr.shape[0]
    #     np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr[:num_data_frames, :])
    # else:
    #     np.save(os.path.join(realdir, 'poses_bounds.npy'), save_arr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
                        help='Dataset path')

    args = parser.parse_args()

    generate_points(args.dataset_path)