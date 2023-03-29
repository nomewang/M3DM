import os
from shutil import copyfile
import cv2
import numpy as np
import tifffile
import yaml
import imageio.v3 as iio
import math
import argparse

# The same camera has been used for all the images
FOCAL_LENGTH = 711.11

def load_and_convert_depth(depth_img, info_depth):
    with open(info_depth) as f:
        data = yaml.safe_load(f)
    mind, maxd = data["normalization"]["min"], data["normalization"]["max"]

    dimg = iio.imread(depth_img)
    dimg = dimg.astype(np.float32)
    dimg = dimg / 65535.0 * (maxd - mind) + mind
    return dimg

def depth_to_pointcloud(depth_img, info_depth, pose_txt, focal_length):
    # input depth map (in meters) --- cfr previous section
    depth_mt = load_and_convert_depth(depth_img, info_depth)

    # input pose
    pose = np.loadtxt(pose_txt)

    # camera intrinsics
    height, width = depth_mt.shape[:2]
    intrinsics_4x4 = np.array([
        [focal_length, 0, width / 2, 0],
        [0, focal_length, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )

    # build the camera projection matrix
    camera_proj = intrinsics_4x4 @ pose

    # build the (u, v, 1, 1/depth) vectors (non optimized version)
    camera_vectors = np.zeros((width * height, 4))
    count=0
    for j in range(height):
        for i in range(width):
            camera_vectors[count, :] = np.array([i, j, 1, 1/depth_mt[j, i]])
            count += 1

    # invert and apply to each 4-vector
    hom_3d_pts= np.linalg.inv(camera_proj) @ camera_vectors.T
    # print(hom_3d_pts.shape)
    # remove the homogeneous coordinate
    pcd = depth_mt.reshape(-1, 1) * hom_3d_pts.T
    return pcd[:, :3]

def remove_point_cloud_background(pc):

    # The second dim is z
    dz =  pc[256,1] - pc[-256,1]
    dy =  pc[256,2] - pc[-256,2]

    norm =  math.sqrt(dz**2 + dy**2)
    start_points = np.array([0, pc[-256, 1], pc[-256, 2]])
    cos_theta = dy / norm
    sin_theta = dz / norm

    # Transform and rotation
    rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, -sin_theta],[0, sin_theta, cos_theta]])
    processed_pc = (rotation_matrix @ (pc - start_points).T).T

    # Remove background point
    for i in range(processed_pc.shape[0]):
        if processed_pc[i,1] > -0.02:
            processed_pc[i, :] = -start_points
        if processed_pc[i,2] > 1.8:
            processed_pc[i, :] = -start_points
        elif processed_pc[i,0] > 1 or processed_pc[i,0] < -1:
            processed_pc[i, :] = -start_points

    processed_pc = (rotation_matrix.T @ processed_pc.T).T + start_points

    index = [0, 2, 1]
    processed_pc = processed_pc[:,index]
    return processed_pc*[0.1, -0.1, 0.1]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path', default='datasets/eyecandies', type=str, help="Original Eyecandies dataset path.")
    parser.add_argument('--target_dir', default='datasets/eyecandies_preprocessed', type=str, help="Processed Eyecandies dataset path")
    args = parser.parse_args()
    
    os.mkdir(args.target_dir)
    categories_list = os.listdir(args.dataset_path)

    for category_dir in categories_list:
        category_root_path = os.path.join(args.dataset_path, category_dir)

        category_train_path = os.path.join(category_root_path, '/train/data')
        category_test_path = os.path.join(category_root_path, '/test_public/data')

        category_target_path = os.path.join(args.target_dir, category_dir)
        os.mkdir(category_target_path)

        os.mkdir(os.path.join(category_target_path, 'train'))
        category_target_train_good_path = os.path.join(category_target_path, 'train/good')
        category_target_train_good_rgb_path = os.path.join(category_target_train_good_path, 'rgb')
        category_target_train_good_xyz_path = os.path.join(category_target_train_good_path, 'xyz')
        os.mkdir(category_target_train_good_path)
        os.mkdir(category_target_train_good_rgb_path)
        os.mkdir(category_target_train_good_xyz_path)

        os.mkdir(os.path.join(category_target_path, 'test'))
        category_target_test_good_path = os.path.join(category_target_path, 'test/good')
        category_target_test_good_rgb_path = os.path.join(category_target_test_good_path, 'rgb')
        category_target_test_good_xyz_path = os.path.join(category_target_test_good_path, 'xyz')
        category_target_test_good_gt_path = os.path.join(category_target_test_good_path, 'gt')
        os.mkdir(category_target_test_good_path)
        os.mkdir(category_target_test_good_rgb_path)
        os.mkdir(category_target_test_good_xyz_path)
        os.mkdir(category_target_test_good_gt_path)
        category_target_test_bad_path = os.path.join(category_target_path, 'test/bad')
        category_target_test_bad_rgb_path = os.path.join(category_target_test_bad_path, 'rgb')
        category_target_test_bad_xyz_path = os.path.join(category_target_test_bad_path, 'xyz')
        category_target_test_bad_gt_path = os.path.join(category_target_test_bad_path, 'gt')
        os.mkdir(category_target_test_bad_path)
        os.mkdir(category_target_test_bad_rgb_path)
        os.mkdir(category_target_test_bad_xyz_path)
        os.mkdir(category_target_test_bad_gt_path)

        category_train_files = os.listdir(category_train_path)
        num_train_files = len(category_train_files)//17
        for i in range(0, num_train_files):
            pc = depth_to_pointcloud(
                    os.path.join(category_train_path,str(i).zfill(3)+'_depth.png'),
                    os.path.join(category_train_path,str(i).zfill(3)+'_info_depth.yaml'),
                    os.path.join(category_train_path,str(i).zfill(3)+'_pose.txt'),
                    FOCAL_LENGTH,
                )
            pc = remove_point_cloud_background(pc)
            pc = pc.reshape(512,512,3)
            tifffile.imwrite(os.path.join(category_target_train_good_xyz_path, str(i).zfill(3)+'.tiff'), pc)
            copyfile(os.path.join(category_train_path,str(i).zfill(3)+'_image_4.png'),os.path.join(category_target_train_good_rgb_path, str(i).zfill(3)+'.png'))
            
        
        category_test_files = os.listdir(category_test_path)
        num_test_files = len(category_test_files)//17
        for i in range(0, num_test_files):
            mask = cv2.imread(os.path.join(category_test_path,str(i).zfill(2)+'_mask.png'))
            if np.any(mask):
                pc = depth_to_pointcloud(
                    os.path.join(category_test_path,str(i).zfill(2)+'_depth.png'),
                    os.path.join(category_test_path,str(i).zfill(2)+'_info_depth.yaml'),
                    os.path.join(category_test_path,str(i).zfill(2)+'_pose.txt'),
                    FOCAL_LENGTH,
                    )
                pc = remove_point_cloud_background(pc)
                pc = pc.reshape(512,512,3)
                tifffile.imwrite(os.path.join(category_target_test_bad_xyz_path, str(i).zfill(3)+'.tiff'), pc)
                cv2.imwrite(os.path.join(category_target_test_bad_gt_path, str(i).zfill(3)+'.png'), mask)
                copyfile(os.path.join(category_test_path,str(i).zfill(2)+'_image_4.png'),os.path.join(category_target_test_bad_rgb_path, str(i).zfill(3)+'.png'))
            else:
                pc = depth_to_pointcloud(
                    os.path.join(category_test_path,str(i).zfill(2)+'_depth.png'),
                    os.path.join(category_test_path,str(i).zfill(2)+'_info_depth.yaml'),
                    os.path.join(category_test_path,str(i).zfill(2)+'_pose.txt'),
                    FOCAL_LENGTH,
                    )
                pc = remove_point_cloud_background(pc)
                pc = pc.reshape(512,512,3)
                tifffile.imwrite(os.path.join(category_target_test_good_xyz_path, str(i).zfill(3)+'.tiff'), pc)
                cv2.imwrite(os.path.join(category_target_test_good_gt_path, str(i).zfill(3)+'.png'), mask)
                copyfile(os.path.join(category_test_path,str(i).zfill(2)+'_image_4.png'),os.path.join(category_target_test_good_rgb_path, str(i).zfill(3)+'.png'))
