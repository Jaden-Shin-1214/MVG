"""
2025.04.09
LKJ
"""

import os, glob
import numpy as np
import json
import csv

import pandas as pd

from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R

import torch


# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")


file_path = os.path.abspath(os.path.dirname(__file__))


def obj_write(obj_path, data_ver, data_face=[], data_vcolor=None):
    dir_path = os.path.dirname(obj_path)
    os.makedirs(dir_path, exist_ok=True)
    print(f"saving file as {obj_path}")
    with open(obj_path, 'w') as obj_f:
        obj_f.write("# OBJ file\n")
        for v_idx, v in enumerate(data_ver):
            if data_vcolor is None:
                obj_f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                vc = data_vcolor[v_idx]
                obj_f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], vc[0], vc[1], vc[2]))
        for f in data_face:
            obj_f.write("f %s %s %s\n" % (int(f[0] + 1), int(f[1] + 1), int(f[2] + 1)))


# def main():
#     print()


# def load_files():
#     # path_dataset = "C:\personal\Dataset\DCM_data_update\Simplified_MotionGlobalTransform"
#     # list_data = sorted(glob.glob(os.path.join(path_dataset, "m.json")))

#     path_data = os.path.join("C:\personal\Dataset\DCM_data_update\Simplified_MotionGlobalTransform", "m0_GlobalTransform.json")

#     # JSON 파일 읽기
#     with open(path_data, "r", encoding="utf-8") as file:
#         data = json.load(file)  # JSON 데이터 로드

#     # "Transform" 데이터 확인
#     # 첫 번째 프레임의 Transform 데이터를 가져옴
#     data_len = len(data["BoneKeyFrameTransformRecord"])
#     for i in trange(data_len):
#         transform_data = data["BoneKeyFrameTransformRecord"][i]["Transform"]

#         v = np.array(transform_data)
#         v2 = v.reshape((-1, 4, 4))
#         v_xyz = v2[:, 3, :]

#         name_obj = f'm0_{str(i).zfill(4)}.obj'
#         path_obj_result = os.path.join(file_path, 'm0', name_obj)

#         obj_write(path_obj_result, v_xyz)

#     print()


# def load_pose():
#     path_dir = "C:\personal\Dataset\dance1"
#     list_dance = sorted(glob.glob(os.path.join(path_dir, "*.csv")))

#     for path_dance in list_dance:

#         data_csv = []
#         with open(path_dance, newline="", encoding="latin1") as csvfile:
#             # reader = csv.DictReader(csvfile)
#             reader = csv.reader(csvfile)
#             for r_idx, row in enumerate(reader):
#                 if r_idx <= 6:
#                     continue
#                 # data_csv.append(row)
#                 row = np.array(row)
#                 row = row[:]

#         data_np = np.loadtxt(path_dance, delimiter=",", skiprows=1, dtype=float)

#         print()


def load_pose_mdi(path_dir, out_dir):
    bones = [
        [0, 1], [1, 2], [2, 3], [1, 4], [1, 8],
        [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
        [0, 12], [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19]
    ]

    def load_mocap_file(mocap_path):
        # 'calibration/sample_motion.csv'
        mocap_data = np.genfromtxt(mocap_path, delimiter=',', skip_header=1, encoding='latin1')
        mocap_data = mocap_data[5:, 2:]  # Remove redundant data
        bone_data = mocap_data[:, :357].reshape(-1, 51, 7)
        # bone_coords = bone_data[:, :, 4:] / 1000.
        bone_coords = bone_data[:, :, 4:] / 100
        # interest_indices = [1, 2, 3, 4, 5, 6, 7, 8, 24, 25, 26, 27, 43, 44, 45, 46, 47, 48, 49, 50]  # 20
        interest_indices = [
            i for i in range(51)
        ]
        bone_coords = bone_coords[:, interest_indices]

        bone_rot = bone_data[:, :, :4]
        bone_rot = bone_rot[:, interest_indices]

        return bone_coords, bone_rot

    list_dance = sorted(glob.glob(os.path.join(path_dir, "*.csv")))

    # path_obj = path_dir + "_obj"
    # os.makedirs(path_obj, exist_ok=True)

    path_npz = out_dir + "_npz"
    os.makedirs(path_npz, exist_ok=True)

    for path_dance in tqdm(list_dance):
        pose_data, rot_data = load_mocap_file(path_dance)

        # path_obj_dir = os.path.join(path_obj, os.path.basename(path_dance).split(".csv")[0])
        # os.makedirs(path_obj_dir, exist_ok=True)

        path_npz_dir = os.path.join(path_npz, os.path.basename(path_dance).split(".csv")[0])
        os.makedirs(path_npz_dir, exist_ok=True)

        for p_idx in trange(len(pose_data)):
            pose_frame = pose_data[p_idx]
            rot_frame = rot_data[p_idx]

            path_result_name = f"frame_{str(p_idx).zfill(4)}.npz"
            path_result = os.path.join(path_npz_dir, path_result_name)
            np.savez(path_result, pos_xyz=pose_frame, rot_xyzw=rot_frame)

            #path_result_name = f"frame_{str(p_idx).zfill(4)}.obj"
            #path_result = os.path.join(path_obj_dir, path_result_name)
            # obj_write(path_result, pose_frame)


def adjust_joint(joint51_npz_path, joint60_npz_path):
    # def quaternion_to_matrix(q, t):
    #     """ Quaternion (w, x, y, z) + translation (x, y, z) -> 4x4 transformation matrix """
    #     rot_matrix = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # Convert quaternion to 3x3 rotation matrix
    #     transform_matrix = np.eye(4)
    #     transform_matrix[:3, :3] = rot_matrix
    #     transform_matrix[:3, 3] = t
    #     return transform_matrix

    # def transform_points(points, transform_matrix):
    #     """ Apply a 4x4 transform matrix to a set of 3D points """
    #     points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    #     transformed_points_h = transform_matrix @ points_h.T
    #     return transformed_points_h[:3].T  # Convert back to 3D

    # def optimize_567(joint51, joint567):
    #     param_scale = torch.ones(1, dtype=torch.float32, device=device, requires_grad=True)
    #     param_rx = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    #     param_ry = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    #     param_rz = torch.zeros(1, dtype=torch.float32, device=device, requires_grad=True)
    #     param_translation = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)

    #     e_lr = 0.005
    #     e_wd = 0.0001
    #     epoch_srt = 500 + 1

    #     optim_srt = torch.optim.Adam(
    #         [param_scale, param_rx, param_ry, param_rz, param_translation],
    #         lr=e_lr,
    #         weight_decay=e_wd
    #     )

    #     joint_src = np.zeros((4, 3))
    #     joint_src[0] = joint51[4]
    #     joint_src[1] = joint51[3]
    #     joint_src[2] = joint51[16]
    #     joint_src[3] = joint51[38]

    #     joint_trg = joint567[:4]

    #     joint_src = torch.from_numpy(joint_src).to(device)
    #     joint_trg = torch.from_numpy(joint_trg).to(device)

    #     with torch.no_grad():
    #         landmark_var_src = torch.mean(torch.std(joint_src, dim=0))
    #         landmark_var_trg = torch.mean(torch.std(joint_trg, dim=0))
    #         ratio_scale = landmark_var_trg / landmark_var_src

    #         landmark_mean_src = torch.mean(joint_src, dim=0) * ratio_scale
    #         landmark_mean_trg = torch.mean(joint_trg, dim=0)
    #         translation_offset = landmark_mean_trg - landmark_mean_src
    #         param_translation += (translation_offset / ratio_scale)

    #     for i in tqdm(range(epoch_srt)):
    #         optim_srt.zero_grad()

    #         param_rotation = torch.stack(
    #             [torch.stack(
    #                 [torch.cos(param_ry) * torch.cos(param_rz),
    #                  torch.sin(param_rx) * torch.sin(param_ry) * torch.cos(param_rz) + torch.cos(param_rx) * torch.sin(
    #                      param_rz),
    #                  -torch.cos(param_rx) * torch.sin(param_ry) * torch.cos(param_rz) + torch.sin(param_rx) * torch.sin(
    #                      param_rz)]),
    #                 torch.stack(
    #                     [- torch.cos(param_ry) * torch.sin(param_rz),
    #                      - torch.sin(param_rx) * torch.sin(param_ry) * torch.sin(param_rz) + torch.cos(
    #                          param_rx) * torch.cos(
    #                          param_rz),
    #                      torch.cos(param_rx) * torch.sin(param_ry) * torch.sin(param_rz) + torch.sin(
    #                          param_rx) * torch.cos(
    #                          param_rz)]),
    #                 torch.stack([torch.sin(param_ry), - torch.sin(param_rx) * torch.cos(param_ry),
    #                              torch.cos(param_rx) * torch.cos(param_ry)])]
    #         ).squeeze()

    #         landmark_src_srt = (ratio_scale * param_scale) * torch.matmul(joint_src, param_rotation) + (
    #                     ratio_scale * param_translation)

    #         loss_landmark = torch.mean(torch.sum((joint_trg - landmark_src_srt) ** 2, dim=1))

    #         if (i % 100) == 0:
    #             print(f"iteration # {i} loss : {loss_landmark}")
    #             # print(f"S: {param_scale} / T: {param_translation}")

    #         loss_landmark.backward()
    #         optim_srt.step()

    #     # SRT detach
    #     with torch.no_grad():
    #         result_scale = param_scale.detach().clone()
    #         result_rotation = torch.stack(
    #             [torch.stack(
    #                 [torch.cos(param_ry) * torch.cos(param_rz),
    #                  torch.sin(param_rx) * torch.sin(param_ry) * torch.cos(param_rz) + torch.cos(
    #                      param_rx) * torch.sin(
    #                      param_rz),
    #                  -torch.cos(param_rx) * torch.sin(param_ry) * torch.cos(param_rz) + torch.sin(
    #                      param_rx) * torch.sin(
    #                      param_rz)]),
    #                 torch.stack(
    #                     [- torch.cos(param_ry) * torch.sin(param_rz),
    #                      - torch.sin(param_rx) * torch.sin(param_ry) * torch.sin(param_rz) + torch.cos(
    #                          param_rx) * torch.cos(
    #                          param_rz),
    #                      torch.cos(param_rx) * torch.sin(param_ry) * torch.sin(param_rz) + torch.sin(
    #                          param_rx) * torch.cos(
    #                          param_rz)]),
    #                 torch.stack([torch.sin(param_ry), - torch.sin(param_rx) * torch.cos(param_ry),
    #                              torch.cos(param_rx) * torch.cos(param_ry)])]
    #         ).squeeze().detach().clone()
    #         result_translation = param_translation.detach().clone()

    #     # output extraction
    #     with torch.no_grad():
    #         v_output_srt = (ratio_scale * result_scale) * torch.matmul(joint_src[4:], result_rotation) + (
    #                     ratio_scale * result_translation)

    #     return v_output_srt

    # def add_j60_567():
    #     path_init = "C:/personal/Dataset/DCM_data_update/Simplified_MotionGlobalTransform/m0_GlobalTransform.json"
    #     with open(path_init, "r", encoding="utf-8") as file:
    #         data = json.load(file)  # JSON 데이터 로드
    #     transform_data = data["BoneKeyFrameTransformRecord"][0]["Transform"]
    #     v_init = np.array(transform_data)
    #     v_init = v_init.reshape((-1, 4, 4))
    #     v_init = v_init[:, 3, :3]
    #     v_init567 = v_init[5:7 + 1]

    #     # v_head = v_init[4]
    #     # v_neck = v_init[3]
    #     # v_sl = v_init[16]
    #     # v_sl = v_init[38]

    #     v_init_temp = np.zeros((7, 3))
    #     v_init_temp[0] = v_init[4]
    #     v_init_temp[1] = v_init[3]
    #     v_init_temp[2] = v_init[16]
    #     v_init_temp[3] = v_init[38]
    #     v_init_temp[4:] = v_init567

    #     return v_init_temp

    def joint_add(joint51):

        # init_j60_567()

        joint_result = np.zeros((60, 3))

        joint_idx = [0, 1, 2, 3, 4, 16, 18, 19, 20, 23,
                     24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                     34, 35, 36, 37, 38, 40, 41, 42, 45, 46,
                     47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
                     57, 58, 59, 8, 9, 10, 11, 12, 13, 14, 15
                     ]
        joint_idx2 = [17, 39]
        joint_idx2_src = [5, 24]

        joint_idx_temp = [5, 6, 7]
        joint_result[joint_idx_temp] = joint51[4]

        joint_result[joint_idx] = joint51
        joint_result[joint_idx2] = joint51[joint_idx2_src]

        joint_idx_avgS = [21, 22, 43, 44]
        joint_idx_avgT1 = [6, 7, 25, 26]
        joint_idx_avgT2 = [7, 8, 26, 27]
        joint_result[joint_idx_avgS] = (joint51[joint_idx_avgT1] + joint51[joint_idx_avgT2]) / 2

        return joint_result

    path_dir = joint51_npz_path
    list_dance = sorted(os.listdir(path_dir))

    path_dir2 = joint60_npz_path
    os.makedirs(path_dir2, exist_ok=True)

    # j567 = add_j60_567()  # 4 3 16 38 5 6 7

    for path_dance in tqdm(list_dance):
        list_pos = sorted(glob.glob(os.path.join(path_dir, path_dance, "*.npz")))

        os.makedirs(os.path.join(path_dir2, path_dance), exist_ok=True)

        for path_pos in list_pos:
            data_dance = np.load(path_pos, allow_pickle=True)
            data_pos = data_dance["pos_xyz"]
            data_rot = data_dance["rot_xyzw"]

            joint60 = joint_add(data_pos)

            name_joint60 = os.path.basename(path_pos)
            path_joint60 = os.path.join(path_dir2, path_dance, name_joint60)

            np.savez(path_joint60, pos_xyz=joint60)

            #path_joint60_obj = path_joint60.replace(".npz", ".obj")
            # obj_write(path_joint60_obj, joint60)


if __name__ == "__main__":
    print("=============== start ===============")

    # load_files()
    # load_pose()
    
    
    load_pose_mdi("My_original_data/MOCAP/input/", "My_original_data/MOCAP/output/")
    adjust_joint("My_original_data/MOCAP/output/_npz/", "My_original_data/MOCAP/output/_npz60")

    print("=============== done ===============")