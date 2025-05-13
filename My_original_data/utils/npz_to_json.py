"""
2025.04.09
LKJ
"""

import os, glob
import numpy as np
import json
import csv
import json

# import pandas as pd

from tqdm import tqdm, trange
# from scipy.spatial.transform import Rotation as R


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


def generate_json(f_idx, data_pos):

    def json_transform(data_p):
        json_trans = []
        for data_p0 in data_p:
            add_trans = [
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                data_p0[0], data_p0[1], data_p0[2], 1.0,
            ]
            json_trans += add_trans
        return json_trans

    json_transform_frame = {
                "FrameTime": f_idx,
                "Transform": [
                    json_transform(data_pos)
                ]
            }
    return json_transform_frame


def generate_json_result(list_transform):
    data_json = {
        "BoneKeyFrameNumber": len(list_transform),
        "BoneKeyFrameTransformRecord": list_transform,
        #"BoneList": [
            #"下半身", "上半身", "上半身2", "首", "頭", "両目", "左目", "右目", "左足", "左ひざ", "左足首", "左つま先", "右足", "右ひざ", "右足首", "右つま先", "左肩P", "左肩", "左腕", "左ひじ", "左手首", "左腕捩", "左手捩", "左親指０", "左親指１", "左親指２", "左人指１", "左人指２", "左人指３", "左中指１", "左中指２", "左中指３", "左薬指１", "左薬指２", "左薬指３", "左小指１", "左小指２", "左小指３", "右肩P", "右肩", "右腕", "右ひじ", "右手首", "右腕捩", "右手捩", "右親指０", "右親指１", "右親指２", "右人指１", "右人指２", "右人指３", "右中指１", "右中指２", "右中指３", "右薬指１", "右薬指２", "右薬指３", "右小指１", "右小指２", "右小指３"
        #]
    }
    return data_json


def main():

    path_dir = "My_original_data/MOCAP/output/_npz60"
    list_dance = sorted(os.listdir(path_dir))

    path_dir2 = "My_original_data/MOCAP/output/_json"
    os.makedirs(path_dir2, exist_ok=True)

    for path_dance in tqdm(list_dance):
        list_pos = sorted(glob.glob(os.path.join(path_dir, path_dance, "*.npz")))

        #os.makedirs(os.path.join(path_dir2, path_dance), exist_ok=True)
        #save_path = os.path.join(path_dir2, path_dance)

        json_result = []
        for p_idx, path_pos in enumerate(list_pos):
            if p_idx % 4 != 0:
                continue
            data_dance = np.load(path_pos, allow_pickle=True)
            data_pos = data_dance["pos_xyz"]

            json_frame = generate_json(p_idx // 4, data_pos)
            json_result.append(json_frame)

        json_result2 = generate_json_result(json_result)

        name_json = path_dance + ".json"
        path_json = os.path.join(path_dir2, name_json)

        # JSON 파일로 저장
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(json_result2, f, indent=4)  # indent=4는 보기 좋게 정렬해줌


if __name__ == "__main__":
    print("=============== start ===============")

    main()

    print("=============== done ===============")