import glob
import os
from tqdm import tqdm
import shutil
import json


def virtual_camera_generatte(number):
    camera_folder = "My_data/amc_camera_json/"
    camera_template = "My_original_data/camera_template.json"
    camrea_name = "c"+number
    camera_path = camera_folder + camrea_name + ".json"
    shutil.copyfile(camera_template, camera_path)

def split_edit(number):
    split_folder = "My_data/split"
    split_files = sorted(glob.glob(split_folder + "/*.json"))
    long2short_file = split_files[1]
    category_file = split_files[2]
    test_file = split_files[3]
    train_file = split_files[4]
    valid_file = split_files[5]

    number_key = str(number)
    number_tag = f"K_{number}_0"

    # 1. long2short 수정
    with open(long2short_file, 'r') as file:
        data = json.load(file)
    if number_key not in data:
        data[number_key] = [[0, -1]]
        with open(long2short_file, 'w') as file:
            json.dump(data, file)


    # 2. category 수정
    with open(category_file, 'r') as file:
        data = json.load(file)
    if number not in data.get("Korean", []):
        data["Korean"].append(number)
        with open(category_file, 'w') as file:
            json.dump(data, file, indent=4)


    # 3. test, train, valid 수정
    for i in range(3, 6):
        with open(split_files[i], 'r') as file:
            data = json.load(file)
        if number_tag not in data:
            data.append(number_tag)
            with open(split_files[i], 'w') as file:
                json.dump(data, file, indent=4)


    print(f"Split files for {number}th file updated successfully.")


def transfer_file(path_motion, path_music, to_motion_path, to_music_path):

    list_pos = sorted(glob.glob(path_motion + "/*.json"))
    list_music = sorted(glob.glob(path_music + "/*.mp3"))

    for motion in tqdm(list_pos):
        motion = motion.split("/")[-1]
        from_motion_name = os.path.join(path_motion, motion)
        to_motion_name = os.path.join(to_motion_path, motion.split("/")[-1])
        shutil.copyfile(from_motion_name, to_motion_name)


    for music in tqdm(list_music):
        folder_name = "amc" +  music.split("/")[-1].split(".")[0][1:]  
        folder_path = os.path.join(to_music_path, folder_name)
        if not os.path.exists(folder_path):
            #print(f"Creating directory: {folder_path}")
            os.makedirs(folder_path)
        else:
            print(f"Directory already exists: {folder_path}")
        from_music_name = os.path.join(path_music, music.split("/")[-1])
        to_music_name = os.path.join(folder_path, music.split("/")[-1].split(".")[0] + ".wav")
        shutil.copyfile(from_music_name, to_music_name)
        split_edit(int(music.split("/")[-1].split(".")[0][1:]))
        virtual_camera_generatte(music.split("/")[-1].split(".")[0][1:])

    
path_motion = "My_original_data/MOCAP/output/_json"
path_music = "My_original_data/MUSIC"
to_motion_path = "My_data/Simplified_MotionGlobalTransform"
to_music_path = "My_data/amc_raw_data"


transfer_file(path_motion, path_music, to_motion_path, to_music_path)



