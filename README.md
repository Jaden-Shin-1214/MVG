# MVG
Music Video Generator with MOCAP data and DanceCamAnimator Code
![Image](https://github.com/user-attachments/assets/630b49f8-93c2-4020-896b-ae51478f6652)
![Image](https://github.com/user-attachments/assets/0e03fe33-0014-4b43-aa50-42f610905a65)

## Preparation

1. Download camera generation model ckpt. from [https://github.com/Carmenw1203/DanceCamAnimator-Official] "DCA_stage1.pt" and "DCA_state_2n3.pt" and move to /checkpoints
2. change your Optirack Motive motion capture file (.csv) name into m*_GlobalTransform.csv
3. Also, change yout music file name into a*.mp3 (Both music and csv file should have frames smaller than 1050)

4. Move Motion Data and music data into 'My_original_data/MOCAP/input' and 'My_original_data/MUSIC' (you don't have to use /DCA_data in generation)

5. Move 'modify_motion_json.py' to 'scripts/my_utils/'
6. IF you want to control minimum camrea keyframe distance, then change 'scripts/MusicDance2Keyframe.py' to given one

7. Move My_data.sh to 'sh/' folder

## Requirements
Using dockerfile, implementation under codes

```.bash
sudo docker build -t dca .
sudo docker run --shm-size=8G --gpus all --name dca -it -v /home/syj/DanceCamAnimator-Official:/workspace dca 
sudo docker exec -it dca /bin/bash
```
Then, generate virtual environment and install requiremtents.txt file

## Quick Start 

Run 
```.bash
bash My_data.sh
```
Then MMD Camera Data will downloaded to 'My_output/test_DCA/etest/extend_vmd/'  


## Setting Music Videos
1. Background

you can use MMD background which compatiable in Blender. We used 'https://www.deviantart.com/birdaaa/art/MMD-x-Blender-The-First-Sound-Stage-Download-913660539' (thanks for BIRDAAA for making)

2. Motion
   
Using file "CSV to Blender", write into Blender Scripts with file path

3. Camera
   
download Blender MMD add-on from [https://github.com/powroupi/blender_mmd_tools] and apply .vmd file to camera object


