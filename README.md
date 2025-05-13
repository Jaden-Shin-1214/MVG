# MVG
Music Video Generator with MOCAP data and DanceCamAnimator Code

## Docker file
Using dockerfile, implementation under codes

'''
sudo docker build -t dca .
sudo docker run --shm-size=8G --gpus all --name dca -it -v /home/syj/DanceCamAnimator-Official:/workspace dca 
sudo docker exec -it dca /bin/bash
'''




## Prepare

1. Download camera generation model from [https://github.com/Carmenw1203/DanceCamAnimator-Official] and install ckpt. "DCA_stage1.pt" and "DCA_state_2n3.pt"
2. change your Optirack Motive motion capture file (.csv) name into m*_GlobalTransform.csv
3. Also, change yout music file name into a*.mp3
* Both music and csv file have frames smaller than 1050

4. Move Motion Data and music data into 'My_original_data/MOCAP/input' and 'My_original_data/MUSIC'
5. 
