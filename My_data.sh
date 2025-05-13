python3 My_original_data/utils/csv_to_npz.py
python3 My_original_data/utils/npz_to_json.py
python3 My_original_data/utils/move_file.py

# align the music audio, motion and camera raw data
python3 scripts/my_utils/modify_motion_json.py \
--input_dir My_data/Simplified_MotionGlobalTransform/ \
--output_dir My_data/Modified_Simplified_MotionGlobalTransform/ 

python3 scripts/amc_align.py \
--amc_raw_dir My_data/amc_raw_data/ \
--amc_raw_camera_json_dir My_data/amc_camera_json/ \
--amc_motion_transform_json_dir My_data/Modified_Simplified_MotionGlobalTransform/ \
--amc_aligned_dir My_data/amc_aligned_data/

# interpolate the camera keyframe data
python3 scripts/camera_preprocess.py \
--camera_keyframe_json_dir My_data/amc_aligned_data/CameraKeyframe \
--camera_interpolation_json_dir My_data/amc_aligned_data/CameraInterpolated \
--camera_centric_json_dir My_data/amc_aligned_data/CameraCentric

# split raw data to sub-sequences
python3 scripts/split_long_data.py --split_from_file True \
--audio_dir My_data/amc_aligned_data/Audio \
--camera_kf_dir My_data/amc_aligned_data/CameraKeyframe \
--camera_c_dir My_data/amc_aligned_data/CameraCentric \
--motion_dir My_data/amc_aligned_data/Simplified_MotionGlobalTransform \
--split_record My_data/split/long2short.json \
--output_dir My_data/amc_aligned_data_split/ 

# split sub-sequences into train, test validation sets according to categories
python3 scripts/split_by_categories.py --split_from_file True \
--audio_dir My_data/amc_aligned_data_split/Audio \
--camera_c_dir My_data/amc_aligned_data_split/CameraCentric \
--motion_dir My_data/amc_aligned_data_split/Simplified_MotionGlobalTransform/ \
--music_categories My_data/split/music_categories.json \
--split_train My_data/split/train.json \
--split_validation My_data/split/validation.json \
--split_test My_data/split/test.json \
--output_dir My_data/amc_data_split_by_categories

# make_dcm_plusplus.sh
python3 scripts/make_dcm_plusplus.py \
--audio_dir My_data/amc_aligned_data/Audio \
--camera_kf_dir My_data/amc_aligned_data/CameraKeyframe \
--camera_c_dir My_data/amc_aligned_data/CameraCentric \
--motion_dir My_data/amc_aligned_data/Simplified_MotionGlobalTransform/ \
--split_train My_data/split/train.json \
--split_validation My_data/split/validation.json \
--split_test My_data/split/test.json \
--output_dir My_DCM++ \
--split_record My_data/split/long2short.json

# test_stage1.sh 
python3 scripts/test_stage1.py \
--data_path My_DCM++/ \
--checkpoint checkpoints/DCA_stage1.pt \
--save_dir My_output/stage1 \
--processed_data_dir My_DCM++/stage1_dataset_backups \
--force_reload \
--exp_name stage1_test

# test_stage2n3.sh
python3 scripts/test_stage2n3.py \
--data_path My_DCM++/ \
--camera_format polar \
--use_generate_keypoints \
--generated_keyframemask_dir My_output/stage1/test_stage1_test/test \
--checkpoint checkpoints/DCA_stage2n3.pt \
--render_dir My_output \
--processed_data_dir My_DCM++/mix_dataset_backups/DCA \
--force_reload \
--exp_name DCA #\
#--render_videos
# If you want redered video, activate upward code

# Evaluate 

python3 scripts/evaluate.py --result_dir My_output/test_DCA/etest --test_dir My_DCM++/Test

# Camera Visualize

python3 scripts/extend_camera_results.py --split_json My_data/split/long2short.json \
--source_camera_dir My_output/test_DCA/etest/CameraCentric \
--target_camera_extend_dir My_output/test_DCA/etest/extend_json \
--target_camera_vmdjson_dir My_output/test_DCA/etest/extend_vmdjson

python3 scripts/json2vmd.py \
--json_dir My_output/test_DCA/etest/extend_vmdjson/ \
--vmd_dir My_output/test_DCA/etest/extend_vmd/ \
--data_type camera
