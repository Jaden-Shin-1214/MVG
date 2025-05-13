import os
import json
import glob
import argparse
from tqdm import tqdm

def switch_xz_and_scale(record, scale_factor):
    if "Transform" in record and len(record["Transform"]) > 0:
        transform = record["Transform"][0]

        # 스케일 먼저
        transform = [x * scale_factor for x in transform]

        # x(z) 값 위치 스위치
        for i in range(60):  # 60개의 joint
            base = 16 * i
            idx_x = base + 12
            idx_z = base + 14
            transform[idx_x], transform[idx_z] = transform[idx_z], transform[idx_x]

        record["Transform"] = transform
 
def normalize_data(data):
    len_data = len(data)
    for i in range(len_data):
        origin_spine_x = data[i]["Transform"][12]
        origin_spine_z = data[i]["Transform"][14]
        for j in range(60):  
            data[i]["Transform"][12 + j * 16] -= origin_spine_x
            data[i]["Transform"][14 + j * 16] -= origin_spine_z        

def process_files(input_dir, output_dir, scale_factor):
    os.makedirs(output_dir, exist_ok=True)
    file_list = glob.glob(os.path.join(input_dir, 'm*_GlobalTransform.json'))

    for file_path in tqdm(file_list, desc="Processing JSON files"):
        with open(file_path, 'r') as f:
            data = json.load(f)

        if "BoneKeyFrameTransformRecord" in data:
            
            for record in data["BoneKeyFrameTransformRecord"]:
                switch_xz_and_scale(record, scale_factor)
            normalize_data(data["BoneKeyFrameTransformRecord"])
        print("✅ 모든 파일이 스케일 & XZ 스위칭 처리되었습니다.")


        filename = os.path.basename(file_path).replace("_GlobalTransform.json", "_GlobalTransform.json")
        save_path = os.path.join(output_dir, filename)

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale and switch X/Z in Transform data")
    parser.add_argument('--input_dir', type=str, required=True, help='원본 JSON 파일들이 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, required=True, help='변환된 JSON 파일들을 저장할 디렉토리')
    parser.add_argument('--scale', type=float, default=1.0, help='스케일 팩터 (기본값: 1.0)')

    args = parser.parse_args()

    process_files(args.input_dir, args.output_dir, args.scale) 
