import os
import shutil


def merge_nuscene_all_file():
    # 定义源文件夹a和目标文件夹b的路径
    folder_a = '/media/NAS/raw_data/ShuoShen/nuscenes/train/nuscenes'
    number = ["05","06","07","08","09","10"]
    # 获取文件夹a下的所有子文件夹列表
    subfolders = ["samples","sweeps"]

    # 遍历每个子文件夹
    for num in number:
        folder_b = f'/media/NAS/raw_data/ShuoShen/nuscenes/train/v1.0-trainval{num}_blobs'
        for subfolder in subfolders:
            target_path = os.path.join(folder_a, subfolder)
            source_path = os.path.join(folder_b, subfolder)
            
            for item in os.listdir(source_path):
                source_item = os.path.join(source_path, item)
                target_item = os.path.join(target_path, item)
                if os.path.isdir(source_item):
                    command1= f'ls -lR {source_item} | grep "-" | wc -l'
                    os.system(command1)
                    command = f'find {source_item} -type f -print0 | xargs -0 mv -t {target_item}'
                    print(command)
                    os.system(command)
                    print(f'已合并子文件夹: {source_item}')
            print(f'已合并文件: {source_path}')
                

def open_pkl():
    import pickle
    with open('/media/NAS/raw_data/ShuoShen/nuscenes_mini/nuscenes_map_infos_temporal_val.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data['infos'][0].keys())
        
        
if __name__ == '__main__':
    open_pkl()
    # open_pkl()