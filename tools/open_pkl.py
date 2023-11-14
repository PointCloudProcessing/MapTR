from PIL import Image
import numpy as np
import json
import pprint
import pickle 
import rosbag

def open_pkl(path ='/media/NAS/raw_data/ShuoShen/nuscenes/train/nuscenes/nuscenes_infos_temporal_train.pkl' ):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print("open_pkl")
        print(data.keys())
        print("length",len(data["infos"]))
        # print(data["infos"][0].keys())
        # print(data["infos"][0])
        # print(data["metadata"])
        # print(data["infos"][20]["sweeps"])
        # print(len(data["infos"][0]["valid_flag"]))
        # print(len(data["infos"][0]["gt_velocity"]))
        # print(len(data["infos"][0]["gt_names"]))
        # print(len(data["infos"][0]["gt_boxes"]))

        
        # check the image
        # for info in data['infos']:
        #     path = info["cams"]["CAM_FRONT"]['data_path']
        #     print(path)
        #     try:
        #         image = Image.open(path)
        #     except (IOError, OSError):
        #         print("Image file is corrupted or in an unsupported format.")
        #     image_array = np.array(image)
        #     min_pixel_value = np.min(image_array)
        #     max_pixel_value = np.max(image_array)

        #     if min_pixel_value < 0 or max_pixel_value > 255:
        #         print("Image contains unexpected pixel values.")

def loadjson(path = '/media/NAS/raw_data/ShuoShen/nuscenes/train/nuscenes/nuscenes_infos_temporal_val_mono3d.coco.json'):
    data = json.load(open(path, 'rb'))
    print(data.keys())
    print(data["images"][0].keys())
    print(data["images"][0]["file_name"])
    print(data["annotations"][0].keys())
    print(data["annotations"][0]["segmentation"])
    print(len(data["images"]))  #320
    print(len(data["annotations"])) #2590
    print(len(data["categories"]))  #10


def generate_ld_pkl(bag_path):
    data = {"infos":[], "metadata":{'version': 'v1.0-trainval'}}
    frame_idx = 0
    with rosbag.Bag(bag_path, mode='r') as bag:
        for _, _, t in bag.read_messages(topics=['/innoviz']):
            print(t)
            data["infos"].append({"bag_path":bag_path, 
                                 "frame_id":frame_idx, 
                                 'scene_token': frame_idx,
                                 "timestamp":t,
                                 "prev":max(0, frame_idx-1), 
                                 "next":frame_idx+1,
                                 "sweeps":[{"frame_idx": i, "topic":"/basler_front", "timestamp":data["infos"][i]["timestamp"] } for i in range(max(0,frame_idx-10),frame_idx)],
                                 "cams":{'type': 'CAM_FRONT', "topic":"/basler_front", 'frame_id': frame_idx},
                                 "can_bus":{"topic":"/ld_imugps_can", 'frame_id': frame_idx},
                                 "map_location":"Anywhere",
                                 "gt_boxes":[],
                                 "gt_names":[], 
                                 "gt_velocity":[],
                                 "num_lidar_pts":[],
                                 "num_radar_pts":[],
                                 "valid_flag":[]
                                 })
            frame_idx += 1
            
        file = open('gherkin.pkl', 'wb')
        pickle.dump(data, file)
        file.close()

def read_bag(bag_path):
    with rosbag.Bag(bag_path, mode='r') as bag:
        for topic, msg, t in bag.read_messages(topics=['/lidar']):
            print(topic)
            print(msg)
            print(t)
            break
        
if __name__ == '__main__':
    # loadjson()
    open_pkl()
    # read_bag("/media/NAS/raw_data/ShuoShen/VectorMapNet_data/0_Build_Dataset/CCA_9003_0140092_Liangdao_CONT_20221102T143314-20221102T143414_sync.bag")
    # generate_ld_pkl("/media/NAS/raw_data/ShuoShen/VectorMapNet_data/0_Build_Dataset/CCA_9003_0140092_Liangdao_CONT_20221102T143314-20221102T143414_sync.bag")