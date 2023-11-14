import struct
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_binary_data_3d(file_path):
    try:
        with open(file_path, 'rb') as file:
            # Read binary data, assuming it's structured in some way
            # Here, we assume the binary data consists of XYZ coordinates
            # You may need to modify this part based on your binary data format
            data = file.read()
            
            # Assuming data is structured as (x1, y1, z1, x2, y2, z2, ...)
            num_points = len(data) // 12  # Each point has 3 float values (4 bytes each)

            # Extract x, y, and z coordinates
            x = struct.unpack(f'{num_points}f', data[0:4*num_points])
            y = struct.unpack(f'{num_points}f', data[4*num_points:8*num_points])
            z = struct.unpack(f'{num_points}f', data[8*num_points:])

            # Create a 3D scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)

            # Set labels
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            plt.show()
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage example:
visualize_binary_data_3d('/media/NAS/raw_data/ShuoShen/nuscenes_mini/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin')