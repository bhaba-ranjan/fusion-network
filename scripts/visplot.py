import pickle
import coloredlogs, logging
import os
import numpy as np

from torch.utils.data import Dataset
coloredlogs.install()

class IndexDataset(Dataset):

    def __init__(self, dir_path):        
        self.root_path = dir_path
        self.pickle_path = os.path.join(dir_path , 'snapshot.pickle')        
        
        logging.info(f'Parsing pickle file: {self.pickle_path}')
    
        with open(self.pickle_path, 'rb') as data:
            self.content = pickle.load(data)

        logging.info('Picklefile loaded')

        # Exclude keys that does not have a local goal [as the robot did not travel 10 meters]
        keys = list(self.content.keys())
        for key in keys:
            if len(self.content[key]['local_goal'])!=12:                
                # print(key)
                self.content.pop(key)

    def __len__(self):
        # As images and point clouds will be in sets of 4
        return int(len(self.content.keys()) - 4)
    
    def __getitem__(self, offset_index) :
        # We are taking 4 sequential images, point clouds each time to account for temporal variation
        start_index = offset_index

        # Get data from respective index       
        gt_cmd_vel = self.content[start_index]['gt_cmd_vel']
        local_goal = self.content[start_index]['local_goal']
        robot_position = self.content[start_index]['robot_position']
        
        # Image paths
        image_paths = [ os.path.join(self.root_path, str(i)+'.jpg') for i in range(start_index, start_index+1) ]
        
        # only keep points that are under 5 + 1 (delta) meters from the robot
        point_clouds = []
        # print(list(self.content.keys()), start_index, end_index)
        for point_snapshot in range(start_index, start_index+1):
            filtered_points = []            
            for point in self.content[point_snapshot]['point_cloud']:
                if point[0] >= 0 and point[0] <= 8.009 and point[1]>=-3 and point[1]<=3:
                    filtered_points.append(point)                           

            point_clouds.append(filtered_points)                

        return (image_paths, point_clouds, local_goal, robot_position, gt_cmd_vel)
    
    import numpy as np
import matplotlib.pyplot as plt


def scale_min_max(points,min_val=0, max_val=6):    

    x = np.expand_dims(np.array(points[:,0]), axis=1)    
    y = np.expand_dims(np.array(points[:,1]), axis=1)
    z = np.expand_dims(np.array(points[:,2]), axis=1)

    x = np.interp(x,(0, 8), (min_val, 8))
    y = np.interp(y,(-3, 3), (min_val, max_val))

    return np.concatenate([x,y,z], axis=1)



def get_voxelized_points(points_array):
    # Define the grid dimensions
    grid_size = 122  # Adjust based on your requirements

    # Create an empty voxel grid
    voxel_grid = np.zeros((162, grid_size, 42))    

    # Calculate the voxel size based on the grid dimensions
    voxel_size = 0.05

    # if points_array.shape[0] == 0:
    #     input_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
    #     input_tensor = input_tensor.unsqueeze(0)
    #     # print(f'returned_zeroed_array {input_tensor.shape}')
    #     return input_tensor

    # scale coordinate value
    points_array = scale_min_max(points_array)
    

    # Map points to the voxel grid
    grid_indices = np.floor(points_array / voxel_size).astype(int)    
    grid_indices = np.clip(grid_indices, np.array([0,0,0]), np.array([161,121,41]))

    # unique_indices, counts = np.unique(grid_indices, return_counts=True, axis=0)

    voxel_grid[grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]] = 1

    # Convert the voxel grid to a PyTorch tensor
    # input_tensor = torch.tensor(voxel_grid, dtype=torch.float32)
    # input_tensor = input_tensor.unsqueeze(0)  # Add batch and channel dimensions

    return grid_indices 

ds = IndexDataset('/home/ranjan/Workspace/my_works/fusion-network/recorded-data/train/138144_nrw_hall')



x = ds.__getitem__(400)[1]
pts = get_voxelized_points(np.array(x[0]))
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(pts[:,0],pts[:,1],pts[:,2])
# plt.show()

# Generate example points
# num_points = 1000
# points = np.random.rand(num_points, 3)  # Replace this with your own numpy array

import open3d as o3d
# Create a PointCloud object from the numpy array
point_cloud = o3d.geometry.PointCloud()
print(pts.shape)
point_cloud.points = o3d.utility.Vector3dVector(pts)
print(pts.shape)
# Visualize the PointCloud
o3d.visualization.draw_geometries([point_cloud])