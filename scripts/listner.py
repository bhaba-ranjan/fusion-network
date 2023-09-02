#!/usr/bin/env python
# coding: utf-8
import numpy as np
import rospy
import subprocess
import message_filters
import sensor_msgs.point_cloud2 as pc2
import cv2
import sys
import os
import pickle
import torch 

from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion, Pose
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, CompressedImage
from nav_msgs.msg import Odometry
from pathlib import Path
import matplotlib.pyplot as plt
from on_robot_transformer import ApplyTransformation
from model_builder.multimodal.multi_net  import MultiModalNet
from model_builder.image.image_head import ImageHeadMLP
from data_builder.gaussian_weights import get_gaussian_weights
from data_builder.cmd_scaler import transform_to_gt_scale


device = "cpu" if torch.cuda.is_available() else "cpu"

print(f'Using device ========================================>  {device}')

weights_base = get_gaussian_weights(6,3)[:,:4] 
weights = np.concatenate([weights_base, weights_base], axis=1)
weights = torch.tensor(weights)
weights = weights.to('cuda')

model = MultiModalNet()
model.to(device)
ckpt = torch.load('/home/administrator/Workspace_Ranjan/fusion-network/scripts/tf8_end_to_end_velocities_160_0.9251317143167459.pth',map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print('model_loaded')
counter = {'index': 0, 'sub-sampler': 1}

constant_goal = None

def get_lidar_points(lidar):
    point_cloud = []
    for point in pc2.read_points(lidar, skip_nans=True, field_names=('x', 'y', 'z')):
        pt_x = point[0]
        pt_y = point[1]
        pt_z = point[2]
        if point[0] >= -1 and point[0] <= 5 and point[1]>=-3 and point[1]<=3 and point[2] >= 0.0299 and point[2] <= 6.0299:
            point_cloud.append([pt_x, pt_y, pt_z])
    return point_cloud

def read_image(rgb_image):
    np_arr = np.frombuffer(rgb_image.data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def marker_callback(xs, ys):
    marker = Marker()
    marker.header.frame_id='base_link'
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    # marker.lifetime = rospy.Duration(1)
    marker.scale.x = 0.02
    marker.color.a = 1.0
    marker.color.r = 1
    marker.color.g = 0.0
    marker.color.b = 0
    marker.pose.orientation = Quaternion(0,0,0,1)

    points_list = []
    points_list.append(Point(y=0,x= 0,z=0))

    for i in range(4):    
        points_list.append(Point(y=ys[i],x= xs[i],z=0))

    marker.points.extend(points_list)
    pub.publish(marker)
    return

def get_goals(pts):
    goals = pts.detach().cpu().numpy()[0]
    x = goals[:4]
    y = goals[4:]
    # print(x)
    # print(y)
    marker_callback(x,y)


def set_local_goal(goal):
    global constant_goal
    # print(f'constant global goal changed: {goal}')
    constant_goal = (goal.position.x, goal.position.y)
    
    return constant_goal

def publish_cmd_vel(cmd_vel):
    cmd_vel = transform_to_gt_scale(cmd_vel, device)
    cmd_vel = cmd_vel.detach().cpu().numpy()[0]
    print(f'publish velocity {cmd_vel[0]}')

    cmd_msg = Twist()
    # cmd_msg.linear.x = 0.3
    # cmd_msg.angular.z = cmd_vel[1]
    # cmd_publisher.publish(cmd_msg)    

def aprrox_sync_callback(lidar, rgb):

    if counter['sub-sampler'] % 1 == 0:
        # TODO: these 4 values will be pickled at index counter['index'] except image
        img = read_image(rgb)
        point_cloud = get_lidar_points(lidar)

        if constant_goal == None:
            print('waiting for local goal...')
            return
        
        align_content = {
            "pcl": point_cloud,
            "images": img,
            "local_goal": constant_goal,
        }
        
        start = rospy.Time().now().to_sec()

        transformer = ApplyTransformation(align_content)
        # print("transformed")

        image, pcl, local_goal = transformer.__getitem__(0)
        pcl = pcl.to(device) 
        image = image.to(device)      
        local_goal = local_goal.to(device)

        image = image.unsqueeze(0)
        pcl = pcl.unsqueeze(0)
        local_goal = local_goal.unsqueeze(0)

        with torch.no_grad():
            
            pts, cmd = model(image, pcl, local_goal)
            
            image.detach()
            pcl.detach()
            local_goal.detach()
            del image, pcl, local_goal
            torch.cuda.empty_cache()
            
            end = rospy.Time().now().to_sec()
            print(f'diff: {end-start}')
            # print("out")
            get_goals(pts/weights)
            publish_cmd_vel(cmd)
            counter['index'] += 1
            print(counter['index'])            
            counter['sub-sampler'] = 0

    counter['sub-sampler'] += 1



rospy.init_node('listen_record_data', anonymous=True)


cmd_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)

# lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
# rgb = message_filters.Subscriber('/zed_node/rgb/image_rect_color/compressed', CompressedImage)
# odom = message_filters.Subscriber('zed_node/odom', Odometry)
# lc_goal = message_filters.Subscriber('/move_base_simple/goal', PoseStamped)

odom = message_filters.Subscriber('/odometry/filtered', Odometry)
lidar = message_filters.Subscriber('/velodyne_points', PointCloud2)
rgb = message_filters.Subscriber('/zed_node/rgb/image_rect_color/compressed', CompressedImage)
pub = rospy.Publisher('/world_point', Marker, queue_size=10)
lc_goal = message_filters.Subscriber('/local_goal', Pose)
# pub_gt = rospy.Publisher('/world_point_gt', Marker, queue_size=10)

ts = message_filters.ApproximateTimeSynchronizer([lidar, rgb], 100, 0.05, allow_headerless=True)

ts.registerCallback(aprrox_sync_callback)
# odom.registerCallback(odom_callback)
lc_goal.registerCallback(set_local_goal)



rospy.spin()

