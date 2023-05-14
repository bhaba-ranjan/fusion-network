import rosbag
import os

def check_topics_in_rosbag(rosbag_file, topics):
    # Open the ROS bag file
    bag = rosbag.Bag(rosbag_file)

    # Get the list of available topics in the bag file
    bag_topics = bag.get_type_and_topic_info().topics.keys()

    # Check if all the specified topics are present in the bag file
    missing_topics = []
    for topic in topics:
        if topic not in bag_topics:
            missing_topics.append(topic)

    # Close the ROS bag file
    bag.close()

    return missing_topics

# Example usage
bagfile_location = "/Users/bhabaranjanpanigrahi/Research/Code/fusion-network/bagfiles"
topics = ['/odom', '/image_raw/compressed', '/velodyne_points']


for path in os.listdir(bagfile_location):
    missing_topics = check_topics_in_rosbag(path, topics)

    if len(missing_topics) == 0:
        print("All topics are present in the ROS bag file.")
    else:
        print("Missing topics in the ROS bag file:")
        for topic in missing_topics:
            print(topic)
