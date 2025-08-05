#!/usr/bin/env python3
import cv2
import requests
import numpy as np
import os
import base64
import sys
from PIL import Image
import pyrealsense2 as rs
from scipy.io.wavfile import write
import scipy.io as scio  # Handling MATLAB files
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
import time
from moveit_msgs.msg import Constraints
import paho.mqtt.client as mqtt
from sensor_msgs.msg import  JointState
import paho.mqtt.client as mqtt
import tf
import tf.transformations as tf_trans
from std_msgs.msg import Bool

# Define the broker address and port
BROKER_ADDRESS = "172.22.247.109"
BROKER_PORT = 15672

# Define the server URL
# url = "http://100.106.58.3:8000/predict"  # Note the '/predict' endpoint
url = "http://172.22.247.237:8000/predict"



class EVFsamClientBroadcastObject():
    def __init__(self):
        self.listener = tf.TransformListener()
        self.broadcaster = tf.TransformBroadcaster()
        self.rate = rospy.Rate(10)  # Broadcast at 10 Hz

        # Initialize parameters
        self.init_params()

        self.init_moveit()

        # self.client = mqtt.Client()
        # self.client.on_connect = self.on_connect
        # self.client.on_message = self.on_message


        self.tf_done_pub = rospy.Publisher("/object_found", Bool, queue_size=1, latch=True)

 
    def init_params(self):

        # # EVF-SAM params
        # self.version = "YxZhang/evf-sam2"
        # self.vis_save_path = "./infer"
        # self.precision = "fp16" # "fp32", "bf16", "fp16"
        # self.image_size = 224
        # self.model_max_length = 512
        # self.local_rank = 0
        # self.load_in_8bit = False
        # self.load_in_4bit = False
        # self.model_type = "ori" # "ori", "effi", "sam2"
        # self.image_path = "assets/zebra.jpg"
        self.prompt = "banana"
        # self.prompt = "pick up a blue cup"

        self.img_w = 1280
        self.img_h = 720
        self.frame_rate = 30
        self.img_save_dir = "image_files"
 
        self.display_count = 0
        self.display_itr = 15
 
        self.data_path = '/home/sandisk/koyo_ws/demo_ws/src/demo_pkg/evfsam_graspnet_demo/data'
 
    def init_moveit(self):
        # Initialize planning group
        self.robot = moveit_commander.RobotCommander(robot_description="my_gen3_lite/robot_description")
        self.arm_group = moveit_commander.MoveGroupCommander(
            "arm", 
            robot_description="my_gen3_lite/robot_description", 
            ns="/my_gen3_lite"
        )
        self.gripper_group = moveit_commander.MoveGroupCommander(
            "gripper", 
            robot_description="my_gen3_lite/robot_description",
            ns="/my_gen3_lite"
        )

        # Set robot arm's speed and acceleration
        self.arm_group.set_max_acceleration_scaling_factor(1)
        self.arm_group.set_max_velocity_scaling_factor(1)
        self.arm_group.clear_path_constraints()

        # We can get the name of the reference frame for this robot:
        planning_frame = self.arm_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        ee_link = self.arm_group.get_end_effector_link()
        print("============ End effector link: %s" % ee_link)

        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", group_names)

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())

        print("============ Printing active joints in arm group")
        print(self.arm_group.get_active_joints())

        print("============ Printing joints in arm group")
        print(self.arm_group.get_joints())

        print("============ Printing active joints in gripper group")
        print(self.gripper_group.get_active_joints())

        print("============ Printing joints in gripper group")
        print(self.gripper_group.get_joints())
        # breakpoint()

        # Targe grasp position
        self.target_pose = PoseStamped()
        self.target_pose_update = True

        # Create an MQTT client instance
        self.client = mqtt.Client()
        
        self.object = None

        # We can get the name of the reference frame for this robot:
        planning_frame = self.arm_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = self.arm_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", self.robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")


    def start_demo(self):
        print("----------------------------------------------------")
        print("DEMO START")
        # self.receive_message()

        self.start_evfsam()
        self.process_img(self.data_path)
        self.start_grasp_planning()

        self.send_message()
    
    ####### MQTT #######

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to Mosquitto broker!")
            self.client.subscribe("robot/arrival")
            print("Waiting for message .....")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        print(f"Message received: {msg.payload.decode()}")
        self.object = msg.payload.decode()
        print("Received object name: ", self.object)
        client.loop_stop()
 
    def receive_message(self):
        print(f"Connecting to broker {BROKER_ADDRESS}:{BROKER_PORT}...")
        self.client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

        # Start loop in a background thread
        self.client.loop_start()
        print("Listening for messages...")

        # Wait for a message to be received
        while self.object is None:
            time.sleep(0.1)

        print("Message received. Exiting receive_message.")

    def send_message(self):
        print(f"Connecting to broker {BROKER_ADDRESS}:{BROKER_PORT}...")
        self.client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

        # Start loop in a background thread
        self.client.loop_start()

        # Publish a message
        print("Publishing message...")
        self.client.publish("robot/arrival", "Grasp done")

        # Wait for the message to be sent
        time.sleep(1)

        # Stop the loop and disconnect
        self.client.loop_stop()
        self.client.disconnect()
        print("Message sent and client disconnected.")

    ####### EVF-SAM Client #######
 
    def start_evfsam(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.img_w, self.img_h, rs.format.bgr8, self.frame_rate)
        config.enable_stream(rs.stream.depth, self.img_w, self.img_h, rs.format.z16, self.frame_rate)
        
        try:
            # pipeline.start(config)
            # Start streaming
            pipeline_profile = pipeline.start(config)
            # Get camera intrinsics
            profile = pipeline_profile.get_stream(rs.stream.color)
            intrinsics = profile.as_video_stream_profile().get_intrinsics()
            camMatrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                [0, intrinsics.fy, intrinsics.ppy],
                                [0, 0, 1]])
            distCoeffs = np.zeros(5)  # Assuming no lens distortion

            print("Streaming started. Press 'q' to quit.")
            os.makedirs(self.img_save_dir, exist_ok=True)
            count = 0
            color_image = None
            depth_image = None

            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    print("Error: Could not read frame.")
                    continue

                # Convert RealSense frames to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # # Flip the images horizontally (left-to-right) and vertically (up-to-down)
                # color_image = cv2.flip(color_image, -1)  # Flip both horizontally and vertically
                # depth_image = cv2.flip(depth_image, -1)  # Flip both horizontally and vertically

                # Normalize depth image for display
                depth_display_image = cv2.convertScaleAbs(depth_image, alpha=0.03)

                # Send the POST requestDisplay images
                cv2.imshow("Color Image", color_image)
                cv2.imshow("Depth Image", depth_display_image)

                # Convert the color frame to PNG format
                _, buffer = cv2.imencode('.png', color_image)

                count += 1

                # Prepare the payload
                files = {
                    "text_prompt": (None, self.prompt),
                    "image": ("image.png", buffer.tobytes(), "image/png")
                }

                # print("Send the POST request")
                # Send the POST request
                response = requests.post(url, files=files)

                if response.status_code == 200:
                    response_json = response.json()
                    print(f"Processed output received for frame {count}")

                    # Decode the base64-encoded image
                    seg_img_data = base64.b64decode(response_json["segmentation_image"])
                    seg_img_array = np.frombuffer(seg_img_data, dtype=np.uint8)
                    segmentation_image = cv2.imdecode(seg_img_array, cv2.IMREAD_COLOR)
                    # Display the processed output image
                    cv2.imshow("Segmentation Image", segmentation_image)

                    # Decode the base64-encoded image
                    bb_img_data = base64.b64decode(response_json["bounding_box_image"])
                    bb_img_array = np.frombuffer(bb_img_data, dtype=np.uint8)
                    bounding_box_image = cv2.imdecode(bb_img_array, cv2.IMREAD_COLOR)
                    # Display the processed output image
                    cv2.imshow("Bounding Box Image", bounding_box_image)

                    # Decode the base64-encoded image
                    mask_img_data = base64.b64decode(response_json["mask_image"])
                    mask_img_array = np.frombuffer(mask_img_data, dtype=np.uint8)
                    # img_array = (img_array * 255).astype(np.uint8)
                    mask_image = cv2.imdecode(mask_img_array, cv2.IMREAD_COLOR)
                    # mask_image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                    # print("mask_image: ", mask_image)
                    # Display the processed output image
                    cv2.imshow("Mask Image", mask_image)

                    xmin = response_json["xmin"]
                    ymin = response_json["ymin"]
                    xmax = response_json["xmax"]
                    ymax = response_json["ymax"]
                    center_x = response_json["center_x"]
                    center_y = response_json["center_y"]
                    print("xmin: ", xmin)
                    print("ymin: ", ymin)
                    print("xmax: ", xmax)
                    print("ymax: ", ymax)
                    print("center_x: ", center_x)
                    print("center_y: ", center_y)

                else:
                    print(f"Error: {response.status_code}")
                    print(response.text)

                # Exit the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                self.display_count += 1
                if self.display_count == self.display_itr:
                    print("Break the loop")
                    break

        finally:
            pipeline.stop()

            # Save the last captured color and depth images
            color_image_path = os.path.join(self.img_save_dir, "color_image.png")
            depth_image_path = os.path.join(self.img_save_dir, "depth_image.png")
            depth_display_image_path = os.path.join(self.img_save_dir, "depth_display_image.png")
            segmentation_image_path = os.path.join(self.img_save_dir, "segmentation_image.png")
            bounding_box_image_path = os.path.join(self.img_save_dir, "bounding_box_image.png")
            mask_image_path = os.path.join(self.img_save_dir, "mask_image.png")

            cv2.imwrite(color_image_path, color_image)
            cv2.imwrite(depth_image_path, depth_image)
            cv2.imwrite(depth_display_image_path, depth_display_image)
            cv2.imwrite(segmentation_image_path, segmentation_image)
            cv2.imwrite(bounding_box_image_path, bounding_box_image)
            cv2.imwrite(mask_image_path, mask_image)

            print(f"Color image saved as {color_image_path}")
            print(f"Depth image saved as {depth_image_path}")
            print(f"Depth display image saved as {depth_display_image_path}")
            print(f"Segmentation image saved as {segmentation_image_path}")
            print(f"Bounding box image saved as {bounding_box_image_path}")
            print(f"Mask image saved as {mask_image_path}")

            self.color_image = color_image
            self.depth_image = depth_image
            self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
            self.center_x, self.center_y = center_x, center_y
            self.intrinsics = intrinsics

            cv2.destroyAllWindows()
    
            
    ####### Image -> object in base frame ######
 
    def process_img(self, data_dir):
        depth = self.depth_image
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        intrinsic[0][0] = self.intrinsics.fx
        intrinsic[1][1] = self.intrinsics.fy
        intrinsic[0][2] = self.intrinsics.ppx
        intrinsic[1][2] = self.intrinsics.ppy
        factor_depth = 1000  # Set depth scaling factor for the camera

        # # Extract 2D bounding box coordinates
        # xmin, ymin, xmax, ymax = self.xmin, self.ymin, self.xmax, self.ymax

        # print("Top-left in camera: ", (xmin, ymin))
        # print("Top-right in camera: ", (xmax, ymin))
        # print("Bottom-left in camera: ", (xmin, ymax))
        # print("Bottom-right in camera: ", (xmax, ymax))

        # # Convert all 4 corners to 3D coordinates
        # top_left_3d = self.pixel_to_camera_coords(xmin, ymin, depth, intrinsic, factor_depth)  # Top-left
        # top_right_3d = self.pixel_to_camera_coords(xmax, ymin, depth, intrinsic, factor_depth)  # Top-right
        # bottom_left_3d = self.pixel_to_camera_coords(xmin, ymax, depth, intrinsic, factor_depth)  # Bottom-left
        # bottom_right_3d = self.pixel_to_camera_coords(xmax, ymax, depth, intrinsic, factor_depth)  # Bottom-right

        # print("Top-left in 3d: ", top_left_3d)
        # print("Top-right in 3d: ", top_right_3d)
        # print("Bottom-left in 3d: ", bottom_left_3d)
        # print("Bottom-right in 3d: ", bottom_right_3d)
    
        # corners_3d = np.array([top_left_3d, top_right_3d, bottom_left_3d, bottom_right_3d])
        # obj_center = np.mean(corners_3d, axis=0)
        # print("Object center in 3D: ", obj_center)

        center_x, center_y = self.center_x, self.center_y
        obj_center = self.pixel_to_camera_coords(center_x, center_y, depth, intrinsic, factor_depth)
        print("Object center in 3D: ", obj_center)

        R_obj = np.eye(3)
        T_camera_color_obj = self.construct_homogeneous_transform(obj_center, R_obj)


        # --- Make Base to Object ---
        T_base_camera_color = self.get_frame1_to_frame2('/base_link', '/camera_color')
        T_base_obj = T_base_camera_color @ T_camera_color_obj

        print("Transformation Matrix (base to camera color):\n", T_base_camera_color)
        print("Transformation Matrix (camera color to object):\n", T_camera_color_obj)
        print("Transformation Matrix (base to object):\n", T_base_obj)


        self.publish_tf(T_base_obj, "/base_link", "/object")

        # self.target_pos = self.transformation_to_pose(T_base_obj)
        # print("Target Pose:\n", self.target_pos)


    def pixel_to_camera_coords(self, x, y, depth, intrinsic, factor_depth):
        Z_pred = depth[y, x] / factor_depth  # Depth (distance from camera)
        print("Predicted Z: ", Z_pred)
        Z = 1
        X = (x - intrinsic[0][2]) * Z / intrinsic[0][0]  # X in camera coordinates
        Y = (y - intrinsic[1][2]) * Z / intrinsic[1][1] 
        # Y = -(y - intrinsic[1][2]) * Z / intrinsic[1][1]  # Invert Y to increase upwards
        return X, Y, Z

    # def pixel_to_camera_coords(self, x, y, depth, intrinsic, factor_depth):
    #     Z = depth[y, x] / factor_depth  # Compute the depth value
    #     X = (x - intrinsic[0][2]) * Z / intrinsic[0][0]  # Compute X in camera coordinates
    #     Y = (y - intrinsic[1][2]) * Z / intrinsic[1][1]  # Compute Y in camera coordinates
    #     return X, Y, Z  # Include Z in the return statement


    ####### utils #######

    def construct_rot_matrix_homogeneous_transform(self, translation, quaternion):
        # Convert quaternion to rotation matrix
        rotation_matrix = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
        # print("Rotation Matrix:")
        # print(rotation_matrix)

        # Construct the 4x4 homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation

        return T
    
    def construct_homogeneous_transform(self, translation, rotation_matrix):
        # Construct the 4x4 homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation

        return T
    
    def get_frame1_to_frame2(self, frame1, frame2):
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.listener.lookupTransform(frame1, frame2, rospy.Time(0))
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        T = self.construct_rot_matrix_homogeneous_transform(trans, rot)

        return T
    
    def transformation_to_pose(self, T):
        # Extract translation (position)
        translation = T[:3, 3]  # [x, y, z]

        # Extract rotation matrix
        rotation_matrix = T[:3, :3]

        # Convert rotation matrix to quaternion
        quaternion = tf_trans.quaternion_from_matrix(T)

        # Create a Pose message
        pose = Pose()
        pose.position.x = translation[0]
        pose.position.y = translation[1]
        pose.position.z = translation[2]

        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]

        return pose

    def publish_tf(self, T, parent_frame, child_frame):
        """ Continuously broadcast the transformation """
        # Extract translation (last column of the first 3 rows)
        translation = T[:3, 3]
        # Extract rotation matrix (upper-left 3×3)
        rotation_matrix = T[:3, :3]
        # Convert rotation matrix to quaternion
        quaternion = tf.transformations.quaternion_from_matrix(T)

        rospy.loginfo(f"Publishing transformation: {parent_frame} → {child_frame}")
        rospy.loginfo(f"Translation: {translation}")
        rospy.loginfo(f"Quaternion: {quaternion}")

        while not rospy.is_shutdown():
            # Broadcast transform
            self.broadcaster.sendTransform(
                translation,    # Position (x, y, z)
                quaternion,     # Orientation (x, y, z, w)
                rospy.Time.now(),
                child_frame,  # Child frame
                parent_frame      # Parent frame
            )

            rospy.loginfo(f"Translation: {translation}")

            # Publish completion signal
            self.tf_done_pub.publish(True)
            rospy.loginfo("Object found and completion signal sent.")

            self.rate.sleep()



    ####### grasp planning #######

    def start_grasp_planning(self):
        rospy.loginfo("Starting grasp planning...")
        rospy.loginfo("Moving to start position")
        self.go_sp()
        time.sleep(5)  # Add a 5-second delay
        print("waited for 5 seconds")
        self.execute_grasp_sequence()

    def execute_grasp_sequence(self):
        
        rospy.loginfo('Starting grasp sequence')

        # self.object = "banana"

        if self.object == "banana":
            # 1. Grasp the object
            self.grasp_banana()


        width = 0.06
        self.gripper_move(3.6 * width)
    
        self.go_safepos()

        # 4. Place the object at the target location
        self.place_obj()
        # breakpoint()
        
        self.go_sp()

        rospy.loginfo("Grasp completed successfully.")


    def grasp_banana(self):

        self.arm_group.set_joint_value_target([0.2710410894139424, -1.670172715503421, -0.00410393123981212, -1.5096645292365638, -1.4751765931200378, -1.2731055169847494])
        self.arm_group.go(wait=True)

        # pose_goal =Pose()
        # pose_goal.position.x = 0.5220786750243936
        # pose_goal.position.y = 0.09561009672562862
        # pose_goal.position.z = -0.04516127388096612
        # pose_goal.orientation.x = -0.6943437946518083
        # pose_goal.orientation.y = 0.7188652322922656
        # pose_goal.orientation.z = 0.021450509382198883
        # pose_goal.orientation.w = 0.025677777885934967
        # self.arm_group.set_pose_target(pose_goal)
        # # `go()` returns a boolean indicating whether the planning and execution was successful.
        # success = self.arm_group.go(wait=True)
        # # Calling `stop()` ensures that there is no residual movement
        # self.arm_group.stop()
        # # It is always good to clear your targets after planning with poses.
        # # Note: there is no equivalent function for clear_joint_value_targets().
        # self.arm_group.clear_pose_targets()
    
    
    def go_safepos(self):
        self.arm_group.set_joint_value_target([0.9311649540823903, -0.8526062292989058, 1.1426484404583457, -1.475054620342112, -1.1986110423408354, -0.6476030128034402])

        self.arm_group.go(wait=True)

        # waypoint_pose =Pose()
        # waypoint_pose.position.x = 0.403037475480482
        # waypoint_pose.position.y = 0.3234225897779779
        # waypoint_pose.position.z = 0.15802923128916288
        # waypoint_pose.orientation.x = -0.7076428207095652
        # waypoint_pose.orientation.y = 0.7062677469214905
        # waypoint_pose.orientation.z = -0.005752896205189294
        # waypoint_pose.orientation.w = 0.019859812232360337
        # self.arm_group.set_pose_target(waypoint_pose)
        # # `go()` returns a boolean indicating whether the planning and execution was successful.
        # success = self.arm_group.go(wait=True)
        # # Calling `stop()` ensures that there is no residual movement
        # self.arm_group.stop()
        # # It is always good to clear your targets after planning with poses.
        # # Note: there is no equivalent function for clear_joint_value_targets().
        # self.arm_group.clear_pose_targets()

    def place_obj(self):
        # self.arm_group.set_joint_value_target([1.3848293857713174, -2.001753315449263, -1.031860406478831, -1.8942212622184798, -1.4528502484374677, -1.0467149864070748])
        self.arm_group.set_joint_value_target([1.2601399172176984, -1.8302055275156999, -0.6814011901976365, -1.781343712047689, -1.2712748600514292, -0.9486067950096464])
        self.arm_group.go()


        self.gripper_move(0.6)

    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)

        self.gripper_move(0.6)

    def gripper_move(self, width):
        joint_state_msg = rospy.wait_for_message("/my_gen3_lite/joint_states", JointState, timeout=1.0)
        # print("joint_state_msg: ", joint_state_msg)

        # Find indices of the gripper joints
        right_finger_bottom_index = joint_state_msg.name.index('right_finger_bottom_joint')
        # print("right finger bottom index: ", right_finger_bottom_index)

        # self.gripper_group.set_joint_value_target([width])
        self.gripper_group.set_joint_value_target(
            {"right_finger_bottom_joint": width})
        self.gripper_group.go()
 
 
 
def main():
    rospy.init_node('evfsam_client_node', anonymous=True)
    evfsam_graspnet = EVFsamClientBroadcastObject()
    evfsam_graspnet.start_demo()
 
if __name__ == "__main__":
    main()