#!/usr/bin/env python3
import rospy
import moveit_commander
import numpy as np
import tf
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans
from sensor_msgs.msg import  JointState
import cv2
import pyrealsense2 as rs
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose

import time

SET_START = 0

BROKER_ADDRESS = "172.22.3.12"
# BROKER_ADDRESS = "172.22.2.12"
BROKER_PORT = 1883


class GraspObjectNode():
    def __init__(self):
        self.listener = tf.TransformListener()
        self._rate = rospy.Rate(10)  # Broadcast at 10 Hz

        self.ready_to_grasp = False
        self.grasp_done = False
        rospy.Subscriber("/object_found", Bool, self.object_found_callback)

        self.init_moveit()

        self.manip_done_pub = rospy.Publisher("/manip_done", Bool, queue_size=1, latch=True)

    

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



    def object_found_callback(self, msg):
        if msg.data:  # If True received
            rospy.loginfo("Received Object found signal. Starting grasp sequence...")
            self.ready_to_grasp = True


    def start(self):
        if SET_START:
            rospy.loginfo("Moving to start position")
            self.go_sp()
        else:
            rospy.loginfo("Moving to start position")
            self.go_sp()
            rospy.loginfo("Waiting for /object_found signal to start grasp sequence...")
            while not rospy.is_shutdown() or not self.grasp_done:
                if self.ready_to_grasp:
                    rospy.loginfo("Signal received. Executing grasp sequence.")
                    T_base_obj = self.get_frame1_to_frame2("/base_link", "/object")
                    self.target_pos = self.transformation_to_pose(T_base_obj)
                    print("Target Pose:\n", self.target_pos)
                    self.execute_grasp_sequence()
                self._rate.sleep()

            # self.send_message()
    
    def start(self):
        if SET_START:
            rospy.loginfo("Moving to start position")
            self.go_sp()
            # T_base_ee = self.get_frame1_to_frame2("/base_link", "/tool_frame")
            # print("T_base_ee: ", T_base_ee)
        else:
            rospy.loginfo("Moving to start position")
            self.go_sp()
            rospy.loginfo("Waiting for /object_found signal to start grasp sequence...")

            while not rospy.is_shutdown() and not self.grasp_done:
                if self.ready_to_grasp:
                    rospy.loginfo("Signal received. Executing grasp sequence.")
                    
                    # Get object transform and convert to pose
                    T_base_obj = self.get_frame1_to_frame2("/base_link", "/object")
                    self.target_pos = self.transformation_to_pose(T_base_obj)
                    print("Target Pose:\n", self.target_pos)
                    
                    # Run the grasp sequence
                    self.execute_grasp_sequence()

                    # Set done flags
                    self.ready_to_grasp = False
                    self.grasp_done = True

                    self.manip_done_pub(True)
                    rospy.loginfo("Manipulation done, completion signal sent.")

                self._rate.sleep()



    ####### grasp planning #######


    def execute_grasp_sequence(self):
        
        rospy.loginfo('Starting grasp sequence')

        # self.object = "banana"
        # if self.object == "banana":
        #     # 1. Grasp the object
        #     self.grasp_banana()

        self.grasp_object()

        width = 0.06
        self.gripper_move(3.6 * width)
        self.go_safepos()
        self.place_obj()
        self.go_sp()

        rospy.loginfo("Grasp completed successfully.")



    def grasp_object(self):
        """
        Grasp object based on object detection.
        """

        # self.target_pos.position.z += 0.05
        # self.target_pos.position.x += 0.125
        # self.target_pos.position.y -= 0.2
        # self.target_pos.position.x += 0.1
        # self.target_pos.position.y -= 0.15
        # self.target_pos.position.x += 0.075
        # self.target_pos.position.y += 0.075
        self.arm_group.set_pose_target(self.target_pos)
        self.arm_group.go()
        # self.plan_cartesian_path(self.target_pos)
        # self.target_pos.position.z += 0.05
        # self.plan_cartesian_path(self.target_pos)
        # self.arm_group.set_pose_target(self.target_pos)
        # self.arm_group.go()


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


        self.gripper_move(0.7)

    def go_sp(self):
        self.arm_group.set_joint_value_target([-0.08498747219394814, -0.2794001977631106,
                                               0.7484180883797364, -1.570090066123494,
                                               -2.114137663337607, -1.6563429070772748])
        self.arm_group.go(wait=True)

        self.gripper_move(0.7)

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
        pose.position.x = translation[0]-0.03 # translation[0]
        pose.position.y = translation[1]+0.065 # translation[1]
        pose.position.z = -0.04931002 # -0.04731002  # translation[2]

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

            # Publish completion signal
            self.tf_done_pub.publish(True)
            rospy.loginfo("Object found and completion signal sent.")

            self.rate.sleep()



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



def main():
    rospy.init_node('grasp_object', anonymous=True)
    grasp_planner_node = GraspObjectNode()
    grasp_planner_node.start()
    rospy.spin()
 
if __name__ == "__main__":
    main()