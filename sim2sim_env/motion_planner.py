import numpy as np
import pybullet as p
import time

class MotionPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.kuka_end_effector_index = 6  # End effector is J6
        self.left_finger_index = 8        # Left gripper finger joint
        self.right_finger_index = 11      # Right gripper finger joint
        self.approach_height = 0.5       # Height above object before descent
        self.grasp_height = 0.25          # Final grasp height above the object
        self.gripper_close_value = 0.02   # Value to close the gripper fingers
        self.lift_height = 0.3            # Lift height after grasping
    
    def compute_ik(self, target_position, target_orientation):
        """Compute inverse kinematics to get joint positions for a target end-effector pose."""
        joint_positions = p.calculateInverseKinematics(
            self.robot.kukaUid, 
            self.kuka_end_effector_index, 
            target_position, 
            target_orientation,
            maxNumIterations=100
        )
        return joint_positions[:7]  # Only return the arm joints (J0 to J6)

    def move_to_position(self, target_position, orientation):
        """Move end effector to the target position using inverse kinematics."""
        # orientation = p.getQuaternionFromEuler([0, -np.pi, 0])
        joint_positions = self.compute_ik(target_position, orientation)
        
        # Apply joint positions to all arm joints
        for i in range(7):
            p.setJointMotorControl2(
                bodyIndex=self.robot.kukaUid,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500
            )
                
        # Simulate to allow smooth movement
        for _ in range(50):  
            p.stepSimulation()
            time.sleep(1 / 240)

    def approach(self, object_position, object_orientation):
        """Approach the object from above."""
        approach_position = [object_position[0], object_position[1], object_position[2] + self.approach_height]
        self.move_to_position(approach_position, object_orientation)
        
    def descend(self, object_position, object_orientation):
        """Descend to the grasp height above the object."""
        grasp_position = [object_position[0], object_position[1], object_position[2] + self.grasp_height]
        self.move_to_position(grasp_position, object_orientation)

    def close_gripper(self):
        """Close the gripper to grasp the object."""
        # Close both gripper fingers
        force = 500
        p.setJointMotorControl2(self.robot.kukaUid, self.left_finger_index, p.POSITION_CONTROL, targetPosition=self.gripper_close_value, force=force)
        p.setJointMotorControl2(self.robot.kukaUid, self.right_finger_index, p.POSITION_CONTROL, targetPosition=-self.gripper_close_value, force=force)
        
        # Apply action for a short duration
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1 / 240)
    
    def lift(self, steps=1):
        """Lift the object smoothly by gradually increasing the Z position."""
        current_pos = list(p.getLinkState(self.robot.kukaUid, self.kuka_end_effector_index)[0])
        target_pos = [current_pos[0], current_pos[1], current_pos[2] + self.lift_height]
        delta_pos = [(target_pos[i] - current_pos[i]) / steps for i in range(3)]
        orientation = p.getQuaternionFromEuler([0, -np.pi, 0])

        for step in range(steps):
            # Incrementally move to the target position
            current_pos = [current_pos[i] + delta_pos[i] for i in range(3)]
            self.move_to_position(current_pos, orientation)  # Keep the same orientation
            p.stepSimulation()


    def execute_grasp(self, object_position, object_orientation):
        # object_position = [object_position[0], object_position[1], object_position[2] - 0.2]
        rotated_offset, _ = p.multiplyTransforms([0, 0, 0], object_orientation, [0, -0.02, 0], [0, 0, 0, 1])

        # Apply the rotated offset to the base position
        adjusted_position = [
            object_position[0] + rotated_offset[0],
            object_position[1] + rotated_offset[1],
            object_position[2] + rotated_offset[2],
        ]
        """Execute the full grasping sequence."""
        gripper_offset = p.getQuaternionFromEuler([0, -np.pi, 0])  # Gripper pointing downward

        # Combine the cube's orientation with the gripper's offset
        adjusted_orientation = p.multiplyTransforms(
            [0, 0, 0], object_orientation,  # Cube's orientation
            [0, 0, 0], gripper_offset       # Gripper's fixed offset
        )[1]
        
        # adjusted_position[2] = -0.17

        self.approach(adjusted_position, adjusted_orientation)
        self.descend(adjusted_position, adjusted_orientation)
        self.close_gripper()
        self.lift()
