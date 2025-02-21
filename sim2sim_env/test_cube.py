import pybullet as p
import numpy as np
import time
import torch
from PIL import Image
from cube_main import CubeDetectorVGG16  # Import your trained VGG detector
from sim2sim_env import Sim2SimEnv
from motion_planner import MotionPlanner

import torch
from torchvision import transforms

import matplotlib.pyplot as plt


def visualize_prediction(predicted_position, actual_position):
    """
    Visualize the predicted position of the cube on the RGB image.
    
    Args:
    - image (numpy.ndarray): The original RGB image as a numpy array.
    - predicted_position (torch.Tensor): The predicted coordinates (x, y) of the cube.
    """
    # Predicted position is assumed to be a tensor of shape [1, 3] for (x, y, z)
    x, y, z = predicted_position[0, 0].item(), predicted_position[0, 1].item(), predicted_position[0, 2].item()
    
    x_a, y_a, z_a = actual_position[0], actual_position[1], actual_position[2]

    # Convert the RGB image to a format that can be shown in the background
    img = Image.fromarray(rgb_image.transpose(1, 2, 0))  # Convert (C, H, W) to (H, W, C)
    fig = plt.figure(figsize=(10, 7))

    # Create a 3D subplot
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cube's predicted position
    ax.scatter(x, y, z, c='red', marker='o', s=100, label='Predicted Cube Position')
    ax.scatter(x_a, y_a, z_a, c='black', marker='o', s=100, label='Actual Cube Position')

    # Plot properties
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Visualization of Cube Prediction')
    ax.legend()

    # Set the viewing angles
    ax.view_init(elev=20, azim=30)  # Adjust elevation and azimuthal angle for better view

    # Show the plot
    plt.show()

def preprocess_image(image):
    """
    Preprocess the image to be compatible with the trained VGG model.
    
    Args:
    - image (PIL.Image): The image to be processed.
    
    Returns:
    - torch.Tensor: The processed image tensor.
    """
    # Transformations to match ImageNet-trained model expectations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Resize image to 224x224
        transforms.ToTensor(),               # Convert PIL image to PyTorch tensor
        transforms.Normalize(                # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0)  # Add a batch dimension
    return image_tensor

def move_robot_to_target(env, target_pos):
    """
    Moves the robot to the target position using inverse kinematics (IK).
    
    Args:
    - env: The simulation environment (Sim2SimEnv instance).
    - target_pos: The 3D coordinates of the target object.
    """

    # Desired position of the end-effector
    desired_position = [target_pos[0], target_pos[1], target_pos[2] + 0.25]  # e.g., [x, y, z]
    
    # Optionally set the desired orientation (e.g., pointing downwards)
    desired_orientation = p.getQuaternionFromEuler([0, -np.pi, 0])

    # Use the PyBullet IK solver to get the joint angles
    kuka_end_effector_index = env._kuka.kukaEndEffectorIndex
    robot_id = env._kuka.kukaUid
    joint_positions = p.calculateInverseKinematics(robot_id,
                                                   kuka_end_effector_index,
                                                   desired_position,
                                                   desired_orientation)

    # Move the robot using the computed joint positions
    for i in range(len(joint_positions)):
        p.setJointMotorControl2(bodyIndex=robot_id,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_positions[i])

    # Step the simulation until the arm reaches the target
    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.01)  # Slow down simulation for visualization

def grasp_object(env):
    """
    Grasp the object after the arm reaches the target position.
    
    Args:
    - env: The simulation environment (Sim2SimEnv instance).
    """
    # Close the gripper to grasp
    finger_angle = 0.3
    for _ in range(100):
        grasp_action = [0, 0, 0, 0, finger_angle]
        env._kuka.applyAction(grasp_action)
        p.stepSimulation()
        finger_angle -= 0.3 / 100
        if finger_angle < 0:
            finger_angle = 0

    # Lift the object slightly
    lift_action = [0, 0, 0.1, 0, finger_angle]
    env._kuka.applyAction(lift_action)
    for _ in range(100):
        p.stepSimulation()

if __name__ == "__main__":
    # Load the trained detector
    detector_model = CubeDetectorVGG16()
    
    print("Loading model...")
    # detector_model.load_state_dict(torch.load("../models/cube_detector_vgg16_100_1.pt"))
    detector_model = torch.nn.DataParallel(detector_model)
    detector_model.load_state_dict(torch.load('../models/cube_detector_vgg16_100_1.pt'))

    print("Evaluating...")
    detector_model.eval()

    # Initialize the test environment
    env = Sim2SimEnv(renders=False)
    num_test_episodes = 60
    all_predictions = []
    all_ground_truths = []

    for episode in range(num_test_episodes):
        print(f"Episode: {episode}")
        obs = env.reset()
        motion_planner = MotionPlanner(env._kuka)
        done = False
        
        # while not done:
            # Get the RGB observation
        rgb_image = obs  # Assuming this is your RGB data in (3, H, W) format
        
        # Convert to PIL Image for your object detector
        image = Image.fromarray(np.transpose(rgb_image, (1, 2, 0)))  # Convert to (H, W, 3)
        
        # Preprocess and make prediction
        image_tensor = preprocess_image(image)  # Your preprocessing function here
        with torch.no_grad():
            prediction = detector_model(image_tensor)
            predicted_position = (prediction[0, 0].item(), prediction[0, 1].item(), prediction[0, 2].item())
            predicted_orientation = (prediction[0, 3].item(), prediction[0, 4].item(), prediction[0, 5].item(), prediction[0, 6].item())
            # predicted_pose = (predicted_position, predicted_orientation)
            predicted_pose = env.get_cube_position_world(predicted_position, predicted_orientation)
        
        # Ground truth position of the cube
        actual_pose= env.get_cube_position()
        # actual_pose = env.get_cube_position_relative_to_camera()
        # temp = env.get_cube_position_relative_to_camera()
        # predicted_pose = env.get_cube_position_world(temp[0], temp[1])
        
        # Store predictions and ground truth
        all_predictions.append(predicted_pose[0])
        all_ground_truths.append(actual_pose[0])
        
        print(f"{predicted_pose}    |    {actual_pose}")
        # print(env.get_dist())
        
        # dis, yaw, pit, ctp = env.get_cam()
        # p.resetDebugVisualizerCamera(dis, yaw, pit, ctp)
        
        # motion_planner.execute_grasp(predicted_pose[0], predicted_pose[1])
        # while True:
        #     pass
            
            # For visualization (optional)
            # visualize_prediction(predicted_position, actual_position)  # Implement this function

            # Step environment (if you want to interact)
            # action = [0, 0, 0]  # Example action, update accordingly
            # obs, _, done, _ = env.step(action)

    # Calculate Accuracy
    mse = np.mean((np.array(all_predictions) - np.array(all_ground_truths))**2)
    print(f"Mean Squared Error of Cube Position Prediction: {mse}")
    
    errors = []
    for pred, gt in zip(all_predictions, all_ground_truths):
        # Calculate Euclidean distance in meters
        distance_error_m = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2 + (pred[2] - gt[2])**2)
        # Convert to centimeters
        distance_error_cm = distance_error_m * 100
        errors.append(distance_error_cm)
    
    # Calculate and return mean error
    mean_error_cm = np.mean(errors)
    print(f"Mean Euclidean Error in Cube Position Prediction: {mean_error_cm:.2f} cm")
