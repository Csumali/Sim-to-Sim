import time
import torch
import numpy as np
import gymnasium as gym
from PIL import Image
import mani_skill.envs
import sim2sim_env
from torchvision import transforms
from vgg16 import CubeDetectorVGG16  # Import your trained VGG detector
# from motion_planner import MotionPlanner  # Import your motion planner

# Load the trained object detector
detector_model = CubeDetectorVGG16()
detector_model = torch.nn.DataParallel(detector_model)
detector_model.load_state_dict(torch.load("../models/cube_detector_vgg16_100_1.pt"))
detector_model.eval()  # Set model to evaluation mode

# Define image transforms for object detection
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize the ManiSkill environment
env = gym.make("Sim2SimEnv", sensor_configs=dict(width=48, height=48), obs_mode="rgb", control_mode="pd_joint_delta_pos", render_mode="human")  # Change according to your task

# motion_planner = MotionPlanner()  # Assuming this initializes the planner for ManiSkill

# Testing loop
num_test_episodes = 100
all_predictions = []
all_ground_truths = []

for episode in range(num_test_episodes):
    print(f"Episode: {episode + 1}/{num_test_episodes}")
    obs, _ = env.reset()
    
    # Extract RGB observation from ManiSkill's observation (change as needed)
    rgb_image = obs["sensor_data"]["base_camera"]["rgb"]  # Assuming ManiSkill provides RGB in this key
    
    # Assuming rgb_image is a PyTorch tensor in (C, H, W) format
    rgb_image_np = rgb_image.squeeze(0).cpu().numpy() # Convert to NumPy array
    
    # Convert RGB to PIL Image and preprocess
    image = Image.fromarray(rgb_image_np)  # Convert (H, W, C) to PIL image
    # image.save(f"data/mani_cam{episode}.png")
    image_tensor = transform(image).unsqueeze(0)  # Preprocess for the detector
    
    # Predict cube position using the detector model
    with torch.no_grad():
        prediction = detector_model(image_tensor).cpu().numpy()[0]
        predicted_position = (prediction[0].item(), prediction[1].item(), prediction[2].item())
        predicted_orientation = (prediction[3].item(), prediction[4].item(), prediction[5].item(), prediction[6].item())
        predicted_position = env.get_cube_position_from_camera((predicted_position, predicted_orientation))
    
    # Get the ground truth position if available (for comparison)
    actual_position = env.get_cube_position()  # Assumes you have a function to retrieve ground truth position
    
    print(f"{predicted_position}    |    {actual_position}")

    # Store predictions and ground truth for evaluation
    all_predictions.append(predicted_position[0])
    all_ground_truths.append(actual_position[0])
    
    # env.get_dist()
    
    # while True:
    #     env.render()

        # Use the motion planner to move the robot to the predicted position
        # motion_planner.execute_grasp(predicted_position)

        # Step the environment forward (take action)
        # obs, reward, done, info = env.step([0] * env.action_space.shape[0])  # Update with actual action if needed

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

# Close the environment after testing
env.close()
