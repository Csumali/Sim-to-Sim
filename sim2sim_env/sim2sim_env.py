from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import random
import os
from gym import spaces
import time
import pybullet as p
import kuka
import numpy as np
import pybullet_data
import pdb
import distutils.dir_util
import glob
from pkg_resources import parse_version
import gym


class Sim2SimEnv(KukaGymEnv):
  """Class for Kuka environment with diverse objects.

  In each episode some objects are chosen from a set of 1000 diverse objects.
  These 1000 objects are split 90/10 into a train and test set.
  """

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=80,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=8,
               dv=0.06,
               removeHeightHack=False,
               blockRandom=0.3,
               cameraRandom=1,
               width=48,
               height=48,
               numObjects=10,
               targetObject="cube_small",
               isTest=False):
    """Initializes the KukaDiverseObjectEnv.

    Args:
      urdfRoot: The diretory from which to load environment URDF's.
      actionRepeat: The number of simulation steps to apply for each action.
      isEnableSelfCollision: If true, enable self-collision.
      renders: If true, render the bullet GUI.
      isDiscrete: If true, the action space is discrete. If False, the
        action space is continuous.
      maxSteps: The maximum number of actions per episode.
      dv: The velocity along each dimension for each action.
      removeHeightHack: If false, there is a "height hack" where the gripper
        automatically moves down for each action. If true, the environment is
        harder and the policy chooses the height displacement.
      blockRandom: A float between 0 and 1 indicated block randomness. 0 is
        deterministic.
      cameraRandom: A float between 0 and 1 indicating camera placement
        randomness. 0 is deterministic.
      width: The image width.
      height: The observation image height.
      numObjects: The number of objects in the bin.
      isTest: If true, use the test set of objects. If false, use the train
        set of objects.
    """

    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._dv = dv
    self._p = p
    self._removeHeightHack = removeHeightHack
    self._blockRandom = blockRandom
    self._cameraRandom = cameraRandom
    self._width = width
    self._height = height
    self._numObjects = numObjects
    self._target_object = os.path.join(self._urdfRoot, f"{targetObject}.urdf")
    self._isTest = isTest

    if self._renders:
      self.cid = p.connect(p.SHARED_MEMORY)
      if (self.cid < 0):
        self.cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      self.cid = p.connect(p.DIRECT)
    self.seed()

    if (self._isDiscrete):
      if self._removeHeightHack:
        self.action_space = spaces.Discrete(9)
      else:
        self.action_space = spaces.Discrete(7)
    else:
      self.action_space = spaces.Box(low=-1, high=1, shape=(3,))  # dx, dy, da
      if self._removeHeightHack:
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))  # dx, dy, dz, da
    self.observation_space = spaces.Box(low=0, high=255, shape=(3, self._height, self._width), dtype=np.uint8)
    self.viewer = None

  def reset(self):
    """Environment reset called at the beginning of an episode.
    """
    # Set the camera settings.
    self.look = [0.23, 0.2, 0.54]
    self.distance = 1.
    self.pitch = -56 + self._cameraRandom * np.random.uniform(-3, 3)
    self.yaw = 245 + self._cameraRandom * np.random.uniform(-3, 3)
    self.roll = 0
    self._view_matrix = p.computeViewMatrixFromYawPitchRoll(self.look, self.distance, self.yaw, self.pitch, self.roll, 2)
    fov = 20. + self._cameraRandom * np.random.uniform(-2, 2)
    self.fov = fov
    aspect = self._width / self._height
    near = 0.01
    far = 10
    self._proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    self._attempted_grasp = False
    self._env_step = 0
    self.terminated = 0

    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    self._floorUid = p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    self._tableUid = p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                                0.000000, 0.000000, 0.0, 1.0)

    p.setGravity(0, 0, -10)
    self._kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self.randomize_scene_textures()
    self._envStepCounter = 0
    p.stepSimulation()

    # Choose the objects in the bin.
    num_distractors = random.randint(0, self._numObjects - 1)
    distractor_urdfList = self._get_random_object(num_distractors, self._isTest)
    self._objectUids = self._randomly_place_objects([self._target_object] + distractor_urdfList)
    self._randomize_object_properties()
    self._observation = self._get_observation()
    return self._observation

  def _randomly_place_objects(self, urdfList):
    """Randomly places the objects in the bin.

    Args:
      urdfList: The list of urdf files to place in the bin.

    Returns:
      The list of object unique ID's.
    """

    # Randomize positions of each object urdf.
    objectUids = []
    cube_position = [0, 0]
    for i in range(len(urdfList)):
      urdf_name = urdfList[i]
      max_attempts = 10
      for attempt in range(max_attempts):
        xpos = 0.4 + self._blockRandom * random.random()
        ypos = self._blockRandom * (random.random() - .5)
        if i > 0:
          distance = np.linalg.norm(np.array([xpos, ypos]) - np.array(cube_position))
          if distance > 0.15:
            break
        else:
          break
      else:
        print(f"Failed to place {urdf_name} after {max_attempts} attempts, skipping.")
        continue
      angle = np.pi / 2 + self._blockRandom * np.pi * random.random()
      orn = p.getQuaternionFromEuler([0, 0, angle])
      urdf_path = os.path.join(self._urdfRoot, urdf_name)
      uid = p.loadURDF(urdf_path, [xpos, ypos, .1], [orn[0], orn[1], orn[2], orn[3]])
      objectUids.append(uid)
      for _ in range(500):
        p.stepSimulation()
      if i == 0:
        cube_position = [xpos, ypos]
    return objectUids

  def _get_observation(self):
    """Return the observation as an image.
    """
    light_direction, light_color, light_distance, light_specular = self.randomize_lighting()
    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=self._view_matrix,
                               projectionMatrix=self._proj_matrix,
                              lightDirection=light_direction,
                              lightColor=light_color,
                              lightDistance=light_distance,
                              shadow=1,
                              renderer=p.ER_TINY_RENDERER)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    np_img_arr = np_img_arr[:, :, :3]
    np_img_arr = np.transpose(np_img_arr, (2, 0, 1))
    
    # Add random Gaussian noise
    noise = np.random.normal(0, 0.05, np_img_arr.shape)
    np_img_arr = np.clip(np_img_arr + noise, 0, 255).astype(np.uint8)
    
    return np_img_arr

  def step(self, action):
    """Environment step.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    dv = self._dv  # velocity per physics step.
    if self._isDiscrete:
      # Static type assertion for integers.
      assert isinstance(action, int)
      if self._removeHeightHack:
        dx = [0, -dv, dv, 0, 0, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0, 0, 0][action]
        dz = [0, 0, 0, 0, 0, -dv, dv, 0, 0][action]
        da = [0, 0, 0, 0, 0, 0, 0, -0.25, 0.25][action]
      else:
        dx = [0, -dv, dv, 0, 0, 0, 0][action]
        dy = [0, 0, 0, -dv, dv, 0, 0][action]
        dz = -dv
        da = [0, 0, 0, 0, 0, -0.25, 0.25][action]
    else:
      dx = dv * action[0]
      dy = dv * action[1]
      if self._removeHeightHack:
        dz = dv * action[2]
        da = 0.25 * action[3]
      else:
        dz = -dv
        da = 0.25 * action[2]

    return self._step_continuous([dx, dy, dz, da, 0.3])

  def _step_continuous(self, action):
    """Applies a continuous velocity-control action.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # Perform commanded action.
    self._env_step += 1
    self._kuka.applyAction(action)
    for _ in range(self._actionRepeat):
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      if self._termination():
        break

    # If we are close to the bin, attempt grasp.
    state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    end_effector_pos = state[0]
    if end_effector_pos[2] <= 0.1:
      finger_angle = 0.3
      for _ in range(500):
        grasp_action = [0, 0, 0, 0, finger_angle]
        self._kuka.applyAction(grasp_action)
        p.stepSimulation()
        #if self._renders:
        #  time.sleep(self._timeStep)
        finger_angle -= 0.3 / 100.
        if finger_angle < 0:
          finger_angle = 0
      for _ in range(500):
        grasp_action = [0, 0, 0.001, 0, finger_angle]
        self._kuka.applyAction(grasp_action)
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)
        finger_angle -= 0.3 / 100.
        if finger_angle < 0:
          finger_angle = 0
      self._attempted_grasp = True
    observation = self._get_observation()
    done = self._termination()
    reward = self._reward()

    debug = {'grasp_success': self._graspSuccess}
    return observation, reward, done, debug

  def _reward(self):
    """Calculates the reward for the episode.

    The reward is 1 if one of the objects is above height .2 at the end of the
    episode.
    """
    reward = 0
    self._graspSuccess = 0
    for uid in self._objectUids:
      pos, _ = p.getBasePositionAndOrientation(uid)
      # If any block is above height, provide reward.
      if pos[2] > 0.2:
        if uid == self._objectUids[0]:
          self._graspSuccess += 1
          reward = 1
        else:
          reward = -1
        break
    return reward

  def _termination(self):
    """Terminates the episode if we have tried to grasp or if we are above
    maxSteps steps.
    """
    return self._attempted_grasp or self._env_step >= self._maxSteps

  def _get_random_object(self, num_objects, test):
    """Randomly choose an object urdf from the random_urdfs directory.

    Args:
      num_objects:
        Number of graspable objects.

    Returns:
      A list of urdf filenames.
    """
    if test:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*0/*.urdf')
    else:
      urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/*[1-9]/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    total_num_objects = len(found_object_directories)
    selected_objects = np.random.choice(np.arange(total_num_objects), num_objects)
    selected_objects_filenames = []
    for object_index in selected_objects:
      selected_objects_filenames += [found_object_directories[object_index]]
    return selected_objects_filenames
  
  def _randomize_object_properties(self):
    """Randomizes the mass, friction, and textures of the objects."""
    for uid in self._objectUids:
        # Randomize mass and friction
        random_mass = random.uniform(0.5, 2.0)
        random_friction = random.uniform(0.3, 1.0)
        p.changeDynamics(uid, -1, mass=random_mass, lateralFriction=random_friction)
        
        # Randomize texture (color, gradient, checker pattern)
        texture_type = random.choice(["rgb", "gradient", "checker"])
        if texture_type == "rgb":
            color = [random.random(), random.random(), random.random(), 1]
        elif texture_type == "gradient":
            color = [random.random(), random.random(), random.random(), 1]  # Implement gradient
        elif texture_type == "checker":
            color = [random.random(), random.random(), random.random(), 1]  # Checker pattern logic

        p.changeVisualShape(uid, -1, rgbaColor=color)

  def randomize_scene_textures(self):
    """Randomizes the textures of the table, floor, skybox, and robot."""
    # Table
    p.changeVisualShape(self._tableUid, -1, rgbaColor=[random.random(), random.random(), random.random(), 1])
    
    # Floor
    p.changeVisualShape(self._floorUid, -1, rgbaColor=[random.random(), random.random(), random.random(), 1])
    
    # Robot (you can randomize Kuka textures similarly)
    p.changeVisualShape(self._kuka.kukaUid, -1, rgbaColor=[random.random(), random.random(), random.random(), 1])
    
  def randomize_lighting(self):
      light_direction = [
          np.random.uniform(-1, 1),  # Random direction x
          np.random.uniform(-1, 1),  # Random direction y
          np.random.uniform(-1, 1)   # Random direction z
      ]
      light_color = [
          np.random.uniform(0.5, 1),  # Random color intensity for red
          np.random.uniform(0.5, 1),  # Random color intensity for green
          np.random.uniform(0.5, 1)   # Random color intensity for blue
      ]
      light_distance = np.random.uniform(1, 3)      # Random distance
      light_specular = [
          np.random.uniform(0, 1),   # Random specular reflection for red
          np.random.uniform(0, 1),   # Random specular reflection for green
          np.random.uniform(0, 1)    # Random specular reflection for blue
      ]
      return light_direction, light_color, light_distance, light_specular

  def get_cube_position(self):
    position, orientation = p.getBasePositionAndOrientation(self._objectUids[0])
    visual_data = p.getVisualShapeData(self._objectUids[0])
    
    # Extract dimensions (Assumes only one visual shape is associated with the cube)
    size = None
    if visual_data:
        # The 'dimensions' are stored in the [3] index of the tuple in visual_data
        size = visual_data[0][3]  # (length, width, height) of the bounding box
    
    print("SIZEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    print(size)
    return position, orientation

  def get_cube_position_relative_to_camera(self):
      objectPos, objectOrn = self.get_cube_position()

      # Camera parameters
      yaw, pitch, roll = np.radians([self.yaw, self.pitch, self.roll])
      direction = np.array([
          -np.cos(pitch) * np.sin(yaw),
          np.cos(pitch) * np.cos(yaw),
          np.sin(pitch)
      ])
      camera_position = self.look - direction * self.distance
      camera_rotation = p.getQuaternionFromEuler([np.radians(self.pitch), np.radians(self.roll), np.radians(self.yaw)])

      # Invert camera transformation
      inv_camera_position, inv_camera_orientation = p.invertTransform(camera_position, camera_rotation)

      # Transform the cube's position and orientation relative to the camera
      relative_position, relative_orientation = p.multiplyTransforms(
          inv_camera_position, inv_camera_orientation,
          objectPos, objectOrn
      )
      return relative_position, relative_orientation

  def get_cube_position_world(self, relative_position, relative_orientation):
      yaw, pitch, roll = np.radians([self.yaw, self.pitch, self.roll])
      direction = np.array([
          -np.cos(pitch) * np.sin(yaw),
          np.cos(pitch) * np.cos(yaw),
          np.sin(pitch)
      ])
      camera_position = self.look - direction * self.distance
      camera_rotation = p.getQuaternionFromEuler([np.radians(self.pitch), np.radians(self.roll), np.radians(self.yaw)])

      world_position, world_orientation = p.multiplyTransforms(
          camera_position, camera_rotation,
          relative_position, relative_orientation
      )
      
      return world_position, world_orientation

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _reset = reset
    _step = step
    
  def get_cam(self):
    return self.distance, self.yaw, self.pitch, self.look
  
  def get_dist(self):
    yaw, pitch, roll = np.radians([self.yaw, self.pitch, self.roll])
    direction = np.array([
        -np.cos(pitch) * np.sin(yaw),
        np.cos(pitch) * np.cos(yaw),
        np.sin(pitch)
    ])
    camera_position = self.look - direction * self.distance
    camera_rotation = p.getQuaternionFromEuler([np.radians(self.pitch), np.radians(self.roll), np.radians(self.yaw)])
    
    cpos, cori = self.get_cube_position()
    
    dist = np.linalg.norm(np.array(camera_position) - np.array(self.look))
    print(f"DIST = {dist}")
    
    collision_shape_data = p.getCollisionShapeData(self._objectUids[0], -1)
    for shape in collision_shape_data:
      if shape[2] == p.GEOM_BOX:  # Check if it's a box shape
          cube_half_extents = shape[3]  # Half extents of the box
          cube_full_size = [dim * 2 for dim in cube_half_extents]
          print(f"Cube dimensions in PyBullet: {cube_full_size}")