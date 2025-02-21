from typing import Any, Dict, Union

import numpy as np
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from custom_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

import pybullet as p


@register_env("Sim2SimEnv", max_episode_steps=50)
class Sim2SimEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    cube_half_size = 0.075
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.27680088, 0.43632513, 1.36903757], target=[0.23, 0.2, 0.54])
        return [CameraConfig("base_camera", pose, 128, 128, 20., 0.01, 10)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube"
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            # xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            # xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            # self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            xyz = torch.tensor([[0.6, 0.0, -0.1]])
            qs = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
    
    def quaternion_to_matrix(self, quaternion):
        """
        Convert a quaternion into a 3x3 rotation matrix.
        
        Args:
            quaternion: A list or array [qx, qy, qz, qw].
        
        Returns:
            A 3x3 rotation matrix.
        """
        qx, qy, qz, qw = quaternion
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])
        
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions.

        Args:
            q1: The first quaternion [qx, qy, qz, qw].
            q2: The second quaternion [qx, qy, qz, qw].

        Returns:
            A quaternion representing the product of q1 and q2.
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ]

    def get_cube_position(self):
        position = self.cube.pose.p.tolist()[0]
        orientation = self.cube.pose.q.tolist()[0]
        return (position, orientation)
    
    def get_cube_position_from_camera(self, camera_coordinates):
        position_camera, orientation_camera = camera_coordinates
        
        # Camera pose in the world frame
        camera_position_world = self._default_sensor_configs[0].pose.p.tolist()[0]
        camera_orientation_world = self._default_sensor_configs[0].pose.q.tolist()[0]
        camera_orientation_world = (0.25224688681483054, -0.39594830010602733, 0.7446704469182975, -0.4744073958338725)

        # Convert camera-to-world transformation
        camera_rotation_world = self.quaternion_to_matrix(camera_orientation_world)
        camera_translation_world = camera_position_world

        camera_to_world_transform = np.eye(4)
        camera_to_world_transform[:3, :3] = camera_rotation_world
        camera_to_world_transform[:3, 3] = camera_translation_world

        # Convert cube's camera pose to world frame
        cube_position_camera_homogeneous = np.hstack((position_camera, [1]))
        cube_position_world = camera_to_world_transform @ cube_position_camera_homogeneous

        # Transform orientation (quaternion)
        cube_orientation_world = self.quaternion_multiply(
            camera_orientation_world, orientation_camera
        )
        
        world_position, world_orientation = p.multiplyTransforms(
          camera_position_world, camera_orientation_world,
          position_camera, orientation_camera
        )

        return cube_position_world[:3].tolist(), cube_orientation_world
    
    def get_dist(self):
        camera_position_world = self._default_sensor_configs[0].pose.p.tolist()[0]
        camera_orientation_world = self._default_sensor_configs[0].pose.q.tolist()[0]

        cpos, cori = self.get_cube_position()
        
        dist = np.linalg.norm(np.array(camera_position_world) - np.array(cpos))
        print(f"DIST = {dist}")
        
        cube_half_size = self.cube_half_size  # Defined during cube creation
        cube_full_size = cube_half_size * 2  # Full size
        print(f"Cube full size in ManiSkill: {cube_full_size}")