import time
import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCube-v1",
  sensor_configs=dict(width=48, height=48),
  human_render_camera_configs=dict(shader_pack="rt"),
  viewer_camera_configs=dict(fov=1),
  render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset() # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()  # a display is required to render
    time.sleep(1./4.)
env.close()