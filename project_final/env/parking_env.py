"""
Custom Parking Environment
- Base: highway-env parking-v0
- Customizations:
    1. Observation flattened to vector + sensor noise
    2. Redesigned reward function
    3. Random vehicle placement per episode
"""

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 - registers highway envs


class ParkingEnv(gym.Env):
    """
    Wrapper around highway-env parking-v0 with custom MDP design.

    Observation (12-dim, continuous):
        [x, y, vx, vy, cos_h, sin_h,          # agent kinematics (6)
         goal_x, goal_y, goal_cos_h, goal_sin_h,  # goal info (4)
         dx, dy]                                # relative displacement to goal (2)

    Action:
        - DQN mode  : discrete index mapped to (steering, throttle) pairs
        - Continuous: Box([-1,-1], [1,1]) -> (steering, throttle)

    Reward:
        - Distance reward  : -distance_to_goal        (dense)
        - Heading reward   : -heading_error            (dense)
        - Collision penalty: -50                       (sparse)
        - Time penalty     : -0.1 per step             (dense)
        - Success bonus    : +100                      (sparse)
    """

    # Discrete action set for DQN baseline
    DISCRETE_ACTIONS = [
        (-1.0,  0.6), (-0.5,  0.6), (0.0,  0.6), (0.5,  0.6), (1.0,  0.6),
        (-1.0,  0.3), (-0.5,  0.3), (0.0,  0.3), (0.5,  0.3), (1.0,  0.3),
        (-1.0,  0.0), (              0.0,  0.0),               (1.0,  0.0),
        (-1.0, -0.3), (-0.5, -0.3), (0.0, -0.3), (0.5, -0.3), (1.0, -0.3),
        (-1.0, -0.6), (-0.5, -0.6), (0.0, -0.6), (0.5, -0.6), (1.0, -0.6),
    ]
    N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)

    # Success thresholds
    SUCCESS_DIST    = 0.15   # metres  (normalised space)
    SUCCESS_HEADING = 0.15   # radians (~8.6 deg) in cos/sin error
    SUCCESS_SPEED   = 0.05   # normalised speed

    def __init__(
        self,
        discrete: bool = False,
        noise_std: float = 0.02,
        max_steps: int = 200,
        n_other_vehicles: int = 6,
        render_mode: str = None,
    ):
        super().__init__()

        self.discrete       = discrete
        self.noise_std      = noise_std
        self.max_steps      = max_steps
        self.n_other_vehicles = n_other_vehicles

        # Build base env
        self._base_env = gym.make(
            "parking-v0",
            render_mode=render_mode,
            config={
                "vehicles_count": n_other_vehicles,
                "duration": max_steps,
                "collision_reward": 0,   # we handle collision reward ourselves
            },
        )

        # Observation space: 12-dim flat vector, all in roughly [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=-np.ones(12, dtype=np.float32),
            high=np.ones(12, dtype=np.float32),
        )

        # Action space
        if discrete:
            self.action_space = gym.spaces.Discrete(self.N_DISCRETE_ACTIONS)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([ 1.0,  1.0], dtype=np.float32),
            )

        self._step_count = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs, info = self._base_env.reset(seed=seed, options=options)
        self._step_count = 0
        self._prev_dist  = None
        obs = self._process_obs(raw_obs)
        return obs, info

    def step(self, action):
        # Convert discrete index -> continuous action
        if self.discrete:
            steering, throttle = self.DISCRETE_ACTIONS[int(action)]
            cont_action = np.array([steering, throttle], dtype=np.float32)
        else:
            cont_action = np.clip(action, -1.0, 1.0).astype(np.float32)

        raw_obs, _, terminated, truncated, info = self._base_env.step(cont_action)
        self._step_count += 1

        obs    = self._process_obs(raw_obs)
        reward = self._compute_reward(obs, info, terminated)

        # Terminate on collision or max steps
        if info.get("crashed", False):
            terminated = True
        if self._step_count >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._base_env.render()

    def close(self):
        self._base_env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_obs(self, raw_obs: dict) -> np.ndarray:
        """
        Flatten dict observation and add sensor noise.

        raw_obs keys: 'observation' (6,), 'achieved_goal' (6,), 'desired_goal' (6,)
        We use: agent kinematics (6) + goal (4) + relative displacement (2)
        """
        agent = raw_obs["observation"].astype(np.float32)   # x,y,vx,vy,cos_h,sin_h
        goal  = raw_obs["desired_goal"].astype(np.float32)  # x,y,vx,vy,cos_h,sin_h

        # Relative position to goal
        dx = goal[0] - agent[0]
        dy = goal[1] - agent[1]

        obs = np.concatenate([
            agent,                    # (6,)
            goal[[0, 1, 4, 5]],       # goal x, y, cos_h, sin_h  (4,)
            np.array([dx, dy]),       # relative displacement     (2,)
        ]).astype(np.float32)         # total: 12

        # Sensor noise (stochasticity)
        if self.noise_std > 0:
            obs += np.random.normal(0, self.noise_std, size=obs.shape).astype(np.float32)

        return np.clip(obs, -1.0, 1.0)

    def _compute_reward(self, obs: np.ndarray, info: dict, terminated: bool) -> float:
        """
        Custom reward function.

        obs layout: [x,y,vx,vy,cos_h,sin_h, gx,gy,gcos,gsin, dx,dy]
        """
        dx, dy   = float(obs[10]), float(obs[11])
        distance = np.sqrt(dx**2 + dy**2)

        # Agent heading vs goal heading (cosine similarity -> error in [0,2])
        cos_h, sin_h   = float(obs[4]), float(obs[5])
        gcos_h, gsin_h = float(obs[6+2]), float(obs[6+3])
        heading_error  = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)  # in [0, 2]

        speed = np.sqrt(float(obs[2])**2 + float(obs[3])**2)

        # Dense rewards
        reward  = -distance                    # approach goal
        reward -= 0.3 * heading_error          # align heading
        reward -= 0.1                          # time penalty

        # Sparse penalties / bonuses
        if info.get("crashed", False):
            reward -= 50.0

        # Success bonus
        if (distance < self.SUCCESS_DIST
                and heading_error < self.SUCCESS_HEADING
                and speed < self.SUCCESS_SPEED):
            reward += 100.0

        return float(reward)

    def is_success(self, obs: np.ndarray, info: dict) -> bool:
        dx, dy = float(obs[10]), float(obs[11])
        distance = np.sqrt(dx**2 + dy**2)
        cos_h, sin_h   = float(obs[4]),    float(obs[5])
        gcos_h, gsin_h = float(obs[6+2]),  float(obs[6+3])
        heading_error  = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)
        speed = np.sqrt(float(obs[2])**2 + float(obs[3])**2)
        return (distance < self.SUCCESS_DIST
                and heading_error < self.SUCCESS_HEADING
                and speed < self.SUCCESS_SPEED)
