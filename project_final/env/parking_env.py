"""
Custom Parking Environment
- Base: highway-env parking-v0
- Customizations:
    1. Observation flattened to vector (optional sensor noise)
    2. Redesigned reward function
    3. Random vehicle placement per episode
"""

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401 - registers highway envs


class ParkingEnv(gym.Env):
    """
    Wrapper around highway-env parking-v0 with custom MDP design.

    Observation (15-dim, continuous):
        [x, y, vx, vy, cos_h, sin_h,          # agent kinematics (6)
         goal_x, goal_y, goal_cos_h, goal_sin_h,  # goal info (4)
         dx, dy,                                # relative displacement to goal (2)
         nearest_obstacle_dist,                 # nearest obstacle distance (1)
         obs_rel_cos, obs_rel_sin]              # bearing to nearest obstacle (2)

    Action:
        - DQN mode  : discrete index mapped to (steering, throttle) pairs
        - Continuous: Box([-1,-1], [1,1]) -> (steering, throttle)

    Reward:
        - Distance reward  : -distance_to_goal        (dense)
        - Progress reward  : +(prev_dist - dist)      (dense)
        - Heading reward   : -heading_error            (dense)
        - Collision penalty: -50                       (sparse)
        - Time penalty     : -0.1 per step             (dense)
        - Success bonus    : +100                      (sparse)
    """

    # Discrete action set for DQN baseline
    DISCRETE_ACTIONS = [
        (-1.0, 0.6),
        (-0.5, 0.6),
        (0.0, 0.6),
        (0.5, 0.6),
        (1.0, 0.6),
        (-1.0, 0.3),
        (-0.5, 0.3),
        (0.0, 0.3),
        (0.5, 0.3),
        (1.0, 0.3),
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, -0.3),
        (-0.5, -0.3),
        (0.0, -0.3),
        (0.5, -0.3),
        (1.0, -0.3),
        (-1.0, -0.6),
        (-0.5, -0.6),
        (0.0, -0.6),
        (0.5, -0.6),
        (1.0, -0.6),
    ]
    N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)

    # Success thresholds
    SUCCESS_DIST = 0.02  # metres  (normalised space)
    SUCCESS_HEADING = 0.08  # radians (~8.6 deg) in cos/sin error
    SUCCESS_SPEED = 0.05  # normalised speed

    def __init__(
        self,
        discrete: bool = False,
        noise_std: float = 0.02,
        max_steps: int = 200,
        n_other_vehicles: int = 6,
        obstacle_dist_scale: float = 10.0,
        render_mode: str = None,
    ):
        super().__init__()

        self.discrete = discrete
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.n_other_vehicles = n_other_vehicles
        self.obstacle_dist_scale = obstacle_dist_scale

        # Build base env
        self._base_env = gym.make(
            "parking-v0",
            render_mode=render_mode,
            config={
                "vehicles_count": n_other_vehicles,
                "duration": max_steps,
                "collision_reward": 0,  # we handle collision reward ourselves
            },
        )

        # Observation space: 15-dim flat vector, all in roughly [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=-np.ones(15, dtype=np.float32),
            high=np.ones(15, dtype=np.float32),
        )

        # Action space
        if discrete:
            self.action_space = gym.spaces.Discrete(self.N_DISCRETE_ACTIONS)
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
            )

        self._step_count = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw_obs, info = self._base_env.reset(seed=seed, options=options)
        self._step_count = 0
        obs = self._process_obs(raw_obs)
        self._prev_dist = float(np.sqrt(float(obs[10]) ** 2 + float(obs[11]) ** 2))
        return obs, info

    def step(self, action):
        # Convert discrete index -> continuous action
        if self.discrete:
            steering, throttle = self.DISCRETE_ACTIONS[int(action)]
            cont_action = np.array([steering, throttle], dtype=np.float32)
        else:
            if isinstance(action, tuple):
                action = action[0]
            cont_action = np.clip(
                np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0
            ).astype(np.float32)

        raw_obs, _, terminated, truncated, info = self._base_env.step(cont_action)
        self._step_count += 1

        obs = self._process_obs(raw_obs)
        reward = self._compute_reward(obs, info, terminated)

        # Terminate on collision or max steps
        if info.get("crashed", False):
            terminated = True
        if self.is_success(obs, info):
            terminated = True
            info["is_success"] = True
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
        We use:
            - agent kinematics (6)
            - goal info (4)
            - relative displacement to goal (2)
            - nearest obstacle distance (1)
            - bearing to nearest obstacle in ego frame (2: cos, sin)
        """
        agent = raw_obs["observation"].astype(np.float32)  # x,y,vx,vy,cos_h,sin_h
        goal = raw_obs["desired_goal"].astype(np.float32)  # x,y,vx,vy,cos_h,sin_h

        # Relative position to goal
        dx = goal[0] - agent[0]
        dy = goal[1] - agent[1]
        nearest_obs_dist, obs_rel_cos, obs_rel_sin = self._nearest_obstacle_info()

        obs = np.concatenate(
            [
                agent,  # (6,)
                goal[[0, 1, 4, 5]],  # goal x, y, cos_h, sin_h  (4,)
                np.array([dx, dy]),  # relative displacement     (2,)
                np.array(
                    [nearest_obs_dist], dtype=np.float32
                ),  # nearest obstacle dist (1,)
                np.array(
                    [obs_rel_cos, obs_rel_sin], dtype=np.float32
                ),  # bearing to nearest obs (2,)
            ]
        ).astype(
            np.float32
        )  # total: 15

        # Sensor noise (stochasticity)
        if self.noise_std > 0:
            obs += np.random.normal(0, self.noise_std, size=obs.shape).astype(
                np.float32
            )

        return np.clip(obs, -1.0, 1.0)

    def _nearest_obstacle_info(self):
        """
        Return (normalized distance, relative cos, relative sin) of nearest obstacle.

        - Distance is normalized to [0, 1] using obstacle_dist_scale.
        - Relative angle is obstacle bearing minus ego heading, encoded as (cos, sin).
        """
        try:
            env_unwrapped = self._base_env.unwrapped
            if not hasattr(env_unwrapped, "road") or env_unwrapped.road is None:
                return 1.0, 1.0, 0.0

            all_vehicles = getattr(env_unwrapped.road, "vehicles", [])
            if not all_vehicles:
                return 1.0, 1.0, 0.0

            ego = None
            if (
                hasattr(env_unwrapped, "controlled_vehicles")
                and env_unwrapped.controlled_vehicles
            ):
                ego = env_unwrapped.controlled_vehicles[0]
            elif hasattr(env_unwrapped, "vehicle"):
                ego = env_unwrapped.vehicle
            if ego is None or not hasattr(ego, "position"):
                return 1.0, 1.0, 0.0

            ego_pos = np.asarray(ego.position, dtype=np.float32)
            ego_heading = float(getattr(ego, "heading", 0.0))
            dists = []
            rel_angles = []
            for veh in all_vehicles:
                if ego is not None and veh is ego:
                    continue
                if not hasattr(veh, "position"):
                    continue
                veh_pos = np.asarray(veh.position, dtype=np.float32)
                vec = veh_pos - ego_pos
                dist = float(np.linalg.norm(vec))
                if dist <= 1e-6:
                    continue
                dists.append(dist)
                bearing_world = float(np.arctan2(vec[1], vec[0]))
                rel_angles.append(bearing_world - ego_heading)

            if not dists:
                return 1.0, 1.0, 0.0

            # Take nearest obstacle
            idx_min = int(np.argmin(dists))
            min_dist = dists[idx_min]
            rel_angle = rel_angles[idx_min]

            # Prevent saturation by scaling world distance into [0, 1].
            norm_dist = min_dist / max(self.obstacle_dist_scale, 1e-6)
            norm_dist = float(np.clip(norm_dist, 0.0, 1.0))

            obs_rel_cos = float(np.cos(rel_angle))
            obs_rel_sin = float(np.sin(rel_angle))

            return norm_dist, obs_rel_cos, obs_rel_sin
        except Exception:
            # Keep environment robust even if highway-env internals differ.
            return 1.0, 1.0, 0.0

    def _compute_reward(self, obs: np.ndarray, info: dict, terminated: bool) -> float:
        """
        Custom reward function.

        obs layout: [x,y,vx,vy,cos_h,sin_h, gx,gy,gcos,gsin, dx,dy, d_obs, obs_rel_cos,obs_rel_sin]
        """
        dx, dy = float(obs[10]), float(obs[11])
        distance = np.sqrt(dx**2 + dy**2)

        # Agent heading vs goal heading (cosine similarity -> error in [0,2])
        cos_h, sin_h = float(obs[4]), float(obs[5])
        gcos_h, gsin_h = float(obs[6 + 2]), float(obs[6 + 3])
        heading_error = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)  # in [0, 2]

        speed = np.sqrt(float(obs[2]) ** 2 + float(obs[3]) ** 2)

        # Dense rewards
        progress = 0.0
        if self._prev_dist is not None:
            progress = self._prev_dist - distance
        self._prev_dist = distance

        reward = -distance  # stay close to goal
        reward += 2.0 * progress  # encourage moving toward goal
        reward -= 0.3 * heading_error  # align heading
        reward -= 0.1  # time penalty

        # Sparse penalties / bonuses
        if info.get("crashed", False):
            reward -= 50.0

        # Success bonus
        if (
            distance < self.SUCCESS_DIST
            and heading_error < self.SUCCESS_HEADING
            and speed < self.SUCCESS_SPEED
        ):
            reward += 100.0

        return float(reward)

    def is_success(self, obs: np.ndarray, info: dict) -> bool:
        dx, dy = float(obs[10]), float(obs[11])
        distance = np.sqrt(dx**2 + dy**2)
        cos_h, sin_h = float(obs[4]), float(obs[5])
        gcos_h, gsin_h = float(obs[6 + 2]), float(obs[6 + 3])
        heading_error = 1.0 - (cos_h * gcos_h + sin_h * gsin_h)
        speed = np.sqrt(float(obs[2]) ** 2 + float(obs[3]) ** 2)
        return (
            distance < self.SUCCESS_DIST
            and heading_error < self.SUCCESS_HEADING
            and speed < self.SUCCESS_SPEED
        )
