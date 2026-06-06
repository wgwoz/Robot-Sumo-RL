import math
import os

import numpy as np
import pygame
import torch

from src.env.collisions import check_sat_collision, get_robot_global_velocity
from src.env.config import *
from src.env.renderer import SumoRenderer
from src.env.robot import SumoRobot


class SumoEnv:
    def __init__(self, render_mode=False, render_vectors=False):
        self.render_mode = render_mode
        self.render_vectors = render_vectors
        self.ARENA_RADIUS = ARENA_RADIUS
        self.center_x, self.center_y = WIDTH // 2, HEIGHT // 2
        pygame.display.set_caption("Robot-Sumo-RL: Cross Play")

        self.screen = None
        self.renderer = None

        self.has_collision_occurred = False

        if self.render_mode:
            if "SDL_VIDEODRIVER" not in os.environ and os.name == "posix":
                os.environ["SDL_AUDIODRIVER"] = "dummy"

            pygame.display.init()
            pygame.font.init()

            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

            self.renderer = SumoRenderer(
                self.screen,
                pygame.font.SysFont("Consolas", 14, bold=True),
                pygame.font.SysFont("Consolas", 16, bold=True),
                show_trails=self.render_vectors,
                show_sensors=self.render_vectors,
                show_ui=True,
            )
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, randPositions=True):
        if self.renderer:
            self.renderer.clear_trails()

        r1_cfg, r2_cfg = self._generate_start_positions(randPositions)

        self.robot1 = SumoRobot(x=r1_cfg["x"], y=r1_cfg["y"], angle=r1_cfg["angle"], mass=ROBOT_MASS)
        self.robot2 = SumoRobot(x=r2_cfg["x"], y=r2_cfg["y"], angle=r2_cfg["angle"], mass=ROBOT_MASS)
        self.robots = [self.robot1, self.robot2]
        self.done = False

        self.has_collision_occurred = False
        self.last_action1 = np.zeros(2)
        self.last_action2 = np.zeros(2)

        return self._get_all_state_vecs()

    def step(self, action1, action2):
        self.last_action1 = action1
        self.last_action2 = action2

        for r, action in zip(self.robots, [action1, action2]):
    # Convert tensors to simple float numbers
    # This ensures the physics engine is working with numbers, not tensors
            a_l = float(action[0]) if torch.is_tensor(action[0]) else action[0]
            a_r = float(action[1]) if torch.is_tensor(action[1]) else action[1]
            r.compute_dynamics(a_l, a_r)
            r.compute_kinematics(dt=1.0)

        is_collision = self._handle_collisions()

        if is_collision:
            self.has_collision_occurred = True

        state_vecs, rewards, done, info = self._calculate_env_logic()

        info["is_collision"] = is_collision
        info["has_collision"] = self.has_collision_occurred

        return state_vecs, rewards, done, info

    def _handle_collisions(self):
        overlap_info = check_sat_collision(
            self.robot1.get_corners(), self.robot2.get_corners()
        )
        if overlap_info:
            overlap, axis = overlap_info
            dir_vec = np.array(
                [self.robot2.x - self.robot1.x, self.robot2.y - self.robot1.y]
            )
            if np.dot(dir_vec, axis) < 0:
                axis = -axis

            m1, m2 = self.robot1.mass, self.robot2.mass
            total_mass = m1 + m2

            # --- 1. Position Correction (Prevent Overlap) ---
            push = axis * (overlap + 0.1)
            self.robot1.x -= push[0] * (m2 / total_mass)
            self.robot1.y -= push[1] * (m2 / total_mass)
            self.robot2.x += push[0] * (m1 / total_mass)
            self.robot2.y += push[1] * (m1 / total_mass)

            # --- 2. Velocity Analysis ---
            gv1 = get_robot_global_velocity(self.robot1)
            gv2 = get_robot_global_velocity(self.robot2)
            v_rel_normal = np.dot(gv1 - gv2, axis)

            if v_rel_normal > 0:
                restitution = 0.05
                impulse_mag = (1 + restitution) * v_rel_normal / (1 / m1 + 1 / m2)
                impulse_vec = impulse_mag * axis

                # Calculate direction-based multipliers
                forward1 = np.array([math.cos(math.radians(self.robot1.angle)), math.sin(math.radians(self.robot1.angle))])
                cos_angle1 = np.dot(axis, forward1)
                if cos_angle1 > 0.5:
                    mult1 = PUSH_BACK_MULT
                elif cos_angle1 < -0.5:
                    mult1 = PUSH_FRONT_MULT
                else:
                    mult1 = PUSH_SIDE_MULT

                forward2 = np.array([math.cos(math.radians(self.robot2.angle)), math.sin(math.radians(self.robot2.angle))])
                cos_angle2 = np.dot(axis, forward2)
                if cos_angle2 > 0.5:
                    mult2 = PUSH_BACK_MULT
                elif cos_angle2 < -0.5:
                    mult2 = PUSH_FRONT_MULT
                else:
                    mult2 = PUSH_SIDE_MULT

                # BREAKING STATIC FRICTION LOGIC
                # Define a threshold (adjust this in config.py)
                IMPACT_THRESHOLD = 0.8 

                if impulse_mag > IMPACT_THRESHOLD:
                    # If the hit is hard enough, the robots "lose their footing"
                    # We reduce the dampening so the impulse carries them further
                    stability_multiplier = 0.8  # Less dampening = more sliding
                else:
                    stability_multiplier = 0.3  # High dampening = stable (current behavior)

                self.robot1.apply_impulse(-impulse_vec * mult1)
                self.robot2.apply_impulse(impulse_vec * mult2)

                # Apply the dynamic stability based on impact
                self.robot1.v_side *= stability_multiplier
                self.robot2.v_side *= stability_multiplier
                self.robot1.omega *= (stability_multiplier + 0.1)
                self.robot2.omega *= (stability_multiplier + 0.1)

                # --- 4. FRICTION BREAK (The la pièce de résistance) ---
                # If the impact is violent enough (e.g. > 0.5 m/s), 
                # both robots lose their static grip and enter dynamic slip.
                if v_rel_normal > 0.9:
                    # We set a flag on the robots. 
                    # The robot.py compute_dynamics will check this flag.
                    self.robot1.is_slipping = True 
                    self.robot2.is_slipping = True
                # --- 5. Soften the Damping ---
                # Original code used 0.2 (killed 80% of velocity). 
                # We use 0.6 (kills 40%) to let the momentum carry the robot.
                self.robot1.v_side *= 0.6
                self.robot2.v_side *= 0.6
                self.robot1.omega *= 0.7
                self.robot2.omega *= 0.7

            return True
        return False


    def _generate_start_positions(self, randPositions):
        dist = self.ARENA_RADIUS * 0.7
        line_angle_deg = np.random.uniform(-180, 180) if randPositions else 0.0
        rad = np.radians(line_angle_deg)

        off_x = dist * np.cos(rad)
        off_y = dist * np.sin(rad)

        r1_x = -off_x
        r1_y = -off_y
        r2_x = off_x
        r2_y = off_y

        r1_angle = line_angle_deg + np.random.uniform(-100, 100)
        r2_angle = (line_angle_deg + 180 + np.random.uniform(-100, 100)) % 360

        return {"x": r1_x, "y": r1_y, "angle": r1_angle}, {
            "x": r2_x,
            "y": r2_y,
            "angle": r2_angle,
        }

    def _get_state_vec(self, viewer, target):
        # 1. Observed State (SENSORS)
        v_wheel_l = getattr(viewer, 'wheel_l_speed', 0.0)
        v_wheel_r = getattr(viewer, 'wheel_r_speed', 0.0)
        omega = viewer.omega / ROTATE_SPEED if ROTATE_SPEED != 0 else 0
        
        curr_l = getattr(viewer, 'current_l', 0.0)
        curr_r = getattr(viewer, 'current_r', 0.0)
        
        # 2. Opponent Sensors (TIGHTENED LOGIC)
        sensor_angles = [-20, 0, 20, -90, 90] 
        opp_distances = []
        
        # IMPORTANT: Explicitly calculate distance between the two robots
        dx = target.x - viewer.x
        dy = target.y - viewer.y
        dist_to_opp = math.hypot(dx, dy)
        
        # SAFETY CHECK: If distance is 0 but robots aren't the same, 
        # it's a physics glitch. We prevent it here.
        if dist_to_opp < 0.1: 
            # If they are practically touching, we use a small value 
            # instead of absolute 0 to prevent division errors
            dist_to_opp = max(dist_to_opp, 0.01)

        angle_to_opp_global = math.atan2(dy, dx)
        robot_rad = math.radians(viewer.angle % 360)
        
        for angle_deg in sensor_angles:
            ray_rad = robot_rad + math.radians(angle_deg)
            diff = angle_to_opp_global - ray_rad
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            
            # Use a tighter cone (10 degrees instead of 15) for better accuracy
            if abs(diff) < math.radians(10):
                # Normalize distance: 0.0 (touching) to 1.0 (edge of arena)
                norm_dist = dist_to_opp / (self.ARENA_RADIUS * 2)
                opp_distances.append(np.clip(norm_dist, 0.0, 1.0))
            else:
                opp_distances.append(1.0) # Sensor sees nothing

        # 3. Multi-Point Line Sensors
        corners = viewer.get_corners() 
        fl_dist = math.hypot(corners[3][0], corners[3][1])
        s_fl_line = 1.0 if fl_dist >= self.ARENA_RADIUS - 40 else 0.0
        fr_dist = math.hypot(corners[2][0], corners[2][1])
        s_fr_line = 1.0 if fr_dist >= self.ARENA_RADIUS - 40 else 0.0
        back_center = (corners[0] + corners[1]) / 2
        bc_dist = math.hypot(back_center[0], back_center[1])
        s_bc_line = 1.0 if bc_dist >= self.ARENA_RADIUS - 40 else 0.0

        return np.array(
            [
                v_wheel_l, v_wheel_r, omega, # 0, 1, 2
                curr_l, curr_r,             # 3, 4
                *opp_distances,             # 5, 6, 7, 8, 9
                s_fl_line, s_fr_line, s_bc_line # 10, 11, 12
            ],
            dtype=np.float32,
        )


    def _get_all_state_vecs(self):
        state_vecs = [
            self._get_state_vec(self.robot1, self.robot2),
            self._get_state_vec(self.robot2, self.robot1),
        ]
        self.robot1.state_vec = state_vecs[0]
        self.robot2.state_vec = state_vecs[1]
        return state_vecs

    def _calculate_env_logic(self):
        winner = 0
        # Only check the CENTER of the robot
        for idx, r in enumerate(self.robots):
            dist_to_center = math.hypot(r.x, r.y)
            if dist_to_center > self.ARENA_RADIUS:
                self.done = True
                winner = 2 if idx == 0 else 1
                break
        return self._get_all_state_vecs(), [0.0, 0.0], self.done, {"winner": winner}
    def render(self, names=None, archs=None):
        if not self.render_mode or self.renderer is None:
            return
        self.clock.tick(FPS)
        self.renderer.draw_arena(self.ARENA_RADIUS)
        state_vecs = self._get_all_state_vecs()
        if self.render_vectors:
            self.renderer.draw_observations_visual(self.robots, state_vecs)
        self.renderer.draw_robot(self.robot1, ROBOT_COLOR_1, 0)
        self.renderer.draw_robot(self.robot2, ROBOT_COLOR_2, 1)
        self.renderer.draw_ui(
            self.robots, observations=state_vecs, names=names, archs=archs
        )
        actions = [self.last_action1, self.last_action2]
        self.renderer.draw_actions(actions)

        pygame.display.flip()
