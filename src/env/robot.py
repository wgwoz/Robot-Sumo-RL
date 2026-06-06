import math

import numpy as np

from src.env.config import *

from src.env.config import ROBOT_SIDE # Make sure this import exists

class SumoRobot:
    def __init__(self, x, y, angle, mass=1):
        self.x = x
        self.y = y
        self.angle = angle
        
        self.mass = mass
        self.ROBOT_WIDTH = ROBOT_SIDE

        self.v = 0.0
        self.v_side = 0.0
        self.omega = 0.0

        self.is_slipping = False  # Track if the robot is currently sliding
        self.wheel_l_speed = 0.0 # Track actual wheel speed for sensors
        self.wheel_r_speed = 0.0
        self.current_l = 0.0      # Track motor current for sensors
        self.current_r = 0.0

        self.mass = mass
        self.width = ROBOT_SIZE_PX

        self.friction = FRICTION
        self.lateral_friction = LATERAL_FRICTION

    def compute_dynamics(self, action_l, action_r):
        # Reset slip flag each frame - it should only be True when currently slipping
        self.is_slipping = False
        
        self.wheel_l_command = action_l
        self.wheel_r_command = action_r

        target_v = (action_l + action_r) / 2 * MAX_SPEED
        target_omega = (action_r - action_l) / 2 * ROTATE_SPEED
###uncomment below if you want model to account for traction control
        dv = target_v - self.v
        #if abs(dv) > ACCELERATION:
        #    #self.is_slipping = True
        #    dv = np.sign(dv) * ACCELERATION
        self.v += dv

        domega = target_omega - self.omega
        #if abs(domega) > ACCEL_ANGULAR:
        #    #self.is_slipping = True
        #    domega = np.sign(domega) * ACCEL_ANGULAR
        self.omega += domega

        self.v = np.clip(self.v, -MAX_SPEED, MAX_SPEED)
        self.omega = np.clip(self.omega, -ROTATE_SPEED, ROTATE_SPEED)

        self.v *= (1.0 - self.friction)
        self.omega *= (1.0 - self.friction)

        self.v_side *= (1.0 - self.lateral_friction)

        self.wheel_l_speed = action_l
        self.wheel_r_speed = action_r

    def compute_kinematics(self, dt=1.0):
        self.angle += math.degrees(self.omega * dt)

        rad = math.radians(self.angle)

        forward_vec = np.array([math.cos(rad), math.sin(rad)])
        side_vec = np.array([-math.sin(rad), math.cos(rad)])

        self.x += (self.v * forward_vec[0] + self.v_side * side_vec[0]) * dt
        self.y += (self.v * forward_vec[1] + self.v_side * side_vec[1]) * dt

        # Check for slip based on wheel speed difference, normalized to max wheel speed
        required_l = (self.v - self.omega * (self.ROBOT_WIDTH * M_TO_PX) / 2) / MAX_SPEED
        required_r = (self.v + self.omega * (self.ROBOT_WIDTH * M_TO_PX) / 2) / MAX_SPEED
        # !!do not remove!! physics that causes wheels to slip when they try to spin too fast
        # if abs(self.wheel_l_command - required_l) > SLIP_THRESHOLD or abs(self.wheel_r_command - required_r) > SLIP_THRESHOLD:
        #     self.is_slipping = True

        self.current_l = np.clip(required_l, -1.0, 1.0)
        self.current_r = np.clip(required_r, -1.0, 1.0)

    def apply_impulse(self, impulse_vec):
        """Convert collision impulse into local velocity changes."""
        dv_x = impulse_vec[0] / self.mass
        dv_y = impulse_vec[1] / self.mass

        rad = math.radians(self.angle)
        forward_vec = np.array([math.cos(rad), math.sin(rad)])
        side_vec = np.array([-math.sin(rad), math.cos(rad)])

        self.v += np.dot(np.array([dv_x, dv_y]), forward_vec)
        self.v_side += np.dot(np.array([dv_x, dv_y]), side_vec)

    def get_corners(self):
        """Calculates the 4 corners of the robot in global coordinates."""
        half_w = self.width / 2
        
        # Local coordinates
        local_corners = [
            (-half_w, -half_w), # BL
            (half_w, -half_w),  # BR
            (half_w, half_w),   # TR
            (-half_w, half_w),  # TL
        ]
        
        global_corners = []
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        for lx, ly in local_corners:
            gx = self.x + (lx * cos_a - ly * sin_a)
            gy = self.y + (lx * sin_a + ly * cos_a)
            global_corners.append([gx, gy]) 
            # <--- MAKE SURE THERE IS NO 'return' HERE!
            
        # THIS RETURN MUST BE OUTSIDE THE FOR LOOP
        return np.array(global_corners) 



