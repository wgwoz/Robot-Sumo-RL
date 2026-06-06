import torch
import numpy as np
import random

class HeuristicActor:
    def __init__(self, strategy_id="Juggernaut", obs_size=13, action_dim=2):
        self.strategy_id = strategy_id
        self.obs_size = obs_size
        self.action_dim = action_dim
        self.wait_counter = 0
        self.search_dir = random.choice([-1.0, 1.0])

    def _apply_jitter(self, value, amount=0.03):
        return value + random.uniform(-amount, amount)

    def _translate_to_wheels(self, v_fwd, omega):
        """Translates (Forward, Turn) into (Left Wheel, Right Wheel)"""
        v_l = v_fwd - (omega * 0.01) 
        v_r = v_fwd + (omega * 0.01)
        return np.clip(v_l, -1.0, 1.0), np.clip(v_r, -1.0, 1.0)

    def _finish_action(self, v_l, v_r):
        """Wraps wheel speeds into the tensor format required by the env."""
        v_l = np.clip(self._apply_jitter(v_l), -1.0, 1.0)
        v_r = np.clip(self._apply_jitter(v_r), -1.0, 1.0)
        action = np.array([v_l, v_r], dtype=np.float32)
        return (torch.FloatTensor(action).unsqueeze(0), torch.zeros(1), torch.FloatTensor(action).unsqueeze(0))

    def sample(self, state):
        obs = state.detach().cpu().numpy()[0]

        # --- SENSOR MAPPING ---
        # 0:WL, 1:WR, 2:Gyro, 3:CurrL, 4:CurrR, 5:S_FL, 6:S_FC, 7:S_FR, 8:S_L, 9:S_R, 10:L_FL, 11:L_FR, 12:L_BC
        
        # 1. EMERGENCY LINE AVOIDANCE (Highest Priority)
        if obs[10] > 0.5 or obs[11] > 0.5 or obs[12] > 0.5:
            # If we hit the line, we must retreat immediately
            v_fwd, omega = -1.0, 0.6 * self.search_dir
            v_l, v_r = self._translate_to_wheels(v_fwd, omega)
            self.wait_counter = 15
            return self._finish_action(v_l, v_r)

        # 2. WAIT/COOLDOWN STATE
        if self.wait_counter > 0:
            self.wait_counter -= 1
            return self._finish_action(obs[0], obs[1]) # Maintain current wheel speed

        # 3. STRATEGY LOGIC
        v_fwd, omega = 0.0, 0.0

        # --- STRATEGY: THE JUGGERNAUT (Brute Force) ---
        if self.strategy_id == "Juggernaut":
            # Logic: Charge everything, only turn if absolutely necessary
            if obs[6] < 1: # Target is roughly in front
                v_fwd, omega = 1.0, 0.0
            elif obs[5] < 1: # Target is slightly left
                v_fwd, omega = 1, 0.5
            elif obs[7] < 1: # Target is slightly right
                v_fwd, omega = 1, -0.5
            else: # Search
                v_fwd, omega = 0.3, 0.6 * self.search_dir

        # --- STRATEGY: THE TACTICIAN (Slippage and Current Aware) ---
        elif self.strategy_id == "Tactician":
            # Check for Stall (High current, low speed)
            is_stalled = (obs[3] > 1.5 and abs(obs[0]) < 0.2) or (obs[4] > 1.5 and abs(obs[1]) < 0.2)
            
            if is_stalled:
                # Recover from stall: Back up slightly and pivot to a new angle
                v_fwd, omega = -0.4, 1.0 * self.search_dir
            elif obs[6] < 0.4: # Opponent is very close
                # Charge, but if we detect slip (High wheel speed, low current), reduce power
                if obs[3] < 0.3 and abs(obs[0]) > 0.8:
                    v_fwd, omega = 0.5, 0.0 # Reduce power to regain grip
                else:
                    v_fwd, omega = 1.0, 0.0
            elif obs[5] < 0.6: v_fwd, omega = 0.6, 0.7
            elif obs[7] < 0.6: v_fwd, omega = 0.6, -0.7
            else: v_fwd, omega = 0.2, 0.5 * self.search_dir

        # --- STRATEGY: THE SPINNER (Slinking/Slipping) ---
        elif self.strategy_id == "Spinner":
            # Logic: Orbit the opponent and hit them from the side
            if obs[8] < 0.5: # Opponent is on the left
                v_fwd, omega = 0.4, 0.8 # Circle around them
            elif obs[9] < 0.5: # Opponent is on the right
                v_fwd, omega = 0.4, -0.8
            elif obs[6] < 0.3: # Very close in front? SLAM!
                v_fwd, omega = 1.0, 0.0
            else:
                v_fwd, omega = 0.2, 0.8 * self.search_dir

        else: # Default
            v_fwd, omega = 0.3, 0.4 * self.search_dir

        # Translate intuitive (v, omega) to physical (left, right)
        v_l, v_r = self._translate_to_wheels(v_fwd, omega)
        return self._finish_action(v_l, v_r)