import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch

from src.agents.SAC128.networks import GaussianActor
from src.agents.SAC128.rewards import get_reward
from src.env.sumo_env import SumoEnv
from src.agents.SAC128.heuristic_actor import HeuristicActor # <--- ADDED THIS IMPORT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- TEST CONFIG ---
# You can now set these to "heuristic" or a path to a .pt file
PLAYER_1_TYPE = "ai"
MODEL_1_PATH = "models/SAC128_sumo_master_128.pt" # Change to "models/SAC128_sumo_master_128.pt" to use AI

PLAYER_2_TYPE = "ai"
MODEL_2_PATH = "models/SAC_sumo_master.pt" # Change to "models/SAC128_sumo_master.pt" to use AI

MAX_STEPS = 1000

# --- GRAPH ---
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
(line1,) = ax.plot([], [], "g-", label="Robot 1 (Green)", linewidth=1.5)
(line2,) = ax.plot([], [], "b-", label="Robot 2 (Blue)", linewidth=1.5)
fig.canvas.manager.set_window_title("Live Reward Analytics")
ax.set_title("Cumulative Reward")
ax.set_xlabel("Step")
ax.set_ylabel("Total Reward")
ax.legend()
ax.grid(True, alpha=0.3)


def load_SAC128_actor(path, device):
    # Check if path is "heuristic_Someting" (e.g., "heuristic_Aggressive")
    if isinstance(path, str) and path.startswith("heuristic_"):
        strategy_name = path.replace("heuristic_", "")
        print(f"Loading Heuristic Strategy: {strategy_name}")
        return HeuristicActor(strategy_id=strategy_name)
    
    # Fallback for the old "heuristic" keyword
    if path == "heuristic":
        return HeuristicActor(strategy_id="Main")

    if not os.path.exists(path):
        print(f"Model not found: {path}")
        return None

    model = GaussianActor(obs_size=13, action_dim=2).to(device) # Updated to 9
    try:
        sd = torch.load(path, map_location=device)

        actor_sd = {}
        if any(k.startswith("actor.") for k in sd.keys()):
            for k, v in sd.items():
                if k.startswith("actor."):
                    actor_sd[k[6:]] = v
        else:
            actor_sd = sd

        model.load_state_dict(actor_sd)
        model.eval()
        return model
    except Exception as e:
        print(f"SAC128 loading error: {e}")
        return None


def get_action(p_type, robot_idx, state, model):
    if p_type == "dummy":
        return [0.0, 0.0]

    if p_type == "human":
        keys = pygame.key.get_pressed()
        v, omega = 0.0, 0.0
        if robot_idx == 0:
            if keys[pygame.K_UP]: v = 1.0
            if keys[pygame.K_DOWN]: v = -1.0
            if keys[pygame.K_LEFT]: omega = 1.0
            if keys[pygame.K_RIGHT]: omega = -1.0
        else:
            if keys[pygame.K_w]: v = 1.0
            if keys[pygame.K_s]: v = -1.0
            if keys[pygame.K_a]: omega = 1.0
            if keys[pygame.K_d]: omega = -1.0
        return [v, omega]

    if p_type == "ai" and model:
        # The model.sample() function handles both Heuristic and SAC128
        obs = torch.FloatTensor(state[robot_idx]).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, _, mu = model.sample(obs)
            return mu.cpu().numpy().flatten()
    return [0.0, 0.0]


def main():
    env = SumoEnv(render_mode=True)
    model1 = load_SAC128_actor(MODEL_1_PATH, DEVICE) if PLAYER_1_TYPE == "ai" else None
    model2 = load_SAC128_actor(MODEL_2_PATH, DEVICE) if PLAYER_2_TYPE == "ai" else None

    scores = [0, 0]
    round_count = 0

    print(
        f"TEST SAC128: {MODEL_1_PATH} vs {MODEL_2_PATH}"
    )

    while True:
        state = env.reset(randPositions=True)
        if isinstance(state, tuple):
            state = state[0]

        done, step_count = False, 0
        total_r1, total_r2 = 0.0, 0.0
        steps_h, r1_h, r2_h = [], [], []

        while not done and step_count < MAX_STEPS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    plt.close()
                    pygame.quit()
                    return

            act1 = get_action(PLAYER_1_TYPE, 0, state, model1)
            act2 = get_action(PLAYER_2_TYPE, 1, state, model2)

            state, _, env_done, info = env.step(act1, act2)
            if isinstance(state, tuple):
                state = state[0]

            done = env_done or (step_count + 1 >= MAX_STEPS)

            info_r2 = info.copy()
            if info.get("winner") == 1:
                info_r2["winner"] = 2
            elif info.get("winner") == 2:
                info_r2["winner"] = 1

            r1_s = get_reward(
                env, info, done, state[0], info.get("is_collision", False)
            )
            r2_s = get_reward(
                env, info_r2, done, state[1], info.get("is_collision", False)
            )

            total_r1 += r1_s
            total_r2 += r2_s

            steps_h.append(step_count)
            r1_h.append(total_r1)
            r2_h.append(total_r2)

            if step_count % 10 == 0:
                line1.set_data(steps_h, r1_h)
                line2.set_data(steps_h, r2_h)
                ax.relim()
                ax.autoscale_view()
                ax.set_title(f"Cumulative Reward (Test)")
                fig.canvas.flush_events()

            sys.stdout.write(
                f"\rStep: {step_count:4d} | R1: {total_r1:7.1f} | R2: {total_r2:7.1f}"
            )
            sys.stdout.flush()

            step_count += 1
            env.render(
                names=[MODEL_1_PATH, MODEL_2_PATH],
                archs=["SAC128" if ".pt" in MODEL_1_PATH else "HEURISTIC", 
                       "SAC128" if ".pt" in MODEL_2_PATH else "HEURISTIC"],
            )

        round_count += 1
        winner = info.get("winner", 0)
        if winner == 1:
            scores[0] += 1
        elif winner == 2:
            scores[1] += 1

        print(
            f"\n🏆 Round {round_count} Over. Winner: Robot {winner} | Score: {scores[0]}-{scores[1]}"
        )
        pygame.time.wait(1500)

        # Hot-reload (only if it's a file, otherwise keep the heuristic object)
        if PLAYER_1_TYPE == "ai" and ".pt" in MODEL_1_PATH:
            model1 = load_SAC128_actor(MODEL_1_PATH, DEVICE)
        if PLAYER_2_TYPE == "ai" and ".pt" in MODEL_2_PATH:
            model2 = load_SAC128_actor(MODEL_2_PATH, DEVICE)


if __name__ == "__main__":
    main()
