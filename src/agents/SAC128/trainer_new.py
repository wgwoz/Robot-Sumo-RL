import glob
import os
import random
import sys
from collections import deque
import multiprocessing

import numpy as np
import pygame
import torch

from src.agents.SAC128.agent import ReplayBuffer, SAC128Agent
from src.agents.SAC128.networks import GaussianActor
from src.agents.SAC128.rewards import get_reward
from src.env.sumo_env import SumoEnv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

cfg = {
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "buffer_capacity": 1000000,
    "start_steps": 15000,
    "update_after": 1000,
    "max_steps": 1000,
    "episodes": 100000,
    "render": False,
    "num_workers": 4,  # Number of parallel environments
    # Use absolute paths here:
    "master_path": os.path.join(ROOT_DIR, "models/SAC128_sumo_master_128.pt"),
    "legacy_master_path": os.path.join(ROOT_DIR, "models/SAC128_sumo_master.pt"),
    "model_dir": os.path.join(ROOT_DIR, "models/"),
}


def get_history_models(dir):
    return glob.glob(os.path.join(dir, "model_v*.pt"))


def safe_load_state_dict(model, path):
    sd = torch.load(path, map_location="cpu")
    own_state = model.state_dict()
    loaded_count = 0

    for key, value in sd.items():
        if key not in own_state:
            continue
        if value.shape == own_state[key].shape:
            own_state[key].copy_(value)
            loaded_count += 1
        else:
            print(f"Skipping incompatible key {key}: {value.shape} -> {own_state[key].shape}")

    if loaded_count > 0:
        return True

    print(f"Warning: No compatible weights could be loaded from {path}.")
    return False


def collect_experiences(agent_state, opp_path, cfg, total_steps):
    """Collect experiences from one environment episode."""
    device = torch.device("cpu")  # Use CPU for workers
    agent = SAC128Agent(obs_size=13, action_dim=2, device=device, hidden_dim=128, lr=cfg["lr"])
    agent.load_state_dict(agent_state)

    opp_net = GaussianActor(13, 2).to(device).eval()
    sd = torch.load(opp_path, map_location=device)
    actor_sd = {}
    for k, v in sd.items():
        if k.startswith("actor."):
            actor_sd[k[6:]] = v
    opp_net.load_state_dict(actor_sd or sd)

    env = SumoEnv(render_mode=False)
    state_vecs = env.reset(randPositions=True)
    done, ep_steps = False, 0
    experiences = []

    while not done and ep_steps < cfg["max_steps"]:
        if total_steps < cfg["start_steps"]:
            act_np = np.random.uniform(-1, 1, 2)
        else:
            s_t = torch.FloatTensor(state_vecs[0]).to(device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.actor.sample(s_t)
            act_np = action.cpu().numpy()[0]
            act_np += np.random.uniform(-0.02, 0.02, 2)

        opp_s_t = torch.FloatTensor(state_vecs[1]).to(device).unsqueeze(0)
        with torch.no_grad():
            _, _, opp_mu = opp_net.sample(opp_s_t)
        opp_act_np = opp_mu.cpu().numpy()[0]
        opp_act_np += np.random.uniform(-0.02, 0.02, 2)

        act_np = np.clip(act_np, -1, 1)
        opp_act_np = np.clip(opp_act_np, -1, 1)

        next_state_vecs, _, env_done, info = env.step(act_np, opp_act_np)
        ep_steps += 1
        done = env_done or ep_steps >= cfg["max_steps"]
        if ep_steps >= cfg["max_steps"]:
            info["winner"] = 0

        rew = get_reward(env, info, done, next_state_vecs[0], info.get("is_collision", False))
        experiences.append((state_vecs[0], act_np, rew, next_state_vecs[0], float(done)))
        state_vecs = next_state_vecs

    winner = info.get("winner", 0)
    return experiences, winner


def train():
    """Main training loop for the SAC128 agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_dir = os.path.join(cfg["model_dir"], "history/SAC128")
    os.makedirs(history_dir, exist_ok=True)

    agent = SAC128Agent(obs_size=13, action_dim=2, device=device, hidden_dim=128, lr=cfg["lr"])
    memory = ReplayBuffer(cfg["buffer_capacity"])

    if os.path.exists(cfg["master_path"]):
        loaded_ok = safe_load_state_dict(agent, cfg["master_path"])
        if loaded_ok:
            print("Loaded MASTER SAC128 model")
        else:
            print("Loaded partial checkpoint into smaller SAC128 model. Training will continue with compatible weights only.")
    elif os.path.exists(cfg["legacy_master_path"]):
        loaded_ok = safe_load_state_dict(agent, cfg["legacy_master_path"])
        if loaded_ok:
            print("Loaded legacy SAC128 checkpoint into smaller model.")
        else:
            print("Legacy SAC128 checkpoint incompatible; starting fresh smaller model.")
        torch.save(agent.state_dict(), cfg["master_path"])
        print("Saved new smaller master checkpoint to the new path.")
    else:
        print("Initializing new master.")
        torch.save(agent.state_dict(), cfg["master_path"])

    win_history = deque(maxlen=100)
    total_steps = 0
    last_update_ep = 0

    # Thresholds: (WinRate, MinEpisodesBreak, MaxDraws)
    c_list = [(0.51, 12, 50), (0.55, 8, 46), (0.60, 4, 40)]

    pool = multiprocessing.Pool(cfg["num_workers"])

    try:
        for ep in range(0, cfg["episodes"], cfg["num_workers"]):
            hist = get_history_models(history_dir)
            is_master = random.random() >= 0.20 or not hist
            opp_path = cfg["master_path"] if is_master else random.choice(hist)

            agent_state = agent.state_dict()
            agent_state = {k: v.cpu() for k, v in agent_state.items()}

            results = [pool.apply_async(collect_experiences, (agent_state, opp_path, cfg, total_steps)) for _ in range(cfg["num_workers"])]

            all_experiences = []
            winners = []
            for r in results:
                exps, winner = r.get()
                all_experiences.extend(exps)
                winners.append(winner)

            for exp in all_experiences:
                memory.push(*exp)

            total_steps += len(all_experiences)

            # Update agent
            update_count = len(all_experiences) // cfg["batch_size"]
            for _ in range(update_count):
                if len(memory) > cfg["batch_size"] and total_steps > cfg["update_after"]:
                    q_l, a_l, alpha_v = agent.update_parameters(memory, cfg["batch_size"], cfg["gamma"], cfg["tau"])

            # Handle win history
            for winner in winners:
                if is_master:
                    win_history.append(1.0 if winner == 1 else (0.5 if winner == 0 else 0.0))

            if win_history:
                wr = sum(win_history) / len(win_history)
                sys.stdout.write(f"\rEp {ep:04d}-{ep+cfg['num_workers']-1:04d} | WR: {wr:.2%} | Alpha: {alpha_v if 'alpha_v' in locals() else 0:.4f}")
                sys.stdout.flush()

            # Master update logic
            if len(win_history) >= 20:
                draw_count = sum(1 for score in win_history if score == 0.5)
                if any(wr >= thr and (ep - last_update_ep) >= wait and draw_count < d for thr, wait, d in c_list):
                    ver = len(get_history_models(history_dir))
                    torch.save(agent.state_dict(), cfg["master_path"])
                    torch.save(agent.state_dict(), os.path.join(history_dir, f"model_v{ver}.pt"))
                    last_update_ep = ep
                    print(f"\nNEW SAC128 MASTER v{ver} WR: {wr:.2%}")

    finally:
        pool.close()
        pool.join()
        pygame.quit()


if __name__ == "__main__":
    train()