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
from src.agents.PPO.agent import create_agent, get_distribution
from src.agents.SAC128.rewards import get_reward
from src.env.sumo_env import SumoEnv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

cfg = {
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "buffer_capacity": 1000000,
    "start_steps": 10000,
    "update_after": 300,
    "max_steps": 600, # Balanced for navigation and aggression
    "episodes": 10000000,
    "render": False,
    "num_workers": 4,
    "master_path": os.path.join(ROOT_DIR, "models/SAC128_sumo_master_128.pt"),
    "legacy_master_path": os.path.join(ROOT_DIR, "models/SAC128_sumo_master.pt"),
    "ppo_master_path": os.path.join(ROOT_DIR, "models/ppo_sumo_master.pt"),
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


def load_opponent_net(opp_path, device):
    sd = torch.load(opp_path, map_location=device)

    has_ppo_prefix = any(
        k.startswith("actor.layer1.") or k.startswith("actor.layer2.") or k.startswith("actor.layer3.")
        for k in sd
    )
    has_ppo_actor = any(
        k.startswith("layer1.") or k.startswith("layer2.") or k.startswith("layer3.")
        for k in sd
    )

    if has_ppo_prefix or has_ppo_actor:
        opp_net = create_agent(13, 128).to(device).eval()
        if has_ppo_actor and not has_ppo_prefix:
            opp_net.actor.load_state_dict(sd)
        else:
            opp_net.load_state_dict(sd)
        return opp_net, "ppo"

    opp_net = GaussianActor(13, 2).to(device).eval()
    actor_sd = {k[6:]: v for k, v in sd.items() if k.startswith("actor.")}
    opp_net.load_state_dict(actor_sd or sd)
    return opp_net, "SAC128"


def collect_experiences(agent_state, opp_path, cfg, total_steps):
    """Collect experiences. If opp_path is None, the opponent is a stationary dummy."""
    device = torch.device("cpu")
    agent = SAC128Agent(obs_size=13, action_dim=2, device=device, hidden_dim=128, lr=cfg["lr"])
    agent.load_state_dict(agent_state)

    opp_net = None
    opp_type = None
    if opp_path is not None:
        opp_net, opp_type = load_opponent_net(opp_path, device)

    env = SumoEnv(render_mode=False)
    state_vecs = env.reset(randPositions=True)
    done, ep_steps = False, 0
    experiences = []
    ep_rew = 0
    last_random_action = np.random.uniform(-1, 1, 2)

    while not done and ep_steps < cfg["max_steps"]:
        # AGENT ACTION
        if total_steps < cfg["start_steps"]:
            if ep_steps % 15 == 0: last_random_action = np.random.uniform(-1, 1, 2)
            act_np = last_random_action
        else:
            s_t = torch.FloatTensor(state_vecs[0]).to(device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.actor.sample(s_t)
            act_np = action.cpu().numpy()[0]
            act_np += np.random.uniform(-0.02, 0.02, 2)

        # OPPONENT ACTION (The "Box" logic)
        if opp_net is None:
            opp_act_np = np.array([0.0, 0.0]) # Stationary dummy
        else:
            opp_s_t = torch.FloatTensor(state_vecs[1]).to(device).unsqueeze(0)
            with torch.no_grad():
                if opp_type == "ppo":
                    o_p, _ = opp_net(opp_s_t)
                    opp_dist = get_distribution(o_p)
                    opp_act = opp_dist.sample()
                    opp_act_np = opp_act.cpu().numpy()[0]
                else:
                    _, _, opp_mu = opp_net.sample(opp_s_t)
                    opp_act_np = opp_mu.cpu().numpy()[0]
            opp_act_np += np.random.uniform(-0.01, 0.01, 2)

        act_np = np.clip(act_np, -1, 1)
        opp_act_np = np.clip(opp_act_np, -1, 1)

        next_state_vecs, _, env_done, info = env.step(act_np, opp_act_np)
        ep_steps += 1
        done = env_done or ep_steps >= cfg["max_steps"]
        
        if ep_steps >= cfg["max_steps"]: info["winner"] = 0

        rew = get_reward(env, info, done, next_state_vecs[0], info.get("is_collision", False))
        ep_rew += rew
        
        experiences.append((state_vecs[0], act_np, rew, next_state_vecs[0], float(done)))
        state_vecs = next_state_vecs

    winner = info.get("winner", 0)
    return experiences, winner, ep_rew

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
            print("Loaded MASTER model. Resuming training...")
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
        torch.save(agent.state_dict(), cfg["master_path"])

    win_history = deque(maxlen=100)
    total_steps = 0
    last_update_ep = 0

    # Thresholds: (WinRate, MinEpisodesBreak, MaxDraws)
    c_list = [
        (0.51, 100, 90),
        (0.55, 80, 85),
        (0.60, 60, 75),
        (0.65, 40, 70),
   
    ]

    pool = multiprocessing.Pool(cfg["num_workers"])

    try:
        for ep in range(0, cfg["episodes"], cfg["num_workers"]):
            # --- 1. VALIDATION MATCH (Every 100 Episodes) ---
            if ep > 0 and ep % 100 == 0:
                print(f"\n--- 🔎 VALIDATION MATCH: Episode {ep} ---")
                # We create a temporary environment for rendering
                val_env = SumoEnv(render_mode=True)
                val_state = val_env.reset(randPositions=True)
                val_done, val_steps = False, 0
                
                # For validation, we let the AI fight the current Master
                val_opp_net = GaussianActor(13, 2).to(device).eval()
                val_sd = torch.load(cfg["master_path"], map_location=device)
                val_actor_sd = {k[6:]: v for k, v in val_sd.items() if k.startswith("actor.")} if "actor." in str(val_sd.keys()) else val_sd
                val_opp_net.load_state_dict(val_actor_sd or val_sd)

                while not val_done and val_steps < cfg["max_steps"]:
                    # Agent Action
                    s_t = torch.FloatTensor(val_state[0]).to(device).unsqueeze(0)
                    with torch.no_grad():
                        action, _, _ = agent.actor.sample(s_t)
                    act_np = action.cpu().numpy()[0]
                    
                    # Opponent Action
                    opp_s_t = torch.FloatTensor(val_state[1]).to(device).unsqueeze(0)
                    with torch.no_grad():
                        _, _, opp_mu = val_opp_net.sample(opp_s_t)
                    opp_act_np = opp_mu.cpu().numpy()[0]

                    val_state, _, val_done, info = val_env.step(act_np, opp_act_np)
                    val_steps += 1
                    val_env.render(names=["Training AI", "Master"], archs=["SAC128", "SAC128"])
                
                # Close the validation window to free resources
                pygame.display.quit()
                print(f"Validation finished. Match lasted {val_steps} steps.\n")

            # --- 2. REGULAR PARALLEL TRAINING ---
            cross_play = ((ep // cfg["num_workers"]) % 3 == 0) and (ep > 0)
            if cross_play:
                opp_path = cfg["ppo_master_path"]
                opp_name = "PPO_MASTER"
                is_master = False  # Cross-play should not affect SAC128 master win_history
            else:
                hist = get_history_models(history_dir)
                is_master = random.random() >= 0.20 or not hist
                opp_path = cfg["master_path"] if is_master else random.choice(hist)
                opp_name = "MASTER" if is_master else os.path.basename(opp_path)

            agent_state = {k: v.cpu() for k, v in agent.state_dict().items()}
            results = [pool.apply_async(collect_experiences, (agent_state, opp_path, cfg, total_steps)) for _ in range(cfg["num_workers"])]

            all_experiences, winners, all_rewards = [], [], []
            for r in results:
                exps, winner, rew = r.get()
                all_experiences.extend(exps)
                winners.append(winner)
                all_rewards.append(rew)

            for exp in all_experiences: memory.push(*exp)
            total_steps += len(all_experiences)
            avg_rew = sum(all_rewards) / len(all_rewards)

            # Neural Network Update
            update_count = len(all_experiences) // cfg["batch_size"]
            for _ in range(update_count):
                if len(memory) > cfg["batch_size"] and total_steps > cfg["update_after"]:
                    q_l, a_l, alpha_v = agent.update_parameters(memory, cfg["batch_size"], cfg["gamma"], cfg["tau"])

            # Only track Win Rate for matches against "Masters"
            if is_master:
                for winner in winners:
                    win_history.append(1.0 if winner == 1 else (0.5 if winner == 0 else 0.0))

            if win_history:
                wr = sum(win_history) / len(win_history)
                sys.stdout.write(f"\rEp {ep:04d}-{ep+cfg['num_workers']-1:04d} | vs {opp_name:12} | WR (vs Masters): {wr:.2%} | Rew: {avg_rew:7.2f} | Alpha: {alpha_v if 'alpha_v' in locals() else 0:.4f}")
                sys.stdout.flush()

            # Master Promotion
            if len(win_history) >= 20:
                draw_count = sum(1 for score in win_history if score == 0.5)
                if any(wr >= thr and (ep - last_update_ep) >= wait and draw_count < d for thr, wait, d in c_list):
                    ver = len(get_history_models(history_dir))
                    torch.save(agent.state_dict(), cfg["master_path"])
                    torch.save(agent.state_dict(), os.path.join(history_dir, f"model_v{ver}.pt"))
                    last_update_ep = ep
                    print(f"\n🔥 NEW MASTER v{ver} PROMOTED! WR: {wr:.2%}")

    finally:
        pool.close()
        pool.join()
        pygame.quit()


if __name__ == "__main__":
    train()