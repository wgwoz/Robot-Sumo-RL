import glob
import os
import random
import sys
import multiprocessing
from collections import deque

import numpy as np
import pygame
import torch
import torch.optim as optim

from src.agents.PPO.agent import (
    calculate_returns,
    create_agent,
    get_distribution,
    update_policy,
)
from src.agents.PPO.rewards import get_reward
from src.env.sumo_env import SumoEnv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

cfg = {
    "lr": 3e-4,
    "gamma": 0.99,
    "ppo_epochs": 10,
    "eps_clip": 0.2,
    "entropy_coef": 0.01,
    "update_every_steps": 2048,
    "max_steps": 600,
    "episodes": 100000,
    "render": False,
    "num_workers": 4,
    "obs_size": 13,
    "master_path": os.path.join(ROOT_DIR, "models/ppo_sumo_master.pt"),
    "model_dir": os.path.join(ROOT_DIR, "models/"),
}


def get_history_models(dir):
    return glob.glob(os.path.join(dir, "model_v*.pt"))


def collect_episode(agent_state, opp_path, cfg, total_steps):
    device = torch.device("cpu")
    model = create_agent(cfg["obs_size"], 128).to(device)
    model.load_state_dict(agent_state)
    model.eval()

    opp_net = create_agent(cfg["obs_size"], 128).to(device).eval()
    if os.path.exists(opp_path):
        opp_net.load_state_dict(torch.load(opp_path, map_location=device))

    env = SumoEnv(render_mode=False)
    state_vecs = env.reset(randPositions=True)

    done = False
    episode_reward = 0.0
    episode_steps = 0

    states, actions, log_probs, values, rewards, masks = [], [], [], [], [], []

    while not done and episode_steps < cfg["max_steps"]:
        s_t = torch.FloatTensor(state_vecs[0]).to(device).unsqueeze(0)
        opp_s_t = torch.FloatTensor(state_vecs[1]).to(device).unsqueeze(0)

        with torch.no_grad():
            action_params, value_pred = model(s_t)
            dist = get_distribution(action_params)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            opp_params, _ = opp_net(opp_s_t)
            opp_dist = get_distribution(opp_params)
            opp_action = opp_dist.sample()

        act_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()[0]
        opp_act_np = torch.clamp(opp_action, -1.0, 1.0).cpu().numpy()[0]

        next_state_vecs, _, env_done, info = env.step(act_np, opp_act_np)
        episode_steps += 1
        done = env_done or episode_steps >= cfg["max_steps"]
        if episode_steps >= cfg["max_steps"]:
            info["winner"] = 0

        rew = get_reward(
            env,
            info,
            done,
            next_state_vecs[0],
            info.get("is_collision", False),
        )

        states.append(state_vecs[0].astype(np.float32))
        actions.append(act_np.astype(np.float32))
        log_probs.append(log_prob.item())
        values.append(value_pred.item())
        rewards.append(rew)
        masks.append(1.0 - float(done))

        episode_reward += rew
        state_vecs = next_state_vecs

    if done:
        last_value = 0.0
    else:
        with torch.no_grad():
            _, last_value = model(
                torch.FloatTensor(state_vecs[0]).to(device).unsqueeze(0)
            )
            last_value = last_value.item()

    returns = calculate_returns(rewards, cfg["gamma"], last_value, masks).cpu().numpy().astype(np.float32)
    advantages = returns - np.array(values, dtype=np.float32)

    return {
        "states": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "episode_reward": episode_reward,
        "winner": int(info.get("winner", 0)),
        "steps": episode_steps,
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history_dir = os.path.join(cfg["model_dir"], "history/PPO")
    os.makedirs(history_dir, exist_ok=True)

    model = create_agent(cfg["obs_size"], 128).to(device)

    if os.path.exists(cfg["master_path"]):
        model.load_state_dict(torch.load(cfg["master_path"], map_location=device))
        print("📁 Loaded MASTER model")
    else:
        print("🆕 Initializing new master.")
        torch.save(model.state_dict(), cfg["master_path"])

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    win_history = deque(maxlen=100)
    last_update_ep = 0
    buffer_steps = 0
    storage = {"s": [], "a": [], "lp": [], "ad": [], "rt": []}
    total_steps = 0

    c_list = [
        (0.51, 40, 90),
        (0.52, 36, 80),
        (0.53, 32, 75),
        (0.54, 28, 70),
        (0.55, 24, 65),
        (0.56, 20, 60),
        (0.57, 16, 55),
        (0.58, 12, 50),
        (0.59, 8, 45),
        (0.60, 5, 40),
    ]

    pool = multiprocessing.Pool(cfg["num_workers"])

    try:
        for ep in range(0, cfg["episodes"], cfg["num_workers"]):
            hist = get_history_models(history_dir)
            is_master = random.random() >= 0.20 or not hist
            opp_path = cfg["master_path"] if is_master else random.choice(hist)
            opp_name = "MASTER" if is_master else os.path.basename(opp_path)

            if ep > 0 and ep % 1000 == 0:
                val_env = SumoEnv(render_mode=True)
                val_state = val_env.reset(randPositions=True)
                val_done = False
                val_steps = 0

                opp_net = create_agent(cfg["obs_size"], 128).to(device).eval()
                if os.path.exists(cfg["master_path"]):
                    opp_net.load_state_dict(torch.load(cfg["master_path"], map_location=device))

                while not val_done and val_steps < cfg["max_steps"]:
                    if pygame.event.peek(pygame.QUIT):
                        break

                    s_t = torch.FloatTensor(val_state[0]).unsqueeze(0).to(device)
                    opp_s_t = torch.FloatTensor(val_state[1]).unsqueeze(0).to(device)

                    with torch.no_grad():
                        a_p, _ = model(s_t)
                        a_dist = get_distribution(a_p)
                        action = a_dist.sample()

                        o_p, _ = opp_net(opp_s_t)
                        o_dist = get_distribution(o_p)
                        opp_action = o_dist.sample()

                    act_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()[0]
                    opp_act_np = torch.clamp(opp_action, -1.0, 1.0).cpu().numpy()[0]

                    val_state, _, val_done, info = val_env.step(act_np, opp_act_np)
                    val_env.render(names=["Training AI", "Master"], archs=["PPO", "PPO"])
                    val_steps += 1

                pygame.display.quit()

            agent_state = {k: v.cpu() for k, v in model.state_dict().items()}
            results = [
                pool.apply_async(collect_episode, (agent_state, opp_path, cfg, total_steps))
                for _ in range(cfg["num_workers"])
            ]

            episodes = [r.get() for r in results]

            all_rewards = []
            winners = []
            for episode in episodes:
                storage["s"].append(torch.from_numpy(episode["states"]).to(device))
                storage["a"].append(torch.from_numpy(episode["actions"]).to(device))
                storage["lp"].append(torch.from_numpy(episode["log_probs"]).to(device))
                storage["ad"].append(torch.from_numpy(episode["advantages"]).to(device))
                storage["rt"].append(torch.from_numpy(episode["returns"]).to(device))

                episode_len = episode["steps"]
                buffer_steps += episode_len
                total_steps += episode_len
                all_rewards.append(episode["episode_reward"])
                winners.append(episode["winner"])

            avg_rew = sum(all_rewards) / len(all_rewards)

            if is_master:
                for winner in winners:
                    win_history.append(1.0 if winner == 1 else (0.5 if winner == 0 else 0.0))

            wr = sum(win_history) / len(win_history) if win_history else 0.0
            sys.stdout.write(
                f"Ep {ep:04d}-{ep+cfg['num_workers']-1:04d} | vs {opp_name:12} | WR: {wr:.2%} | Rew: {avg_rew:7.2f}\r"
            )
            sys.stdout.flush()

            if buffer_steps >= cfg["update_every_steps"]:
                adv_b = torch.cat(storage["ad"]).detach()
                update_policy(
                    model,
                    optimizer,
                    torch.cat(storage["s"]),
                    torch.cat(storage["a"]),
                    torch.cat(storage["lp"]).detach(),
                    (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8),
                    torch.cat(storage["rt"]).detach(),
                    cfg["ppo_epochs"],
                    cfg["eps_clip"],
                    cfg["entropy_coef"],
                )
                storage = {k: [] for k in storage}
                buffer_steps = 0
                print("\n--- [Policy Updated] ---")

            if len(win_history) >= 100:
                draw_count = sum(1 for score in win_history if score == 0.5)
                update_triggered = False
                for threshold_wr, wait_ep, max_draws in c_list:
                    if (
                        wr >= threshold_wr
                        and (ep - last_update_ep) >= wait_ep
                        and draw_count < max_draws
                    ):
                        update_triggered = True
                        break

                if update_triggered:
                    ver = len(get_history_models(history_dir))
                    torch.save(model.state_dict(), cfg["master_path"])
                    torch.save(
                        model.state_dict(),
                        os.path.join(history_dir, f"model_v{ver}.pt"),
                    )
                    last_update_ep = ep
                    print(
                        f"\n🔥 [NEW MASTER] v{ver} WR: {wr:.2%} | Draws: {draw_count} | Ep: {ep}"
                    )

    finally:
        pool.close()
        pool.join()
        pygame.quit()


if __name__ == "__main__":
    train()
