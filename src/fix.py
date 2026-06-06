import re

with open('agents/SAC/trainer.py', 'r') as f:
    content = f.read()

# Remove opp_net
content = re.sub(r'    opp_net = GaussianActor\(13, 2\)\.to\(device\)\.eval\(\)\s*\n', '', content)

# Replace the for loop body
old_pattern = re.compile(r'        for ep in range\(0, cfg\[\"episodes\"\], cfg\[\"num_workers\"\]\):\s*\n.*?pygame\.quit\(\)', re.DOTALL)

new_code = '''        for ep in range(0, cfg["episodes"], cfg["num_workers"]):
            hist = get_history_models(history_dir)
            is_master = random.random() >= 0.20 or not hist
            opp_path = cfg["master_path"] if is_master else random.choice(hist)

            agent_state = agent.state_dict()

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
                sys.stdout.write(f"\\rEp {ep:04d}-{ep+cfg['num_workers']-1:04d} | WR: {wr:.2%} | Alpha: {alpha_v if 'alpha_v' in locals() else 0:.4f}")
                sys.stdout.flush()

            # Master update logic
            if len(win_history) >= 20:
                draw_count = sum(1 for score in win_history if score == 0.5)
                if any(wr >= thr and (ep - last_update_ep) >= wait and draw_count < d for thr, wait, d in c_list):
                    ver = len(get_history_models(history_dir))
                    torch.save(agent.state_dict(), cfg["master_path"])
                    torch.save(agent.state_dict(), os.path.join(history_dir, f"model_v{ver}.pt"))
                    last_update_ep = ep
                    print(f"\\nNEW SAC MASTER v{ver} WR: {wr:.2%}")

    finally:
        pool.close()
        pool.join()
        pygame.quit()'''

content = old_pattern.sub(new_code, content)

with open('agents/SAC/trainer.py', 'w') as f:
    f.write(content)

print('File updated')