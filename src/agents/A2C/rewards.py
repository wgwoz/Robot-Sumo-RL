def get_reward(env, info, done, state_vec, is_collision):
    if done:
        winner = info.get("winner")
        if winner == 1: return 50.0
        if winner == 2: return -30.0
        return -20.0

    r = 0.0
    v_fwd = state_vec[0]
    dist_center_opp = state_vec[4] # Front-Center sensor
    on_line = state_vec[8]

    # Reward moving forward IF the center sensor sees the opponent (accurate aiming)
    if v_fwd > 0.2 and dist_center_opp < 0.5:
        r += 0.05 * (1.0 - dist_center_opp)

    if is_collision:
        # Reward high-speed collision if we are facing them (center sensor is low)
        if dist_center_opp < 0.3:
            r += 0.5 * v_fwd
        else:
            r -= 0.2 # Penalty for side-impacts (anti-dancing)

    # Line safety: If on the line, reward moving AWAY from it (facing center)
    # Since we don't have center-angle anymore, we penalize staying on the line
    if on_line > 0.5:
        r -= 0.5 

    # Time penalty
    r -= 0.03
    return float(r)