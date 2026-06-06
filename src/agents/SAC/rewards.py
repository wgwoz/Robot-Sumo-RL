import math

def get_reward(env, info, done, state_vec, is_collision):
    # 1. THE FINAL OUTCOME (Scaled down to stabilize Alpha)
    # The proportions remain the same, but the smaller numbers 
    # prevent the optimizer's entropy from exploding.
    if done:
        winner = info.get("winner")
        if winner == 1: return 5000.0   # Win
        if winner == 2: return -800.0  # Loss
        return -800.0                  # Draw (Still heavily punished)

    r = 0.0
    v_avg = (state_vec[0] + state_vec[1]) / 2 
    ang_vel = state_vec[2]
    
    dist_opp = state_vec[6] 
    left_sensor = state_vec[5]
    right_sensor = state_vec[7]
    
    front_line_sensors = state_vec[10:12]
    on_edge = any(s > 0.5 for s in front_line_sensors)

    # 2. EDGE SURVIVAL MODE (Scaled 10x down)
    if state_vec[12] > 0.5:
        r -= 3  # Soft boundary warning
        if v_avg > 0: 
            r += 2 
            
    elif on_edge:
        r -= 5  # Hard boundary panic
        if v_avg < 0: 
            r += 3 
            
    # 3. HUNTING MODE (Safe inside the ring)
    else:
        
        sees_target = False
        
        # Center sensor (Highest priority)
        if dist_opp < 1.0:
            r += 20.0 * (1.0 - dist_opp)*v_avg
            sees_target = True
            
        # Side sensors (Lower priority, pulls robot to center)
        if left_sensor < 1.0:
            r += 11 * (1.0 - left_sensor)*v_avg
            sees_target = True
            
        if right_sensor < 1.0:
            r += 11 * (1.0 - right_sensor)*v_avg
            sees_target = True

        # Single, clean search penalty
        if not sees_target:
            r -= 15  # Gentle push to find the opponent, no panic spinning
            
        # Calculate opponent's distance from center
        opp_dist_center = math.hypot(env.robot2.x, env.robot2.y) / env.ARENA_RADIUS
        my_dist_center = math.hypot(env.robot2.x, env.robot2.y) / env.ARENA_RADIUS
        
        
        
        # 4. IMPACT (The Strike)
        if is_collision and state_vec[6]<1:
            
            r += 10.0 + (abs(v_avg) * 5.0) # Standard push reward
            
            # PROGRESS REWARD: Is the opponent being pushed toward the edge?
            
        
            if my_dist_center < opp_dist_center:
                r += opp_dist_center * 10.0 # High reward for clearing the ring



        if my_dist_center > opp_dist_center and state_vec[6]<0.4:
                r -= my_dist_center * 30.0 # High reward for clearing the ring

        if v_avg < 0: 
            r -= 6

        if v_avg > 0.6: 
            r += 1

    # 5. CONSTANT RESTRAINTS
    # Scaled down so the robot is willing to turn to search
    if abs(ang_vel) > 0.5:
        r -= 0.05 if on_edge else 5 


    if env.robot1.is_slipping:
        r -= 5

    if state_vec[8]<1 or state_vec[9] <1 :
        r -= 5
    
    if (env.robot2.state_vec[8] < 1 or env.robot2.state_vec[9] < 1 ) and state_vec[6]<1:
        r+= 15

    # Small time penalty to discourage draws
    r -= 0.5 

    
    return float(r)