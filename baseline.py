import numpy as np
import time
from src.config import *
from train import make_env  # 复用我们之前写好的环境工厂函数

def run_passive_baseline():
    print("📉 Running Passive Baseline (Do Nothing Agent)...")
    
    # 1. 创建评估环境 (和 PPO 用一模一样的圣诞节数据)
    # 注意：这里不需要 town_map_override，因为我们不训练，只跑一次
    env, _ = make_env("2022-12-22", "2022-12-26")
    
    # 2. 初始化
    obs, _ = env.reset()
    done = False
    total_reward = 0
    total_steps = 0
    
    # 获取维修队数量，以便构造全 0 的动作
    # action_space 是 MultiDiscrete，shape 对应维修队数量
    n_crews = len(env.action_space.nvec) 
    
    print(f"👻 Agent Strategy: Always send Action 0 (Stay & Wait)")
    
    # 3. 循环运行
    start_time = time.time()
    
    while not done:
        # --- 核心逻辑：全 0 动作 ---
        # 意思就是：所有车都不许主动移动，就在原地等着接单
        action = np.zeros(n_crews, dtype=int)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_steps += 1
        done = terminated or truncated
        
        # 可选：打印进度
        if total_steps % 24 == 0: # 每24步(6小时)打印一次
            print(f"Step {total_steps}: Reward so far = {total_reward:.2f}")

    end_time = time.time()
    
    print("-" * 40)
    print(f"✅ Baseline Finished.")
    print(f"⏱️ Time Elapsed: {end_time - start_time:.2f} seconds")
    print(f"🏆 Final Score (Total Reward): {total_reward:.2f}")
    print("-" * 40)

if __name__ == "__main__":
    run_passive_baseline()