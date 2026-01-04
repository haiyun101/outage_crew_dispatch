import pandas as pd
import numpy as np
import time
from stable_baselines3 import PPO
from src.eval_env import CrewDispatchEvalEnv
from src.data_loader import load_and_process_data
from src.config import *
from collections import defaultdict

# --- Define Baselines ---

class RandomAgent:
    """Baseline 1: Randomly moves crews around."""
    def __init__(self, action_space):
        self.action_space = action_space
        
    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None

class GreedyAgent:
    """
    Baseline 2: The 'Do Nothing' / Reactive Agent.
    
    It outputs Action=0 (Stay) for all crews.
    Because the environment has built-in logic to 'auto-dispatch to nearest outage'
    when a crew is idle, Action=0 effectively means "Stay at station until needed, 
    then go to nearest job."
    """
    def __init__(self, num_crews):
        self.num_crews = num_crews
        
    def predict(self, obs, deterministic=True):
        # Return [0, 0, 0, ...] -> All crews stay/wait for assignments
        return [0] * self.num_crews, None

# --- Evaluation Loop ---

def run_evaluation(agent_name, agent, env, n_episodes=3):
    """Runs the agent in the environment and tracks CMO."""
    print(f"🧐 Evaluating {agent_name}...")
    total_cmos = []
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_cmo = 0.0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract CMO from info dict (preferred) or infer from reward
            # Since reward = -CMO in our EvalEnv:
            step_cmo = -reward 
            episode_cmo += step_cmo
            
        total_cmos.append(episode_cmo)
        print(f"   > Episode {i+1}: Total CMO = {episode_cmo:,.0f}")
        
    avg_cmo = np.mean(total_cmos)
    print(f"   👉 Average CMO: {avg_cmo:,.0f}\n")
    return avg_cmo

def main():
    # ==========================================
    # 1. 关键修复：构建“全量地图” (Master Map)
    # ==========================================
    print("🗺️ Building Master Town Map from Full Training Data (to match model shape)...")
    # 我们加载训练期的数据，仅仅是为了拿到所有城镇的列表和坐标
    # 这样能保证 Evaluation 环境的 Observation Space 和 Training 时一模一样 (1440,)
    train_outages, _, train_town_names = load_and_process_data(
        OUTAGE_FILE, DENSITY_FILE, FORECAST_FILE, TRAIN_START_DATE, TRAIN_END_DATE
    )
    
    coords_df = train_outages.drop_duplicates(subset="City/Town")[["City/Town", "Longitude", "Latitude"]]
    master_towns_map = {row["City/Town"]: (row["Longitude"], row["Latitude"]) 
                        for _, row in coords_df.iterrows() if row["City/Town"] in train_town_names}
    
    print(f"   ✅ Master Map loaded with {len(master_towns_map)} towns.")

    # ==========================================
    # 2. 加载“圣诞风暴”评估数据
    # ==========================================
    print("\n🎄 Loading Christmas Storm Data (2022-12-22 to 2022-12-26)...")
    eval_outages, eval_forecast, _ = load_and_process_data(
        OUTAGE_FILE, DENSITY_FILE, FORECAST_FILE, "2022-12-22", "2022-12-26"
    )
    
    # 构建天气查询字典
    forecast_dict = defaultdict(lambda: defaultdict(dict))
    if not eval_forecast.empty:
        for _, row in eval_forecast.iterrows():
            forecast_dict[row["datetime"]][row["town"]][row["forecast_hour"]] = row["predicted_prob"]

    # ==========================================
    # 3. 初始化评估环境 (使用 Master Map!)
    # ==========================================
    env = CrewDispatchEvalEnv(
        towns=master_towns_map,  # 👈 关键点：传入 90 个城镇的全量地图，而不是 35 个
        stations=STATIONS,
        outages_df=eval_outages,
        forecast_df=eval_forecast,
        crew_per_station=CREWS_PER_STATION,
        forecast_dict=forecast_dict
    )
    
    num_crews = len(env.action_space.nvec)
    print(f"   ✅ Environment initialized. Action Space size: {num_crews} crews.")

    # ==========================================
    # 4. 加载选手 (Agents)
    # ==========================================
    agents = {}
    
    # A. Random
    agents["Random"] = RandomAgent(env.action_space)
    
    # B. Greedy (The benchmark)
    agents["Greedy (Standard)"] = GreedyAgent(num_crews)
    
    # C. Your PPO Model
    print("\n🤖 Loading Trained PPO Model...")
    try:
        # 加载模型
        ppo_model = PPO.load("ppo_crew_dispatch_final.zip")
        agents["PPO AI"] = ppo_model
    except FileNotFoundError:
        print("⚠️ Model file not found. Skipping PPO evaluation.")

    # ==========================================
    # 5. 开始比赛 (Tournament)
    # ==========================================
    results = {}
    print("\n" + "="*40)
    print("🏁 STARTING TOURNAMENT 🏁")
    print("="*40 + "\n")
    
    for name, agent in agents.items():
        score = run_evaluation(name, agent, env, n_episodes=3)
        results[name] = score

    # 6. Final Report
    print("="*50)
    print("📊 FINAL RESULTS (Lower CMO is Better)")
    print("="*50)
    
    # Sort by CMO (ascending is better)
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
    
    baseline = results.get("Greedy (Standard)", None)
    
    for name, score in sorted_results.items():
        diff_str = ""
        if baseline and name != "Greedy (Standard)":
            pct_change = ((score - baseline) / baseline) * 100
            if pct_change < 0:
                diff_str = f"(✅ Improved by {abs(pct_change):.1f}%)"
            else:
                diff_str = f"(❌ Worsened by {pct_change:.1f}%)"
        
        print(f"{name.ljust(20)}: {score:,.0f} CMO {diff_str}")

if __name__ == "__main__":
    main()