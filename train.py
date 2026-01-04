import argparse
import time
import os
from collections import defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.config import *
from src.data_loader import load_and_process_data
from src.env import CrewDispatchEnv

def build_forecast_index(forecast_df):
    """Helper to build the fast lookup dictionary for weather."""
    print("Building forecast lookup index...")
    forecast_dict = defaultdict(lambda: defaultdict(dict))
    if not forecast_df.empty:
        for _, row in forecast_df.iterrows():
            forecast_dict[row["datetime"]][row["town"]][row["forecast_hour"]] = row["predicted_prob"]
    return forecast_dict

def make_env(start_date, end_date, town_map_override=None):
    """
    Factory function to create an environment for a specific date range.
    
    Args:
        town_map_override: Critical! We must pass the MASTER town map to the 
                           test env so the action space sizes match.
    """
    # 1. Load Data for this specific range
    outages, forecast_df, town_names = load_and_process_data(
        OUTAGE_FILE, DENSITY_FILE, FORECAST_FILE, start_date, end_date
    )

    # 2. Build or Use Map
    if town_map_override:
        towns_map = town_map_override
    else:
        # Build master map from data
        coords_df = outages.drop_duplicates(subset="City/Town")[["City/Town", "Longitude", "Latitude"]]
        towns_map = {row["City/Town"]: (row["Longitude"], row["Latitude"]) 
                     for _, row in coords_df.iterrows() if row["City/Town"] in town_names}

    # 3. Build Forecast Dict
    forecast_dict = build_forecast_index(forecast_df)

    # 4. Create Env
    env = CrewDispatchEnv(
        towns=towns_map,
        stations=STATIONS,
        outages_df=outages,
        forecast_df=forecast_df,
        crew_per_station=CREWS_PER_STATION,
        forecast_dict=forecast_dict
    )
    return Monitor(env), towns_map

def main():
    parser = argparse.ArgumentParser(description="Train RL Agent with Christmas Evaluation")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS, help="Total training timesteps")
    args = parser.parse_args()

    print("🚀 Initializing Training Pipeline...")

    # ==========================================
    # 1. Setup TRAIN Environment (Full Year)
    # ==========================================
    print(f"\n[1/3] Setting up TRAINING Environment ({TRAIN_START_DATE} to {TRAIN_END_DATE})...")
    train_env, master_town_map = make_env(TRAIN_START_DATE, TRAIN_END_DATE)

    # ==========================================
    # 2. Setup EVAL Environment (Christmas Storm)
    # ==========================================
    # We use specific dates for the "Christmas Storm" scenario
    # IMPORTANT: We pass master_town_map so the test env has the same "Action Space" size
    print(f"\n[2/3] Setting up EVALUATION Environment (Christmas Scenario)...")
    eval_env, _ = make_env("2022-12-22", "2022-12-26", town_map_override=master_town_map)

    # ==========================================
    # 3. Setup Callbacks & Agent
    # ==========================================
    print("\n[3/3] Configuring Agent and Callbacks...")
    
    # EvalCallback: Runs every 10k steps to test the model on Christmas data
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/results',
        eval_freq=10000,        # Test every 10,000 steps
        n_eval_episodes=3,      # Run 3 episodes to get a stable average
        deterministic=True,     # Use best strategy (not random) for testing
        render=False
    )

    # PPO Agent with TensorBoard logging
    model = PPO(
        "MultiInputPolicy", 
        train_env, 
        verbose=1,
        learning_rate=1e-4,             # Lower LR for stability
        tensorboard_log="./tb_logs"     # Enable TensorBoard
    )

    # ==========================================
    # 4. Start Training
    # ==========================================
    print(f"\n🔥 Starting Training for {args.steps} steps...")
    start_time = time.time()
    
    model.learn(total_timesteps=args.steps, callback=eval_callback)
    
    end_time = time.time()
    print(f"✅ Training finished in {(end_time - start_time)/60:.2f} minutes.")
    
    # Save Final Model
    # 获取当前时间，例如 20260103_2130
    timestamp = time.strftime("%Y%m%d_%H%M")
    save_name = f"{MODEL_SAVE_PATH}_{timestamp}"
    model.save(save_name)
    print(f"Final Model saved to {save_name}.zip")

if __name__ == "__main__":
    main()