import argparse
import time
from collections import defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.config import *
from src.data_loader import load_and_process_data
from src.env import CrewDispatchEnv

def main():
    parser = argparse.ArgumentParser(description="Train or Test RL Agent for Crew Dispatch")
    parser.add_argument("--test", action="store_true", help="Run in test mode (short duration)")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS, help="Total training timesteps")
    args = parser.parse_args()

    # 1. Determine Date Range
    if args.test:
        print("🚀 Running in TEST mode (Christmas Storm Scenario)...")
        start_date = TEST_START_DATE
        end_date = TEST_END_DATE
        total_steps = 10_000 # Short run for testing
    else:
        print("🏋️ Running in TRAIN mode (Full Year)...")
        start_date = TRAIN_START_DATE
        end_date = TRAIN_END_DATE
        total_steps = args.steps

    # 2. Load Data
    outages, forecast_df, town_names = load_and_process_data(
        OUTAGE_FILE, DENSITY_FILE, FORECAST_FILE, start_date, end_date
    )
    
    # 3. Build Coordinates Map
    # Extract coordinates from outage data for all towns
    coords_df = outages.drop_duplicates(subset="City/Town")[["City/Town", "Longitude", "Latitude"]]
    towns_map = {row["City/Town"]: (row["Longitude"], row["Latitude"]) 
                 for _, row in coords_df.iterrows() if row["City/Town"] in town_names}

    # 4. Build Forecast Dictionary for fast lookup
    print("Building forecast lookup index...")
    forecast_dict = defaultdict(lambda: defaultdict(dict))
    if not forecast_df.empty:
        for _, row in forecast_df.iterrows():
            forecast_dict[row["datetime"]][row["town"]][row["forecast_hour"]] = row["predicted_prob"]

    # 5. Initialize Environment
    env = CrewDispatchEnv(
        towns=towns_map,
        stations=STATIONS,
        outages_df=outages,
        forecast_df=forecast_df,
        crew_per_station=CREWS_PER_STATION,
        forecast_dict=forecast_dict
    )
    env = Monitor(env) # Wrap for logging

    # 6. Initialize Model
    print("Initializing PPO Agent...")
    model = PPO("MultiInputPolicy", env, verbose=1)

    # 7. Start Training
    start_time = time.time()
    model.learn(total_timesteps=total_steps)
    end_time = time.time()

    print(f"✅ Training finished in {(end_time - start_time)/60:.2f} minutes.")
    
    # 8. Save Model
    save_name = f"{MODEL_SAVE_PATH}_{'test' if args.test else 'full'}"
    model.save(save_name)
    print(f"Model saved to {save_name}.zip")

if __name__ == "__main__":
    main()