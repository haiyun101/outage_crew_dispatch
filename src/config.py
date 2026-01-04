import os

# --- Paths ---
DATA_DIR = "data"
OUTAGE_FILE = os.path.join(DATA_DIR, "outage_western_100_cleaned.csv")
DENSITY_FILE = os.path.join(DATA_DIR, "western_100_spatial_features.csv")
FORECAST_FILE = os.path.join(DATA_DIR, "probability_forecasts_stdcl_2022.csv")
MODEL_SAVE_PATH = "ppo_crew_dispatch"

# --- Simulation Settings ---
# Station coordinates (Latitude, Longitude) -> Note: The env uses (Lon, Lat) order for x,y
STATIONS = {
    "Greenfield": (-72.599, 42.587),
    "Amherst": (-72.5199, 42.3732),
    "Northampton": (-72.6402, 42.3251),
}

# Default dates for training
TRAIN_START_DATE = "2022-01-01"
TRAIN_END_DATE = "2023-01-01"

# Default dates for testing (The "Christmas Storm" scenario)
TEST_START_DATE = "2022-12-22"
TEST_END_DATE = "2022-12-26"

# --- Hyperparameters ---
CREWS_PER_STATION = 6  # Total agents = 3 stations * 6 crews = 18
TOTAL_TIMESTEPS = 500_000

# 假设：让一个维修队在外面待命 15分钟 的成本 (工资+损耗)
# 等同于 让 5 个用户停电 15分钟 的痛苦
# 这个值不能太高，否则 Agent 会为了省工资而拒绝修电
LABOR_COST_PER_STEP = 5.0 

# 之前的移动成本
TRAVEL_COST_PER_KM = 1.0