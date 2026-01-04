import gymnasium as gym
import numpy as np
import pandas as pd
import random
from datetime import timedelta
from collections import defaultdict
from src.utils import calculate_distance
from src.config import TRAVEL_COST_PER_KM, LABOR_COST_PER_STEP

class Crew:
    """
    Represents a single utility repair crew.
    """
    def __init__(self, crew_id, station_name, location):
        self.crew_id = crew_id
        self.station_name = station_name
        self.location = location  # (Longitude, Latitude)
        self.status = "idle"      # 'idle', 'traveling', 'repairing'
        self.arrival_time = None
        self.available_at = None

    def is_available(self, current_time):
        if self.status == "idle":
            return True
        if self.available_at and current_time >= self.available_at:
            self.status = "idle"
            return True
        return False

    def dispatch(self, destination, current_time, travel_time_hours, repair_time_minutes=30):
        self.status = "busy"
        self.location = destination
        self.arrival_time = current_time + timedelta(hours=travel_time_hours)
        self.available_at = self.arrival_time + timedelta(minutes=repair_time_minutes)

    def reposition(self, destination, current_time, travel_distance_km):
        # Assume average speed of 60 km/h
        travel_time_hours = travel_distance_km / 60.0
        self.status = "idle" 
        self.location = destination

    def get_current_position(self, current_time):
        return self.location

def assign_best_outage(crew, outages, current_time, weights=None):
    """
    Heuristic to assign the best outage to a crew based on distance and priority.
    """
    if not outages:
        return None
        
    if weights is None:
        # Default heuristic weights
        weights = {"travel": 0.5, "wait": 0.3, "pop": 0.2}

    best_outage = None
    min_score = float('inf')

    for outage in outages:
        dist = calculate_distance(crew.get_current_position(current_time), outage['location'])
        
        # Normalize distance (approx max 100km)
        norm_dist = min(dist / 100.0, 1.0)
        
        # Wait time penalty
        wait_minutes = (current_time - outage['time_reported']).total_seconds() / 60.0
        norm_wait = min(wait_minutes / 300.0, 1.0) # Cap at 5 hours
        
        # Population priority (inverse because higher density is better)
        norm_pop = 1.0 / (outage.get('pop_density', 10) + 1)

        score = (weights["travel"] * norm_dist) - (weights["wait"] * norm_wait) + (weights["pop"] * norm_pop)
        
        if score < min_score:
            min_score = score
            best_outage = outage
            
    return best_outage


class CrewDispatchEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Simulates crew dispatching for power outage restoration.
    """
    def __init__(self, towns, stations, outages_df, forecast_df, crew_per_station=3, forecast_dict=None):
        super(CrewDispatchEnv, self).__init__()
        
        self.towns = towns  # Dict: {name: (lon, lat)}
        self.stations = stations # Dict: {name: (lon, lat)}
        self.outages_df = outages_df
        self.forecast_df = forecast_df
        self.forecast_dict = forecast_dict or {}
        self.crew_per_station = crew_per_station

        # Define Action and Observation Space
        n_crews = len(stations) * crew_per_station
        
        # Action: Where each crew should go (index of towns + 1 for stay)
        self.action_space = gym.spaces.MultiDiscrete([len(towns) + 1] * n_crews)

        # Observation: Forecast probs + Crew states
        self.observation_space = gym.spaces.Dict({
            "forecast_probs": gym.spaces.Box(low=0, high=1, shape=(len(towns) * 8 * 2,), dtype=np.float32),
            "crew_states": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_crews, 5), dtype=np.float32)
        })
        
        self.crews = []
        self.current_outages = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Set time window
        if not self.forecast_df.empty:
            forecast_times = self.forecast_df["datetime"]
            self.time = forecast_times.min().floor("15min")
            self.end_time = forecast_times.max().ceil("15min")
        else:
            # Fallback if no forecast
            self.time = self.outages_df["Date and Time Out"].min().floor("15min")
            self.end_time = self.outages_df["Date and Time Out"].max().ceil("15min")
            
        self.prev_time = self.time - timedelta(minutes=15)
        
        # Initialize Crews
        self._init_crews(self.crew_per_station)
        
        # Prepare Outage Schedule
        self.schedule = self._get_outage_schedule()
        self.current_outages = []
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. Calculate Agent Movement Costs (Travel Cost)
        # We calculate this BEFORE applying the action to know how much distance they intend to move.
        move_distance_km = 0.0
        if action is not None:
            for crew, dest_idx in zip(self.crews, action):
                # If crew is idle and ordered to move to a town (dest_idx > 0)
                if crew.status == "idle" and dest_idx > 0:
                    town_name = list(self.towns.keys())[dest_idx - 1]
                    dest_loc = self.towns[town_name]
                    # Calculate distance from current location to destination
                    dist = calculate_distance(crew.location, dest_loc)
                    move_distance_km += dist

        # 2. Apply Agent Actions (Physical Repositioning)
        self._apply_agent_action(action)

        total_reward = 0.0
        total_labor_cost = 0.0
        total_outage_penalty = 0.0
        
        # 3. Advance Simulation (4 steps of 15 mins = 1 hour)
        for _ in range(4):
            self.time += timedelta(minutes=15)
            
            # --- A. Advance Physics & Auto-Dispatch ---
            num_resolved_this_step = self._advance_simulation()
            
            # --- B. Calculate Outage & Repair Reward ---
            step_reward, step_outage_penalty = self._calculate_base_reward(num_resolved_this_step)
            total_reward += step_reward
            total_outage_penalty += step_outage_penalty
            
            # --- C. Calculate Labor Cost (Shift Pay) ---
            # Crews not at a station cost money to maintain in the field.
            step_labor_cost = 0.0
            for crew in self.crews:
                is_at_station = False
                for s_coords in self.stations.values():
                    # Check if crew is near any station (within 0.5km tolerance)
                    if calculate_distance(crew.location, s_coords) < 0.5:
                        is_at_station = True
                        break
                
                if not is_at_station:
                    step_labor_cost += LABOR_COST_PER_STEP
            
            total_labor_cost += step_labor_cost
            
            self.prev_time = self.time

        # 4. Final Reward Assembly
        # Start with Base Reward (Outage Penalties + Fix Bonuses)
        # Subtract Travel Cost (Fuel/Wear)
        # Subtract Labor Cost (Overtime/Field Pay)
        
        travel_penalty = move_distance_km * TRAVEL_COST_PER_KM
        
        total_reward -= travel_penalty
        total_reward -= total_labor_cost

        terminated = self.time >= self.end_time
        truncated = False
        
        # Info for TensorBoard tracking
        info = {
            "travel_penalty": travel_penalty,
            "labor_penalty": total_labor_cost,
            "outage_penalty": total_outage_penalty,
            "total_cost": -total_reward # approximate cost view
        }
        
        return self._get_obs(), total_reward, terminated, truncated, info

    def _init_crews(self, per_station):
        self.crews = []
        cid = 1
        for name, loc in self.stations.items():
            for _ in range(per_station):
                self.crews.append(Crew(cid, name, loc))
                cid += 1

    def _get_outage_schedule(self):
        return sorted([
            {
                "location": (row["Longitude"], row["Latitude"]),
                "time_reported": row["Date and Time Out"],
                "pop_density": row.get("pop_density", 1) * 10,
                "status": "pending",
                "assigned_crew": None,
                "resolve_time": self.end_time # Default to end
            }
            for _, row in self.outages_df.iterrows()
        ], key=lambda o: o["time_reported"])

    def _advance_simulation(self):
        """
        Advances the simulation physics. 
        Returns: The integer number of outages that were resolved in this step.
        """
        # Add new outages
        new_outages = [o for o in self.schedule if o["time_reported"] <= self.time]
        self.schedule = [o for o in self.schedule if o["time_reported"] > self.time]
        self.current_outages.extend(new_outages)

        # Update crew status
        for crew in self.crews:
            crew.is_available(self.time) # Checks if repair is done

        # Check resolved outages
        resolved = []
        for outage in self.current_outages:
            if outage["status"] == "assigned" and outage["resolve_time"] <= self.time:
                outage["status"] = "resolved"
                resolved.append(outage)
        
        # Remove resolved outages from active list
        for r in resolved:
            self.current_outages.remove(r)
            
        num_resolved = len(resolved)

        # Auto-dispatch logic for available crews
        available_crews = [c for c in self.crews if c.status == "idle"]
        pending_outages = [o for o in self.current_outages if o["status"] == "pending"]

        for crew in available_crews:
            if not pending_outages:
                break
            
            best_outage = assign_best_outage(crew, pending_outages, self.time)
            
            if best_outage:
                dist = calculate_distance(crew.location, best_outage['location'])
                travel_time = dist / 60.0
                crew.dispatch(best_outage['location'], self.time, travel_time)
                
                best_outage["status"] = "assigned"
                best_outage["assigned_crew"] = crew
                best_outage["resolve_time"] = crew.available_at
                
                pending_outages.remove(best_outage)
                
        return num_resolved

    def _apply_agent_action(self, action):
        if action is None:
            return
        for crew, dest_idx in zip(self.crews, action):
            if crew.status == "idle" and dest_idx > 0:
                town_name = list(self.towns.keys())[dest_idx - 1]
                dest_loc = self.towns[town_name]
                dist = calculate_distance(crew.location, dest_loc)
                crew.reposition(dest_loc, self.time, dist)

    def _get_obs(self):
        # 1. Crew States
        crew_states = []
        for c in self.crews:
            x, y = c.location
            home_x, home_y = self.stations[c.station_name]
            time_avail = (c.available_at - self.time).total_seconds() / 60 if c.available_at and c.available_at > self.time else 0
            crew_states.append([x, y, int(c.status != "idle"), time_avail, home_x])

        # 2. Forecast Probs
        town_list = list(self.towns.keys())
        forecast_tensor = np.zeros((len(town_list), 8, 2))
        
        for i, town in enumerate(town_list):
            for lead in range(8):
                prob = self.forecast_dict.get(self.time, {}).get(town, {}).get(lead, 0.0)
                forecast_tensor[i, lead, 0] = prob
                forecast_tensor[i, lead, 1] = lead / 7.0

        return {
            "forecast_probs": forecast_tensor.flatten().astype(np.float32),
            "crew_states": np.array(crew_states, dtype=np.float32)
        }

    def _calculate_base_reward(self, n_resolved=0):
        """
        Calculates the Operational Reward (Outages & Repairs only).
        Costs (Travel & Labor) are handled in step().
        """
        reward = 0.0
        outage_penalty = 0.0
        
        # 1. The Stick: Penalty for active outages (Pop Density * Time)
        # Factor = 1.0 (Strong Penalty to match real CMO impact)
        PENALTY_FACTOR = 1.0
        
        for o in self.current_outages:
            if o["status"] in ["pending", "assigned"]:
                penalty = o["pop_density"] * PENALTY_FACTOR
                outage_penalty += penalty
                reward -= penalty
        
        # 2. The Carrot: Big Bonus for finishing repairs
        # Must be large enough to offset Travel + Labor costs
        COMPLETION_BONUS = 100.0
        
        if n_resolved > 0:
            reward += (n_resolved * COMPLETION_BONUS)
        
        return reward, outage_penalty