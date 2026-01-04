from src.env import CrewDispatchEnv
from src.utils import calculate_distance # 确保引入计算距离的函数

class CrewDispatchEvalEnv(CrewDispatchEnv):
    def step(self, action):
        # 1. 记录移动距离 (这需要在 apply_agent_action 之前做，或者修改 apply_agent_action 返回距离)
        total_distance_km = 0.0
        
        # 这里我们需要稍微 hack 一下 apply_agent_action 或者手动计算
        # 简单起见，我们在 apply action 之前计算意图
        if action is not None:
            for crew, dest_idx in zip(self.crews, action):
                if crew.status == "idle" and dest_idx > 0:
                    town_name = list(self.towns.keys())[dest_idx - 1]
                    dest_loc = self.towns[town_name]
                    dist = calculate_distance(crew.location, dest_loc)
                    total_distance_km += dist

        # 2. 也是很重要的一点：自动调度 (Auto-Dispatch) 产生的距离也要算！
        # 这需要修改 _advance_simulation 里的逻辑让它返回距离，或者我们在外部估算。
        # 为了不伤筋动骨改代码，我们这里先只加一个“显式调度惩罚”。
        
        # ... 调用父类的 step ...
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 3. 在 Reward 里扣除运营成本
        # 假设：每公里成本 = 10 CMO (即跑 1公里 等同于 10个人停电15分钟的损失)
        # 这个系数需要根据业务调节
        TRAVEL_COST_FACTOR = 10.0 
        
        operational_cost = total_distance_km * TRAVEL_COST_FACTOR
        
        # 更新 Reward (原来的 reward 是 -CMO)
        reward -= operational_cost
        
        # 更新 info 以便观察
        info['cmo'] = info.get('outage_penalty', 0) # 这里的 key 可能要根据你实际代码调整
        info['op_cost'] = operational_cost
        info['total_cost'] = -reward
        
        return obs, reward, terminated, truncated, info