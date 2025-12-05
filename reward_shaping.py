# reward_shaping.py
import gymnasium as gym
import numpy as np

class FlappyBirdRewardShaping(gym.Wrapper):
    """
    Reward Shaping cho Flappy Bird để agent học nhanh hơn
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0
        self.frames_alive = 0
        self.last_horizontal_distance = None
        
    def reset(self, **kwargs):
        """Reset khi bắt đầu episode mới"""
        self.last_score = 0
        self.frames_alive = 0
        self.last_horizontal_distance = None
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Thực hiện action và reshape reward"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Lưu reward gốc để debug
        original_reward = reward
        shaped_reward = 0.0
        
        # ===== 1.  REWARD CHO VƯỢT ỐNG =====
        # Thưởng RẤT LỚN khi vượt ống (động lực chính)
        if reward >= 1.0:  # Vượt ống
            shaped_reward += 3.5  # Tăng từ 1. 0 → 3.5
            self. last_score += 1
            
        # ===== 2.  REWARD CHO SỐNG SÓT =====
        # Thưởng cho việc sống lâu (nhưng không quá nhiều)
        elif reward > 0 and reward < 1.0:  # Frame reward (+0.1)
            self.frames_alive += 1
            shaped_reward += 0.1  # Giữ nguyên
            
            # Bonus nhỏ nếu sống lâu
            if self.frames_alive % 100 == 0:  # Mỗi 50 frames
                shaped_reward += 0.5
                
        # ===== 3. PENALTY KHI CHẾT =====
        elif reward <= -0.5:  # Chết hoặc chạm trần
            if reward == -1.0:  # Chết (va chạm)
                shaped_reward = -1.5  # Tăng penalty từ -1.0 → -10.0
                
                # Penalty NẶNG hơn nếu chết sớm (chưa vượt ống nào)
                if self.last_score == 0:
                    shaped_reward -= 2.0  # Tổng -3.5
                    
                # Penalty nhẹ hơn nếu đã vượt nhiều ống
                elif self.  last_score >= 5:
                    shaped_reward += 2.0  # Giảm penalty: -1.5
                    
            elif reward == -0.5:  # Chạm trần
                shaped_reward = -3.0  # Tăng penalty từ -0. 5 → -3.0
        
        # ===== 4. REWARD DỰA TRÊN VỊ TRÍ (OPTIONAL - NÂNG CAO) =====
        # Thưởng nếu bird ở vị trí tốt (gần giữa màn hình)
        # Observation: [bird_y, bird_velocity, pipe_horizontal_distance, pipe_top_y, pipe_bottom_y, ...]
        if len(obs) >= 5:
            bird_y = obs[0]
            pipe_top_y = obs[3] if len(obs) > 3 else 0
            pipe_bottom_y = obs[4] if len(obs) > 4 else 1
            
            # Tính vị trí lý tưởng (giữa 2 ống)
            pipe_center_y = (pipe_top_y + pipe_bottom_y) / 2
            distance_to_center = abs(bird_y - pipe_center_y)
            
            # Thưởng nhỏ nếu bird gần vị trí lý tưởng
            if distance_to_center < 0.1:
                shaped_reward += 0.2
            elif distance_to_center < 0.2:
                shaped_reward += 0.1
        
        # Debug: In ra để kiểm tra (comment out sau khi test)
        # if terminated or shaped_reward > 1:
        #     print(f"Original: {original_reward:.1f} → Shaped: {shaped_reward:.1f}, Score: {self.last_score}, Frames: {self.frames_alive}")
        
        return obs, shaped_reward, terminated, truncated, info