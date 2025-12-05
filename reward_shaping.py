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
        
    def reset(self, **kwargs):
        """Reset khi bắt đầu episode mới"""
        self.last_score = 0
        self.frames_alive = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Thực hiện action và reshape reward"""
        obs, reward, terminated, truncated, info = self. env.step(action)
        
        original_reward = reward
        shaped_reward = 0.0
        
        # ===== 1.  REWARD CHO VƯỢT ỐNG =====
        # ĐỘNG LỰC CHÍNH - thưởng lớn
        if reward >= 1.0:  
            shaped_reward += 10.0  # GIẢM từ 15 → 10
            self.last_score += 1
            
        # ===== 2.  REWARD CHO SỐNG SÓT =====
        # Thưởng nhỏ cho mỗi frame còn sống
        elif reward > 0 and reward < 1.0:  
            self.frames_alive += 1
            shaped_reward += 0.1  # Giữ nguyên
            
        # ===== 3.  PENALTY KHI CHẾT =====
        elif reward <= -0.5:  
            if reward == -1.0:  # Chết (va chạm)
                # GIẢM PENALTY - không khắc nghiệt quá
                shaped_reward = -5.0  # GIẢM từ -10 → -5
                
                # Penalty thêm nếu chết quá sớm
                if self.frames_alive < 20:  # Chết trong 20 frames đầu
                    shaped_reward -= 3.0  # Tổng -8
                    
            elif reward == -0.5:  # Chạm trần
                shaped_reward = -2.0  # GIẢM từ -3 → -2
        
        # XÓA PHẦN POSITION REWARD (dòng 64-80) - gây nhiễu! 
        
        return obs, shaped_reward, terminated, truncated, info