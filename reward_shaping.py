import gymnasium as gym
import numpy as np

class FlappyBirdRewardShaping(gym.Wrapper):
    """
    Reward Shaping tinh giản (Không dùng code dẫn đường):
    Tập trung vào việc 'Khuếch đại tín hiệu' thành công để agent học nhanh hơn
    so với mặc định.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Giữ nguyên biến để tương thích với agent.py
        self.last_score = 0
        self.frames_alive = 0
        self.last_horizontal_distance = None
        
    def reset(self, **kwargs):
        self.last_score = 0
        self.frames_alive = 0
        self.last_horizontal_distance = None
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Cập nhật biến tracking
        if not terminated:
            self.frames_alive += 1
            
        if reward >= 1.0:
            self.last_score += 1

        # --- LOGIC REWARD MỚI ---
        shaped_reward = 0.0

        # 1. REWARD VƯỢT ỐNG (QUAN TRỌNG NHẤT)
        # Thay vì +1.0 như mặc định, ta tăng lên +5.0
        # Lý do: Làm cho khoảnh khắc thành công trở nên "đáng nhớ" hơn nhiều
        # so với việc chỉ sống sót thêm vài frame.
        if reward >= 1.0:
            shaped_reward = 5.0 

        # 2. REWARD SỐNG SÓT (DUY TRÌ ĐỘNG LỰC)
        # Giữ mức +0.1 như mặc định. 
        # Nếu agent sống 50 frames (bay từ ống này sang ống kia), nó được +5.0
        # Tổng reward vượt 1 ống = 5.0 (vượt) + 5.0 (bay) = 10.0 -> Cân bằng tốt.
        elif reward > 0 and reward < 1.0:
            shaped_reward = 0.1

        # 3. HÌNH PHẠT KHI CHẾT HOẶC CHẠM TRẦN
        # Giữ nguyên -1.0. 
        # Đừng phạt nặng hơn (-3, -5) vì sẽ gây ra hiện tượng "Risk Aversion" (sợ rủi ro).
        # Khi phạt nặng, agent thà rơi tự do chết nhanh còn hơn bay lên để rồi va chạm.
        elif reward <= -0.5:
            shaped_reward = -1.0

        return obs, shaped_reward, terminated, truncated, info