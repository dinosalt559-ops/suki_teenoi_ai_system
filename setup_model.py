import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

def train_and_save_model():
    # ข้อมูลตัวอย่างสำหรับเทรนโมเดลพยากรณ์จำนวนลูกค้า
    # Features: [is_weekend, is_public_holiday, is_payday_season, rainfall_mm, nearby_major_event]
    X_train = np.array([
        [0, 0, 0, 0.0, 0],
        [1, 0, 0, 5.2, 0],
        [0, 1, 1, 0.0, 1],
        [1, 1, 1, 12.5, 1],
        [0, 0, 1, 0.0, 0],
        [1, 0, 0, 0.0, 1]
    ])
    # Target: จำนวนลูกค้าที่มาใช้บริการ
    y_train = np.array([350, 480, 600, 750, 420, 520])

    model = LinearRegression()
    model.fit(X_train, y_train)

    # บันทึกโมเดลลงไฟล์ model.pkl
    model_filename = "model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    
    print(f"✅ เทรนและบันทึกไฟล์โมเดลสำเร็จ: {model_filename}")

if __name__ == "__main__":
    train_and_save_model()