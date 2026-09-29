import os
import pickle
import numpy as np
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Suki Teenoi AI Inventory", version="9.0 - Multi-Branch")

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    branch_id: int
    target_date: str
    is_weekend: int = 0
    is_public_holiday: int = 0
    is_payday_season: int = 0
    rainfall_mm: float = 0.0
    nearby_major_event: int = 0

# 📍 ฐานข้อมูลพิกัดแต่ละสาขา สำหรับดึงสภาพอากาศจริงเฉพาะพื้นที่
BRANCH_LOCATIONS = {
    1: {"name": "สาขา ธัญบุรี (คลอง 7)", "lat": 14.06, "lon": 100.73, "base_cust": 400},
    2: {"name": "สาขา รังสิต (ฟิวเจอร์พาร์ค)", "lat": 13.98, "lon": 100.61, "base_cust": 650},
    3: {"name": "สาขา อนุสาวรีย์ชัยสมรภูมิ", "lat": 13.76, "lon": 100.53, "base_cust": 800},
    4: {"name": "สาขา บางนา (เซ็นทรัลบางนา)", "lat": 13.66, "lon": 100.63, "base_cust": 550}
}

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    np.random.seed(42)
    n_samples = 1000
    is_weekend = np.random.randint(0, 2, n_samples)
    is_holiday = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    is_payday = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    rainfall = np.random.exponential(scale=5.0, size=n_samples)
    event = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    
    y_train = (
        300 + (is_weekend * 180) + (is_holiday * 150) + (is_payday * 120) 
        - (rainfall * 4.5) + (event * 250) + np.random.normal(0, 25, n_samples)
    )
    y_train = np.clip(y_train, 150, 1500)
    X_train = np.column_stack((is_weekend, is_holiday, is_payday, rainfall, event))
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

with open(MODEL_PATH, "rb") as f:
    ml_model = pickle.load(f)

# 📦 ฐานข้อมูลวัตถุดิบ 50 รายการ
raw_materials = [
    {"id": "M01", "name": "เนื้อวัวสไลด์ (Sliced Beef)", "stock": 45.0, "rop": 50.0, "eoq": 80.0, "price_per_kg": 220.0},
    {"id": "M02", "name": "เนื้อริบอายสไลด์ (Ribeye Sliced)", "stock": 20.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 320.0},
    {"id": "M03", "name": "เนื้อใบพาย (Blade Beef)", "stock": 25.0, "rop": 35.0, "eoq": 60.0, "price_per_kg": 270.0},
    {"id": "M04", "name": "หมูสามชั้นสไลด์ (Pork Belly)", "stock": 60.0, "rop": 40.0, "eoq": 70.0, "price_per_kg": 150.0},
    {"id": "M05", "name": "สันคอหมูสไลด์ (Sliced Pork Neck)", "stock": 35.0, "rop": 45.0, "eoq": 80.0, "price_per_kg": 160.0},
    {"id": "M06", "name": "หมูไม้ไผ่ / หมูเด้ง (Minced Pork)", "stock": 25.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 130.0},
    {"id": "M07", "name": "หมูหมักนุ่ม (Marinated Pork)", "stock": 50.0, "rop": 40.0, "eoq": 70.0, "price_per_kg": 140.0},
    {"id": "M08", "name": "เบคอน (Bacon)", "stock": 20.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 180.0},
    {"id": "M09", "name": "ตับหมู (Pork Liver)", "stock": 10.0, "rop": 15.0, "eoq": 25.0, "price_per_kg": 95.0},
    {"id": "M10", "name": "หัวใจหมู (Pork Heart)", "stock": 8.0, "rop": 12.0, "eoq": 20.0, "price_per_kg": 110.0},
    {"id": "M11", "name": "เซี่ยงจี้หมู (Pork Kidney)", "stock": 6.0, "rop": 10.0, "eoq": 15.0, "price_per_kg": 130.0},
    {"id": "M12", "name": "ไส้อ่อนหมู (Pork Intestine)", "stock": 12.0, "rop": 20.0, "eoq": 35.0, "price_per_kg": 125.0},
    {"id": "M13", "name": "หมูสามชั้นพริกไทยดำ (Black Pepper Pork)", "stock": 18.0, "rop": 25.0, "eoq": 40.0, "price_per_kg": 155.0},
    {"id": "M14", "name": "ลูกชิ้นหมูเอ็นกระดูก (Tendons Pork Ball)", "stock": 22.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 120.0},
    {"id": "M15", "name": "เกี๊ยวหมู (Pork Wonton)", "stock": 30.0, "rop": 35.0, "eoq": 60.0, "price_per_kg": 110.0},
    {"id": "S01", "name": "กุ้งสดแกะเปลือก (Shrimp)", "stock": 15.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 280.0},
    {"id": "S02", "name": "กุ้งแม่น้ำผ่าหลัง (River Prawn)", "stock": 12.0, "rop": 25.0, "eoq": 45.0, "price_per_kg": 350.0},
    {"id": "S03", "name": "ปลาหมึกกรอบ (Crispy Squid)", "stock": 18.0, "rop": 25.0, "eoq": 40.0, "price_per_kg": 150.0},
    {"id": "S04", "name": "ปลาหมึกสด (Fresh Squid)", "stock": 22.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 190.0},
    {"id": "S05", "name": "หนวดปลาหมึกยักษ์ (Giant Squid Tentacle)", "stock": 14.0, "rop": 20.0, "eoq": 35.0, "price_per_kg": 210.0},
    {"id": "S06", "name": "ปลาดอลลี่ (Pangasius Fillet)", "stock": 40.0, "rop": 35.0, "eoq": 60.0, "price_per_kg": 90.0},
    {"id": "S07", "name": "เนื้อปลาแซลมอนสไลด์ (Salmon Slice)", "stock": 8.0, "rop": 18.0, "eoq": 30.0, "price_per_kg": 450.0},
    {"id": "S08", "name": "แมงกะพรุน (Jellyfish)", "stock": 12.0, "rop": 20.0, "eoq": 30.0, "price_per_kg": 120.0},
    {"id": "S09", "name": "หอยแมลงภู่ชิลี (New Zealand Mussel)", "stock": 15.0, "rop": 22.0, "eoq": 40.0, "price_per_kg": 180.0},
    {"id": "S10", "name": "ลูกชิ้นปลาลวก (Fish Ball)", "stock": 25.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 115.0},
    {"id": "B01", "name": "ลูกชิ้นชีส (Cheese Meatballs)", "stock": 25.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 130.0},
    {"id": "B02", "name": "เต้าหู้ปลา (Fish Tofu)", "stock": 30.0, "rop": 25.0, "eoq": 45.0, "price_per_kg": 100.0},
    {"id": "B03", "name": "ปูอัด (Crab Stick)", "stock": 28.0, "rop": 25.0, "eoq": 40.0, "price_per_kg": 110.0},
    {"id": "B04", "name": "ลูกชิ้นกุ้ง (Shrimp Balls)", "stock": 15.0, "rop": 20.0, "eoq": 40.0, "price_per_kg": 140.0},
    {"id": "B05", "name": "เต้าหู้ไข่ (Egg Tofu)", "stock": 40.0, "rop": 30.0, "eoq": 60.0, "price_per_kg": 50.0},
    {"id": "B06", "name": "เต้าหู้ซีฟู้ด (Seafood Tofu)", "stock": 20.0, "rop": 25.0, "eoq": 45.0, "price_per_kg": 120.0},
    {"id": "B07", "name": "ลูกชิ้นลาวาไข่เค็ม (Salted Egg Lava Ball)", "stock": 16.0, "rop": 22.0, "eoq": 35.0, "price_per_kg": 160.0},
    {"id": "B08", "name": "ไส้กรอกชีสรมควัน (Smoked Cheese Sausage)", "stock": 22.0, "rop": 28.0, "eoq": 45.0, "price_per_kg": 145.0},
    {"id": "B09", "name": "ฟองเต้าหู้ม้วน (Fried Bean Curd Roll)", "stock": 15.0, "rop": 20.0, "eoq": 35.0, "price_per_kg": 220.0},
    {"id": "B10", "name": "ลูกชิ้นปลาหมึก (Squid Ball)", "stock": 18.0, "rop": 24.0, "eoq": 40.0, "price_per_kg": 125.0},
    {"id": "V01", "name": "ผักบุ้ง (Morning Glory)", "stock": 25.0, "rop": 35.0, "eoq": 60.0, "price_per_kg": 45.0},
    {"id": "V02", "name": "ผักกาดขาว (Chinese Cabbage)", "stock": 30.0, "rop": 40.0, "eoq": 70.0, "price_per_kg": 40.0},
    {"id": "V03", "name": "กะหล่ำปลีฝอย (Shredded Cabbage)", "stock": 15.0, "rop": 20.0, "eoq": 40.0, "price_per_kg": 35.0},
    {"id": "V04", "name": "เห็ดเข็มทอง (Enoki Mushroom)", "stock": 20.0, "rop": 30.0, "eoq": 50.0, "price_per_kg": 60.0},
    {"id": "V05", "name": "เห็ดออรินจิ (Eryngii Mushroom)", "stock": 18.0, "rop": 25.0, "eoq": 45.0, "price_per_kg": 70.0},
    {"id": "V06", "name": "เห็ดชิเมจิขาว/ดำ (Shimeji Mushroom)", "stock": 14.0, "rop": 20.0, "eoq": 35.0, "price_per_kg": 85.0},
    {"id": "V07", "name": "ข้าวโพดหวาน (Sweet Corn)", "stock": 22.0, "rop": 20.0, "eoq": 40.0, "price_per_kg": 35.0},
    {"id": "V08", "name": "ข้าวโพดอ่อน (Baby Corn)", "stock": 16.0, "rop": 22.0, "eoq": 35.0, "price_per_kg": 55.0},
    {"id": "V09", "name": "สาหร่ายวากาเมะ (Wakame Seaweed)", "stock": 8.0, "rop": 15.0, "eoq": 25.0, "price_per_kg": 150.0},
    {"id": "V10", "name": "ขึ้นฉ่ายและต้นหอม (Chinese Celery & Scallion)", "stock": 12.0, "rop": 18.0, "eoq": 30.0, "price_per_kg": 65.0},
    {"id": "C01", "name": "วุ้นเส้น (Glass Noodles)", "stock": 15.0, "rop": 20.0, "eoq": 40.0, "price_per_kg": 50.0},
    {"id": "C02", "name": "บะหมี่หยก (Jade Noodles)", "stock": 20.0, "rop": 25.0, "eoq": 50.0, "price_per_kg": 60.0},
    {"id": "C03", "name": "ม่าม่า / บะหมี่กึ่งสำเร็จรูป (Instant Noodles)", "stock": 30.0, "rop": 30.0, "eoq": 60.0, "price_per_kg": 45.0},
    {"id": "C04", "name": "ไข่ไก่ (Chicken Eggs)", "stock": 40.0, "rop": 50.0, "eoq": 80.0, "price_per_kg": 55.0},
    {"id": "C05", "name": "ชีสยืดมอซซาเรลล่า (Mozzarella Cheese)", "stock": 10.0, "rop": 20.0, "eoq": 40.0, "price_per_kg": 250.0},
    {"id": "D01", "name": "น้ำจิ้มสุกี้สูตรตี๋น้อย (Suki Sauce)", "stock": 50.0, "rop": 60.0, "eoq": 120.0, "price_per_kg": 85.0},
    {"id": "D02", "name": "น้ำจิ้มซีฟู้ดแซ่บ (Seafood Sauce)", "stock": 35.0, "rop": 45.0, "eoq": 80.0, "price_per_kg": 95.0},
    {"id": "D03", "name": "น้ำซุปดำ (Black Soup Base)", "stock": 30.0, "rop": 40.0, "eoq": 70.0, "price_per_kg": 70.0},
    {"id": "D04", "name": "น้ำซุปใส (Clear Soup Base)", "stock": 25.0, "rop": 35.0, "eoq": 60.0, "price_per_kg": 50.0},
    {"id": "D05", "name": "กระเทียมสับและพริกขี้หนูสวน (Garlic & Chili Set)", "stock": 20.0, "rop": 25.0, "eoq": 50.0, "price_per_kg": 60.0},
]

def get_weather_by_coords(lat, lon, target_date_str: str):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum,temperature_2m_max&timezone=Asia%2FBangkok&start_date={target_date_str}&end_date={target_date_str}"
    try:
        resp = requests.get(url)
        data = resp.json()
        rain = float(data['daily']['precipitation_sum'][0])
        temp = float(data['daily']['temperature_2m_max'][0])
        return rain, temp
    except Exception:
        return 0.0, 32.0 

@app.post("/api/v1/predict-and-budget")
def predict_and_budget(payload: PredictionRequest):
    try:
        target_date_obj = datetime.strptime(payload.target_date, "%Y-%m-%d")
        
        # ดึงข้อมูลพิกัดตามสาขาที่เลือก
        branch_info = BRANCH_LOCATIONS.get(payload.branch_id, BRANCH_LOCATIONS[1])
        real_rainfall, real_temp = get_weather_by_coords(branch_info["lat"], branch_info["lon"], payload.target_date)
        
        is_weekend = 1 if target_date_obj.weekday() >= 5 else 0
        is_payday = 1 if target_date_obj.day >= 28 or target_date_obj.day <= 2 else 0
        
        day_type_desc = "วันหยุดสุดสัปดาห์ (High Traffic)" if is_weekend else "วันธรรมดา (จันทร์-ศุกร์)"
        payday_desc = "ช่วงเงินเดือนออก (กำลังซื้อสูง)" if is_payday else "ช่วงกลางเดือน (การบริโภคปกติ)"
        
        if real_rainfall == 0:
            weather_desc = "ท้องฟ้าโปร่ง (ไม่มีฝนตก)"
            weather_impact = "ส่งผลดีต่อลูกค้า walk-in"
        elif real_rainfall < 10:
            weather_desc = "ฝนตกเล็กน้อยปรอยๆ"
            weather_impact = "กระทบลูกค้าเดินเท้าเล็กน้อย"
        else:
            weather_desc = "ฝนตกหนัก"
            weather_impact = "ความเสี่ยงลูกค้าลดลง"

        ai_insights = {
            "date_str": target_date_obj.strftime("%d/%m/%Y"),
            "temperature": f"{real_temp} °C",
            "rainfall": f"{real_rainfall} มม. ({weather_desc})",
            "weather_impact": weather_impact,
            "day_type": day_type_desc,
            "payday_status": payday_desc,
            "branch_name": branch_info["name"]
        }
        
        features = np.array([[is_weekend, payload.is_public_holiday, is_payday, real_rainfall, payload.nearby_major_event]])
        
        base_predicted = int(ml_model.predict(features)[0])
        # ปรับสัดส่วนตามฐานลูกค้าของแต่ละสาขา
        scale_factor = branch_info["base_cust"] / 400.0
        predicted_cust = int(base_predicted * scale_factor)
        predicted_cust = max(120, predicted_cust) 
        
        multiplier = predicted_cust / branch_info["base_cust"]

        po_items = []
        total_budget = 0.0

        for item in raw_materials:
            safety_factor = 1.15 if (is_weekend or payload.is_public_holiday or is_payday) else 1.05
            adjusted_rop = item["rop"] * multiplier * safety_factor
            
            order_triggered = item["stock"] <= adjusted_rop
            suggested_order = item["eoq"] * multiplier if order_triggered else 0.0
            suggested_order = round(suggested_order, 2)
            
            req_budget = suggested_order * item["price_per_kg"]
            total_budget += req_budget

            po_items.append({
                "item_id": item["id"],
                "item_name": item["name"],
                "current_stock_kg": item["stock"],
                "reorder_point_rop_kg": round(adjusted_rop, 2),
                "suggested_order_kg": suggested_order,
                "order_triggered": order_triggered,
                "required_budget_thb": round(req_budget, 2)
            })

        return {
            "branch_id": payload.branch_id,
            "target_date": payload.target_date,
            "predicted_customers": predicted_cust,
            "total_estimated_budget_thb": round(total_budget, 2),
            "ai_insights": ai_insights,
            "po_items": po_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))