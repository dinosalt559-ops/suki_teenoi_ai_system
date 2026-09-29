from pydantic import BaseModel
from typing import List

# ==========================================
# 1. โครงสร้างทั่วไปที่ใช้ร่วมกัน
# ==========================================
class POItem(BaseModel):
    item_id: str
    item_name: str
    current_stock_kg: float
    reorder_point_rop_kg: float
    suggested_order_kg: float
    order_triggered: bool
    required_budget_thb: float

# ==========================================
# 2. สำหรับหน้าเว็บ (แบบกรอกข้อมูลเอง - ของเก่า)
# ==========================================
class PredictionRequest(BaseModel):
    branch_id: int
    target_date: str
    is_weekend: int
    is_public_holiday: int
    is_payday_season: int
    rainfall_mm: float
    nearby_major_event: int

class PredictionResponse(BaseModel):
    branch_id: int
    target_date: str
    predicted_customers: int
    total_estimated_budget_thb: float
    po_items: List[POItem]

# ==========================================
# 3. สำหรับดึงสภาพอากาศอัตโนมัติล่วงหน้า 2 วัน (ฟีเจอร์ใหม่)
# ==========================================
class DailyPrediction(BaseModel):
    target_date: str
    predicted_customers: int
    rainfall_mm: float
    is_weekend: int
    total_estimated_budget_thb: float
    po_items: List[POItem]

class MultiDayForecastResponse(BaseModel):
    branch_id: int
    forecasts: List[DailyPrediction]