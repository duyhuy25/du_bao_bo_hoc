from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Student Dropout Early Warning API")

# Setup templates and static files
project_root = os.path.join(os.path.dirname(__file__), "..")
app.mount("/static", StaticFiles(directory=os.path.join(project_root, "app", "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(project_root, "app", "templates"))

# Load model on startup
model_path = os.path.join(project_root, "models", "dropout_model.pkl")
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
    else:
        print("Warning: Model not found. Please train the model first.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any

class StudentAdmissionInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Age_at_enrollment: int = Field(alias="Age at enrollment", ge=15, le=100)
    Gender: int = Field(ge=0, le=1)
    Marital_status: int = Field(alias="Marital status", ge=1, le=6)
    Nacionality: int = Field(ge=1)
    International: int = Field(ge=0, le=1)
    Displaced: int = Field(ge=0, le=1)
    Educational_special_needs: int = Field(alias="Educational special needs", ge=0, le=1)
    Debtor: int = Field(ge=0, le=1)
    Tuition_fees_up_to_date: int = Field(alias="Tuition fees up to date", ge=0, le=1)
    Scholarship_holder: int = Field(alias="Scholarship holder", ge=0, le=1)
    Course: int
    Application_mode: int = Field(alias="Application mode")
    Application_order: int = Field(alias="Application order", ge=0, le=9)
    Daytime_evening_attendance: int = Field(alias="Daytime/evening attendance\t", ge=0, le=1)
    Previous_qualification: int = Field(alias="Previous qualification")
    Previous_qualification_grade: float = Field(alias="Previous qualification (grade)", ge=0.0, le=200.0)
    Admission_grade: float = Field(alias="Admission grade", ge=0.0, le=200.0)
    Mothers_qualification: int = Field(alias="Mother's qualification")
    Fathers_qualification: int = Field(alias="Father's qualification")
    Mothers_occupation: int = Field(alias="Mother's occupation")
    Fathers_occupation: int = Field(alias="Father's occupation")
    Unemployment_rate: float = Field(alias="Unemployment rate")
    Inflation_rate: float = Field(alias="Inflation rate")
    GDP: float

@app.post("/predict")
@app.post("/api/support-priority")
async def predict_support_priority(student: StudentAdmissionInput):
    if not pipeline:
        return {"error": "Model not loaded on server."}
        
    try:
        # Lấy dữ liệu chuẩn hóa đúng tên đặc trưng ban đầu (hỗ trợ cả Pydantic v1 & v2)
        if hasattr(student, "model_dump"):
            data = student.model_dump(by_alias=True)
        else:
            data = student.dict(by_alias=True)
        df = pd.DataFrame([data])
        
        # Predict probability
        probs = pipeline.predict_proba(df)[0]
        classes = pipeline.classes_
        
        result_probs = {c: round(float(p), 4) for c, p in zip(classes, probs)}
        
        # Logic for "Gợi ý ưu tiên liên hệ hỗ trợ" (Early warning support priority)
        dropout_prob = result_probs.get("Dropout", 0.0)
        
        priority = "Normal"
        suggestion = "Không cần hỗ trợ đặc biệt."
        
        if dropout_prob > 0.6:
            priority = "Critical"
            suggestion = "ƯU TIÊN LIÊN HỆ GẤP: Sinh viên có nguy cơ bỏ học rất cao (>60%). Cần hẹn gặp cố vấn học tập ngay."
        elif dropout_prob > 0.4:
            priority = "High"
            suggestion = "CẦN THEO DÕI: Sinh viên có dấu hiệu rủi ro. Cần gửi email hỏi thăm và nhắc nhở các chính sách hỗ trợ tài chính/học vụ."
            
        return {
            "prediction_probabilities": result_probs,
            "support_priority": priority,
            "suggestion": suggestion
        }
        
    except Exception as e:
        return {"error": str(e)}
