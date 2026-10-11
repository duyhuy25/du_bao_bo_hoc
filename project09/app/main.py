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

@app.post("/predict")
@app.post("/api/support-priority")
async def predict_support_priority(data: dict):
    if not pipeline:
        return {"error": "Model not loaded on server."}
        
    try:
        # Convert dictionary to DataFrame for the pipeline
        # Needs all features except Target and 1st/2nd sem features. 
        # (DropFutureFeatures in pipeline will handle dropping future features if they are accidentally included)
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
