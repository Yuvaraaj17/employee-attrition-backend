from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import joblib
from tqdm import tqdm

from helpers import *
from generate_review import *
from model import AttritionNN

MODEL_PATH = "Model1/model"
SENTIMENT_MODEL_DIR = "Model1/sentiment_model"
OHE_PATH = "Model1/ohe.pkl"
ORD_ENC_PATH = "Model1/ord_enc.pkl"
SCALER_PATH = "Model1/scaler.pkl"
PT_PATH = "Model1/power_transformer.pkl"

app = FastAPI()

class InferParameters(BaseModel):
    EMPLOYEE_ID: int
    AGE: int
    GENDER: str
    YEARS_AT_COMPANY: int
    JOB_ROLE: str
    MONTHLY_INCOME: int
    WORK_LIFE_BALANCE: str
    JOB_SATISFACTION: str
    PERFORMANCE_RATING: str
    NUMBER_OF_PROMOTIONS: int
    OVERTIME: str
    DISTANCE_FROM_HOME: int
    EDUCATION_LEVEL: str
    MARITAL_STATUS: str
    NUMBER_OF_DEPENDENTS: int
    JOB_LEVEL: str
    COMPANY_SIZE: str
    COMPANY_TENURE: int
    REMOTE_WORK: str
    LEADERSHIP_OPPORTUNITIES: str
    INNOVATION_OPPORTUNITIES: str
    COMPANY_REPUTATION: str
    EMPLOYEE_RECOGNITION: str
    MANAGER_REVIEW: Optional[str] = None
    EMPLOYEE_REVIEW_PERSONALIZED: Optional[str] = None

def load_model(model_dir):
    TOKENIZER = AutoTokenizer.from_pretrained(model_dir)
    MODEL = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return TOKENIZER, MODEL

TOKENIZER, MODEL = load_model(SENTIMENT_MODEL_DIR)
OHE = joblib.load(OHE_PATH)
ORD_ENC = joblib.load(ORD_ENC_PATH)
SCALER = joblib.load(SCALER_PATH)
PT = joblib.load(PT_PATH)
ATTRITION_MODEL = torch.load(f"{MODEL_PATH}/attrition_model.pth")


def get_positive_prob(text, model, tokenizer, max_len=514-2, stride=50):
    if not isinstance(text, str) or text.strip() == "":
        return None
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(input_ids):
        end = min(start + max_len, len(input_ids))
        chunk_ids = [tokenizer.bos_token_id] + input_ids[start:end] + [tokenizer.eos_token_id]
        chunks.append(chunk_ids)
        if end == len(input_ids):
            break
        start += max_len - stride
    all_logits = []
    for chunk in chunks:
        inputs = torch.tensor([chunk]).to(model.device)  # send to GPU if available
        with torch.no_grad():
            outputs = model(inputs)
            all_logits.append(outputs.logits)

    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1)
    avg_probs = probs.mean(dim=0)

    return avg_probs[1].item()  


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/get-attrition-score")
async def get_attrition_score(body: InferParameters):
    df = pd.DataFrame([body.model_dump()])
    validate_input(df)

    QUARTILES_DF = pd.DataFrame.from_dict(QUARTILES_DATA, orient='index').reset_index()
    QUARTILES_DF.columns = QUARTILES_COLS
    df = df.merge(QUARTILES_DF, on=["JOB_ROLE", "JOB_LEVEL"], how="left")
    df["SALARY_HAPPINESS"] = df.apply(salary_happiness, axis=1)

    if "MANAGER_REVIEW" in df.columns:
        df["MANAGER_REVIEW"] = df.apply(
            lambda row: row["MANAGER_REVIEW"]
            if pd.notna(row.get("MANAGER_REVIEW")) and str(row["MANAGER_REVIEW"]).strip() != ""
            else generate_manager_review(row),
            axis=1
        )
    else:
        df["MANAGER_REVIEW"] = df.apply(generate_manager_review, axis=1)

    if "EMPLOYEE_REVIEW_PERSONALIZED" in df.columns:
       df["EMPLOYEE_REVIEW_PERSONALIZED"] = df.apply(
            lambda row: row["EMPLOYEE_REVIEW_PERSONALIZED"]
            if pd.notna(row.get("EMPLOYEE_REVIEW_PERSONALIZED")) and str(row["EMPLOYEE_REVIEW_PERSONALIZED"]).strip() != ""
            else generate_employee_review_personalized(row),
            axis=1
        )
    else:
        df["EMPLOYEE_REVIEW_PERSONALIZED"] = df.apply(generate_employee_review_personalized, axis=1)

    employee_results = [
        get_positive_prob(x, MODEL, TOKENIZER) for x in tqdm(df["EMPLOYEE_REVIEW_PERSONALIZED"], desc="Employee Reviews")
    ]
    manager_results = [
        get_positive_prob(x, MODEL, TOKENIZER) for x in tqdm(df["MANAGER_REVIEW"], desc="Manager Reviews")
    ]

    df["EMPLOYEE_POS_PROB"] = employee_results
    df["MANAGER_POS_PROB"] = manager_results

    OHE_RESULT = OHE.transform(df[OHE_COLS])
    OHE_FEATURE_NAMES = OHE.get_feature_names_out(OHE_COLS)
    df_ohe = pd.DataFrame(OHE_RESULT, columns=OHE_FEATURE_NAMES, index=df.index)
    df = df.drop(columns=OHE_COLS).join(df_ohe)


    df[ORD_COLS] = ORD_ENC.transform(df[ORD_COLS])

    df = df.drop(columns=DROP_COLS, errors="ignore")

    for col in df.columns:
        if col not in EXCLUDE_COLS:
            try:
                df[col] = df[col].astype(int)
            except Exception as e:
                print(f"Error converting column '{col}' to int: {e}")
                print(df[col]) 
   
    df[NUMERIC_COLS] = PT.transform(df[NUMERIC_COLS])
    df[NUMERIC_COLS] = SCALER.transform(df[NUMERIC_COLS])

    input_tensor = torch.tensor(df.values, dtype=torch.float32)
    loaded_model = AttritionNN(input_dim=input_tensor.shape[1])
    loaded_model.load_state_dict(ATTRITION_MODEL)
    loaded_model.eval()

    with torch.no_grad():
        pred_probs = loaded_model(input_tensor)
        pred_labels = (pred_probs >= 0.5).int()

    df["PRED_PROB"] = pred_probs.numpy()
    df["PRED_LABEL"] = pred_labels.numpy()

    return df[["PRED_PROB", "PRED_LABEL"]].to_dict(orient="records")