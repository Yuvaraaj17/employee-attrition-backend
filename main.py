import pandas as pd
import argparse
import torch
import joblib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from helpers import *
from generate_review import *
from model import AttritionNN


MODEL_PATH = "Model1/model"
SENTIMENT_MODEL_DIR = "Model1/sentiment_model"
OHE_PATH = "Model1/ohe.pkl"
ORD_ENC_PATH = "Model1/ord_enc.pkl"
SCALER_PATH = "Model1/scaler.pkl"
PT_PATH = "Model1/power_transformer.pkl"

def load_model(model_dir):
    TOKENIZER = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
    MODEL = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
    return TOKENIZER, MODEL

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

def main():
    parser = argparse.ArgumentParser(description="Generate employee and manager reviews")
    parser.add_argument("--csv_file", type=str, default=None)

    for arg in REQUIRED_ARGS:
        parser.add_argument(f"--{arg}", required=True)

    parser.add_argument("--EMPLOYEE_REVIEW_PERSONALIZED", type=str, default=None)
    parser.add_argument("--MANAGER_REVIEW", type=str, default=None)

    args = parser.parse_args()

    if args.csv_file:
        df = pd.read_csv(args.csv_file)
    else:
        arg_dict = {k: v for k, v in vars(args).items() if k != "csv_file"}
        missing_args = [arg for arg in REQUIRED_ARGS if arg_dict.get(arg) is None]
        if missing_args:
            raise ValueError(f"Missing required arguments: {missing_args}")
        df = pd.DataFrame([arg_dict])

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

    TOKENIZER, MODEL = load_model(SENTIMENT_MODEL_DIR)
    
    employee_results = [
        get_positive_prob(x, MODEL, TOKENIZER) for x in tqdm(df["EMPLOYEE_REVIEW_PERSONALIZED"], desc="Employee Reviews")
    ]
    manager_results = [
        get_positive_prob(x, MODEL, TOKENIZER) for x in tqdm(df["MANAGER_REVIEW"], desc="Manager Reviews")
    ]

    df["EMPLOYEE_POS_PROB"] = employee_results
    df["MANAGER_POS_PROB"] = manager_results

    OHE = joblib.load(OHE_PATH)
    OHE_RESULT = OHE.transform(df[OHE_COLS])
    OHE_FEATURE_NAMES = OHE.get_feature_names_out(OHE_COLS)
    df_ohe = pd.DataFrame(OHE_RESULT, columns=OHE_FEATURE_NAMES, index=df.index)
    df = df.drop(columns=OHE_COLS).join(df_ohe)

    ORD_ENC = joblib.load(ORD_ENC_PATH)
    df[ORD_COLS] = ORD_ENC.transform(df[ORD_COLS])

    df = df.drop(columns=DROP_COLS, errors="ignore")

    for col in df.columns:
        if col not in EXCLUDE_COLS:
            try:
                df[col] = df[col].astype(int)
            except Exception as e:
                print(f"Error converting column '{col}' to int: {e}")
                print(df[col]) 

    SCALER = joblib.load(SCALER_PATH)
    PT = joblib.load(PT_PATH)

    df[NUMERIC_COLS] = PT.transform(df[NUMERIC_COLS])
    df[NUMERIC_COLS] = SCALER.transform(df[NUMERIC_COLS])

    input_tensor = torch.tensor(df.values, dtype=torch.float32)
    loaded_model = AttritionNN(input_dim=input_tensor.shape[1])
    loaded_model.load_state_dict(torch.load(f"{MODEL_PATH}/attrition_model.pth"))
    loaded_model.eval()

    with torch.no_grad():
        pred_probs = loaded_model(input_tensor)
        pred_labels = (pred_probs >= 0.5).int()

    df["PRED_PROB"] = pred_probs.numpy()
    df["PRED_LABEL"] = pred_labels.numpy()

    return df[["PRED_PROB", "PRED_LABEL"]].to_dict(orient="records")

if __name__ == "__main__":
    result = main()
    print(result)

