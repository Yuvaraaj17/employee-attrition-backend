import pandas as pd

QUARTILES_DATA = {
    ("Education", "Entry"):   {"Q1": 3949.00, "Q3": 5055.25},
    ("Education", "Mid"):     {"Q1": 3932.75, "Q3": 5037.25},
    ("Education", "Senior"):  {"Q1": 3961.00, "Q3": 5033.75},
    ("Finance", "Entry"):     {"Q1": 7198.00, "Q3": 9855.00},
    ("Finance", "Mid"):       {"Q1": 7156.00, "Q3": 9807.00},
    ("Finance", "Senior"):    {"Q1": 7226.50, "Q3": 9830.25},
    ("Healthcare", "Entry"):  {"Q1": 7318.25, "Q3": 8700.00},
    ("Healthcare", "Mid"):    {"Q1": 7321.00, "Q3": 8677.00},
    ("Healthcare", "Senior"): {"Q1": 7317.75, "Q3": 8667.00},
    ("Media", "Entry"):       {"Q1": 5647.00, "Q3": 6307.00},
    ("Media", "Mid"):         {"Q1": 5664.00, "Q3": 6328.00},
    ("Media", "Senior"):      {"Q1": 5653.00, "Q3": 6353.00},
    ("Technology", "Entry"):  {"Q1": 8106.00, "Q3": 10129.50},
    ("Technology", "Mid"):    {"Q1": 8066.25, "Q3": 10110.00},
    ("Technology", "Senior"): {"Q1": 8100.00, "Q3": 10145.00},
}

QUARTILES_COLS = ["JOB_ROLE", "JOB_LEVEL", "Q1", "Q3"]

OHE_COLS = [
    "GENDER", "JOB_ROLE", "OVERTIME", "MARITAL_STATUS",
    "REMOTE_WORK", "LEADERSHIP_OPPORTUNITIES", "INNOVATION_OPPORTUNITIES", # "ATTRITION"
]

ORD_COLS = [
    "WORK_LIFE_BALANCE", "JOB_SATISFACTION", "PERFORMANCE_RATING",
    "EDUCATION_LEVEL", "JOB_LEVEL", "COMPANY_SIZE", "COMPANY_REPUTATION", "EMPLOYEE_RECOGNITION"
]

NUMERIC_COLS = [
    "AGE", "YEARS_AT_COMPANY", "MONTHLY_INCOME", "WORK_LIFE_BALANCE",
    "JOB_SATISFACTION", "PERFORMANCE_RATING", "NUMBER_OF_PROMOTIONS",
    "DISTANCE_FROM_HOME", "EDUCATION_LEVEL", "NUMBER_OF_DEPENDENTS",
    "JOB_LEVEL", "COMPANY_SIZE", "COMPANY_TENURE", "COMPANY_REPUTATION",
    "EMPLOYEE_RECOGNITION", "EMPLOYEE_POS_PROB", "MANAGER_POS_PROB"
]

INT_COLUMNS = [
    "EMPLOYEE_ID", "AGE", "YEARS_AT_COMPANY", "MONTHLY_INCOME",
    "NUMBER_OF_PROMOTIONS", "NUMBER_OF_DEPENDENTS", "COMPANY_TENURE"
]

OTHER_COLS = ["DISTANCE_FROM_HOME"]

CATEGORICAL_CONSTRAINTS = {
    "WORK_LIFE_BALANCE": ["Below Average","Fair","Good","Excellent"], # Below average is invalid during transform
    "JOB_SATISFACTION": ["Low","Medium","High","Very High"],
    "PERFORMANCE_RATING": ["Low","Below Average","Average","High"], # Very High is invalid during transform
    "OVERTIME": ["Yes","No"],
    "EDUCATION_LEVEL": ["High School","Associate Degree","Master’s Degree","PhD"], # Bachelor's Degree is invalid during transform
    "MARITAL_STATUS": ["Single","Married","Divorced"],
    "JOB_LEVEL": ["Entry","Mid","Senior"],
    "COMPANY_SIZE": ["Small","Medium","Large"],
    "REMOTE_WORK": ["Yes","No"],
    "LEADERSHIP_OPPORTUNITIES": ["Yes","No"],
    "INNOVATION_OPPORTUNITIES": ["Yes","No"],
    "COMPANY_REPUTATION": ["Poor","Fair","Good","Excellent"], # Very Poor is invalid during transform
    "EMPLOYEE_RECOGNITION": ["Low","Medium","High","Very High"], # Very Low is invalid during transform
}

EXCLUDE_COLS = ['EMPLOYEE_POS_PROB', 'MANAGER_POS_PROB']

DROP_COLS = ['EMPLOYEE_ID', 'Q1', 'Q3', 'SALARY_HAPPINESS', 'MANAGER_REVIEW', 'EMPLOYEE_REVIEW_PERSONALIZED']

REQUIRED_ARGS = [
    "EMPLOYEE_ID", "AGE", "GENDER", "YEARS_AT_COMPANY", "JOB_ROLE", "MONTHLY_INCOME",
    "WORK_LIFE_BALANCE", "JOB_SATISFACTION", "PERFORMANCE_RATING",
    "NUMBER_OF_PROMOTIONS", "OVERTIME", "DISTANCE_FROM_HOME",
    "EDUCATION_LEVEL", "MARITAL_STATUS", "NUMBER_OF_DEPENDENTS",
    "JOB_LEVEL", "COMPANY_SIZE", "COMPANY_TENURE", "REMOTE_WORK",
    "LEADERSHIP_OPPORTUNITIES", "INNOVATION_OPPORTUNITIES",
    "COMPANY_REPUTATION", "EMPLOYEE_RECOGNITION"
]

def validate_input(df):
    for col in INT_COLUMNS + OTHER_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        try:
            df[col] = df[col].astype(int)
        except (ValueError, TypeError):
            invalid_values = df[col][~df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit())].unique()
            raise ValueError(f"Column {col} must contain only integers or numeric values. Invalid values: {invalid_values}")

    for col, valid_values in CATEGORICAL_CONSTRAINTS.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        
        lower_map = {v.lower(): v for v in valid_values}

        def map_value(val):
            if val is None:
                return None
            val_str = str(val).strip()
            if val_str.lower() in lower_map:
                return lower_map[val_str.lower()]
            return val_str 

        df[col] = df[col].apply(map_value)

        invalid = df.loc[~df[col].isin(valid_values), col].unique()
        if len(invalid) > 0:
            raise ValueError(f"Invalid values {invalid} in column {col}")

    return True


def salary_happiness(row):
    if row['MONTHLY_INCOME'] <= row['Q1']:
        return "not happy with salary"
    elif row['MONTHLY_INCOME'] <= row['Q3']:
        return "okay with salary"
    else:
        return "happy with salary"


