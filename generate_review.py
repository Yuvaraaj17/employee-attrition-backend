def generate_manager_review(row):
    leadership_map = {
        'Yes': "The employee has been entrusted with leadership responsibilities and demonstrates capability in this area",
        'No':  "The employee has not yet been assigned leadership responsibilities, but shows potential for future opportunities"
    }
    
    innovation_map = {
        'Yes': "actively engages in innovation opportunities and contributes creative ideas",
        'No':  "has limited exposure to innovation opportunities but could benefit from more involvement"
    }
    
    reputation_map = {
        'Very Poor': "holds a very poor perception of the company's reputation, which may impact engagement",
        'Poor':      "views the company's reputation as poor, which could affect motivation",
        'Fair':      "has a fair perception of the company's reputation, seeing both strengths and weaknesses",
        'Good':      "perceives the company's reputation as good and remains engaged",
        'Excellent': "perceives the company's reputation as excellent and shows strong alignment with company values"
    }
    
    performance_map = {
        'Low':          "Currently, the employee's performance is below expectations and requires improvement",
        'Below Average':"The employee's performance is slightly below average and improvement is encouraged",
        'Average':      "The employee meets expectations with consistent, average performance",
        'High':         "The employee consistently delivers high performance and exceeds expectations",
        'Very High':    "The employee demonstrates exceptional performance and consistently goes above and beyond expectations"
    }

    return (
        f"{leadership_map[row['LEADERSHIP_OPPORTUNITIES']]}. "
        f"The employee {innovation_map[row['INNOVATION_OPPORTUNITIES']]}. "
        f"The employee {reputation_map[row['COMPANY_REPUTATION']]}. "
        f"{performance_map[row['PERFORMANCE_RATING']]}."
    )


def generate_employee_review_personalized(row):
    salary_map = {
        'not happy with salary': "am dissatisfied with my current compensation",
        'okay with salary':      "am fairly satisfied with my compensation",
        'happy with salary':     "am pleased with my compensation"
    }
    
    recognition_map = {
        'Very Low': "receive little recognition for my efforts",
        'Low':      "receive limited recognition",
        'Medium':   "receive a fair amount of recognition",
        'High':     "am well recognized for my work",
        'Very High':"am extremely well recognized and valued for my work"
    }
    
    remote_map = {
        'Yes':  "I work remotely",
        'No':   "I work on-site",
        'Fair': "I have a hybrid work arrangement"
    }
    
    if row['NUMBER_OF_PROMOTIONS'] == 0:
        promotion_statement = "and have not yet received a promotion"
    elif row['NUMBER_OF_PROMOTIONS'] <= 2:
        promotion_statement = "and have received a few promotions"
    else:
        promotion_statement = "and have been promoted multiple times"
    
    balance_map = {
        'Poor':          "I struggle to maintain work-life balance",
        'Below Average': "I have below average work-life balance",
        'Fair':          "I have a fair work-life balance",
        'Good':          "I maintain a good work-life balance",
        'Excellent':     "I enjoy excellent work-life balance"
    }
    
    satisfaction_map = {
        'Very Low': "I am very dissatisfied with my job",
        'Low':      "I am dissatisfied with my job",
        'Medium':   "I am moderately satisfied with my job",
        'High':     "I am highly satisfied with my job",
        'Very High':"I am extremely satisfied with my job"
    }

    return (
        f"I am a {row['AGE']}-year-old {row['JOB_ROLE']} with {row['YEARS_AT_COMPANY']} years at the company. "
        f"I {salary_map[row['SALARY_HAPPINESS']]}, "
        f"{recognition_map[row['EMPLOYEE_RECOGNITION']]} {promotion_statement}. "
        f"{remote_map[row['REMOTE_WORK']]} and {balance_map[row['WORK_LIFE_BALANCE']]}. "
        f"Overall, {satisfaction_map[row['JOB_SATISFACTION']]}."
    )