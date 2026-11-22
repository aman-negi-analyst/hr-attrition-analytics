"""
Generate realistic HR employee attrition dataset.
Simulates 1,500 employees across 8 departments with realistic attrition drivers.
"""
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

departments = {
    'Engineering':      {'attrition': 0.18, 'avg_salary': 85000, 'count': 350},
    'Sales':            {'attrition': 0.28, 'avg_salary': 55000, 'count': 280},
    'HR':               {'attrition': 0.20, 'avg_salary': 48000, 'count': 120},
    'Finance':          {'attrition': 0.12, 'avg_salary': 72000, 'count': 160},
    'Marketing':        {'attrition': 0.22, 'avg_salary': 58000, 'count': 140},
    'Operations':       {'attrition': 0.16, 'avg_salary': 45000, 'count': 200},
    'Product':          {'attrition': 0.14, 'avg_salary': 90000, 'count': 130},
    'Customer Support': {'attrition': 0.32, 'avg_salary': 35000, 'count': 120},
}

records = []
emp_id = 1

for dept, props in departments.items():
    for _ in range(props['count']):
        age = int(np.clip(np.random.normal(32, 7), 22, 58))
        experience = max(0, min(int(np.random.normal(age - 22, 2)), age - 18))
        salary = round(props['avg_salary'] * np.random.uniform(0.7, 1.5) / 1000) * 1000

        satisfaction = np.random.choice([1,2,3,4], p=[0.15, 0.25, 0.35, 0.25])
        work_life    = np.random.choice([1,2,3,4], p=[0.10, 0.25, 0.40, 0.25])
        performance  = np.random.choice([1,2,3,4], p=[0.05, 0.15, 0.55, 0.25])
        overtime     = np.random.choice(['Yes','No'], p=[0.30, 0.70])
        distance     = min(int(np.random.exponential(12)), 60)

        prob = props['attrition']
        if satisfaction <= 2: prob += 0.15
        if work_life <= 2:    prob += 0.10
        if overtime == 'Yes': prob += 0.08
        if salary < props['avg_salary'] * 0.85: prob += 0.10
        if experience < 2:    prob += 0.12
        if distance > 30:     prob += 0.05
        prob = min(prob, 0.85)

        records.append({
            'employee_id':        f'EMP{emp_id:04d}',
            'age':                age,
            'department':         dept,
            'job_level':          np.random.choice([1,2,3,4,5], p=[0.20,0.30,0.25,0.15,0.10]),
            'monthly_salary':     int(salary),
            'years_experience':   experience,
            'years_at_company':   max(0, min(experience, int(np.random.uniform(0, experience+1)))),
            'job_satisfaction':   satisfaction,
            'work_life_balance':  work_life,
            'performance_rating': performance,
            'overtime':           overtime,
            'distance_from_home': distance,
            'num_companies_worked': max(1, int(np.random.exponential(2))),
            'training_last_year': np.random.choice([0,1,2,3,4,5,6], p=[0.05,0.15,0.25,0.25,0.15,0.10,0.05]),
            'attrition':          'Yes' if np.random.random() < prob else 'No',
        })
        emp_id += 1

df = pd.DataFrame(records)
df.to_csv('hr_data.csv', index=False)
print(f"Dataset: {len(df)} employees | Attrition: {(df['attrition']=='Yes').mean()*100:.1f}%")
