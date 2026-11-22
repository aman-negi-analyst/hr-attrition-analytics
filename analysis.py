"""
HR Attrition & People Analytics
Author: Aman Negi
Description: Employee attrition analysis — identifying key drivers of attrition,
department risk profiling, salary band analysis, and ML-based attrition prediction.
Targets: Deloitte, Accenture, TCS, Infosys, Amazon, Google HR Analytics roles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150
})

BLUE   = '#2563EB'
RED    = '#EF4444'
GREEN  = '#10B981'
ORANGE = '#F59E0B'
PURPLE = '#8B5CF6'
DARK   = '#1F2937'

df = pd.read_csv('hr_data.csv')
df['attrition_flag'] = (df['attrition'] == 'Yes').astype(int)

print("=" * 60)
print("HR ATTRITION ANALYTICS — EXECUTIVE SUMMARY")
print("=" * 60)
print(f"Total Employees    : {len(df):,}")
print(f"Attrition Rate     : {df['attrition_flag'].mean()*100:.1f}%")
print(f"Employees Left     : {df['attrition_flag'].sum():,}")
print(f"Avg Monthly Salary : ₹{df['monthly_salary'].mean():,.0f}")
print(f"Avg Age            : {df['age'].mean():.1f} years")
print(f"Avg Experience     : {df['years_experience'].mean():.1f} years")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# FIGURE 1 — Attrition Dashboard
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('HR Attrition Analytics Dashboard', fontsize=16, fontweight='bold', color=DARK, y=1.01)

# 1. Attrition by Department
dept_attr = df.groupby('department')['attrition_flag'].mean() * 100
dept_attr = dept_attr.sort_values(ascending=True)
colors_dept = [RED if v > 25 else ORANGE if v > 18 else GREEN for v in dept_attr]
axes[0,0].barh(dept_attr.index, dept_attr.values, color=colors_dept)
axes[0,0].axvline(df['attrition_flag'].mean()*100, color=DARK, linestyle='--',
                   label=f"Avg: {df['attrition_flag'].mean()*100:.1f}%")
axes[0,0].set_title('Attrition Rate by Department', fontweight='bold')
axes[0,0].set_xlabel('Attrition Rate (%)')
axes[0,0].legend(fontsize=8)
for i, v in enumerate(dept_attr.values):
    axes[0,0].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=8, fontweight='bold')

# 2. Salary vs Attrition
stayed = df[df['attrition'] == 'No']['monthly_salary']
left   = df[df['attrition'] == 'Yes']['monthly_salary']
axes[0,1].hist(stayed/1000, bins=30, alpha=0.6, color=GREEN, label='Stayed', edgecolor='white')
axes[0,1].hist(left/1000,   bins=30, alpha=0.6, color=RED,   label='Left',   edgecolor='white')
axes[0,1].axvline(stayed.mean()/1000, color=GREEN, linestyle='--', linewidth=2)
axes[0,1].axvline(left.mean()/1000,   color=RED,   linestyle='--', linewidth=2)
axes[0,1].set_title('Salary Distribution: Stayed vs Left', fontweight='bold')
axes[0,1].set_xlabel('Monthly Salary (₹ Thousands)')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()

# 3. Attrition by Job Satisfaction
sat_attr = df.groupby('job_satisfaction')['attrition_flag'].mean() * 100
sat_labels = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
colors_sat = [RED, ORANGE, GREEN, GREEN]
bars = axes[0,2].bar([sat_labels[i] for i in sat_attr.index], sat_attr.values, color=colors_sat)
axes[0,2].set_title('Attrition Rate by Job Satisfaction', fontweight='bold')
axes[0,2].set_ylabel('Attrition Rate (%)')
for bar, val in zip(bars, sat_attr.values):
    axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontweight='bold')

# 4. Overtime impact
ot_attr = df.groupby('overtime')['attrition_flag'].mean() * 100
colors_ot = [GREEN, RED]
bars2 = axes[1,0].bar(ot_attr.index, ot_attr.values, color=colors_ot, width=0.4)
axes[1,0].set_title('Attrition Rate: Overtime vs No Overtime', fontweight='bold')
axes[1,0].set_ylabel('Attrition Rate (%)')
for bar, val in zip(bars2, ot_attr.values):
    axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

# 5. Attrition by Years at Company
df['tenure_band'] = pd.cut(df['years_at_company'],
                            bins=[-1,1,3,5,10,50],
                            labels=['0-1 yr','1-3 yrs','3-5 yrs','5-10 yrs','10+ yrs'])
tenure_attr = df.groupby('tenure_band', observed=True)['attrition_flag'].mean() * 100
colors_ten = [RED, ORANGE, GREEN, GREEN, GREEN]
axes[1,1].bar(tenure_attr.index, tenure_attr.values, color=colors_ten)
axes[1,1].set_title('Attrition Rate by Tenure Band', fontweight='bold')
axes[1,1].set_ylabel('Attrition Rate (%)')
axes[1,1].set_xticklabels(tenure_attr.index, rotation=15)
for i, val in enumerate(tenure_attr.values):
    axes[1,1].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold')

# 6. Work-life balance vs performance heatmap
pivot_wl = df.pivot_table(values='attrition_flag',
                           index='work_life_balance',
                           columns='job_satisfaction',
                           aggfunc='mean') * 100
wl_labels = {1:'Low', 2:'Med', 3:'High', 4:'V.High'}
pivot_wl.index = [wl_labels[i] for i in pivot_wl.index]
pivot_wl.columns = [wl_labels[i] for i in pivot_wl.columns]
sns.heatmap(pivot_wl, annot=True, fmt='.0f', cmap='RdYlGn_r',
            ax=axes[1,2], linewidths=0.5,
            cbar_kws={'label': 'Attrition %'})
axes[1,2].set_title('Attrition % — Work-Life Balance × Job Satisfaction', fontweight='bold')
axes[1,2].set_xlabel('Job Satisfaction')
axes[1,2].set_ylabel('Work-Life Balance')

plt.tight_layout()
plt.savefig('01_attrition_dashboard.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Chart 1 saved: Attrition Dashboard")

# ════════════════════════════════════════════════════════════════
# FIGURE 2 — Deep Dive Analysis
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Attrition Deep Dive — Salary, Age & Risk Profiling', fontsize=14, fontweight='bold', color=DARK)

# Salary band attrition
df['salary_band'] = pd.cut(df['monthly_salary'],
                            bins=[0, 30000, 50000, 70000, 100000, 999999],
                            labels=['<30K','30-50K','50-70K','70-100K','100K+'])
sal_attr = df.groupby('salary_band', observed=True)['attrition_flag'].mean() * 100
sal_count = df.groupby('salary_band', observed=True)['employee_id'].count()
colors_sal = [RED, RED, ORANGE, GREEN, GREEN]
axes[0].bar(sal_attr.index, sal_attr.values, color=colors_sal)
axes[0].set_title('Attrition Rate by Salary Band', fontweight='bold')
axes[0].set_ylabel('Attrition Rate (%)')
for i, (val, cnt) in enumerate(zip(sal_attr.values, sal_count.values)):
    axes[0].text(i, val + 0.5, f'{val:.1f}%\n(n={cnt})', ha='center', fontsize=8, fontweight='bold')

# Age vs Attrition
df['age_band'] = pd.cut(df['age'], bins=[21,25,30,35,40,45,60],
                         labels=['22-25','26-30','31-35','36-40','41-45','45+'])
age_attr = df.groupby('age_band', observed=True)['attrition_flag'].mean() * 100
colors_age = [RED, RED, ORANGE, GREEN, GREEN, GREEN]
axes[1].bar(age_attr.index, age_attr.values, color=colors_age)
axes[1].set_title('Attrition Rate by Age Group', fontweight='bold')
axes[1].set_ylabel('Attrition Rate (%)')
for i, val in enumerate(age_attr.values):
    axes[1].text(i, val + 0.5, f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

# Department risk bubble chart
dept_summary = df.groupby('department').agg(
    attrition_rate=('attrition_flag', 'mean'),
    avg_salary=('monthly_salary', 'mean'),
    headcount=('employee_id', 'count')
).reset_index()
dept_summary['attrition_rate'] *= 100

scatter_colors = [RED if r > 25 else ORANGE if r > 18 else GREEN
                  for r in dept_summary['attrition_rate']]
scatter = axes[2].scatter(dept_summary['avg_salary']/1000,
                           dept_summary['attrition_rate'],
                           s=dept_summary['headcount']*1.5,
                           c=scatter_colors, alpha=0.7, edgecolors='white', linewidth=1.5)
for _, row in dept_summary.iterrows():
    axes[2].annotate(row['department'],
                     (row['avg_salary']/1000, row['attrition_rate']),
                     textcoords='offset points', xytext=(5, 3), fontsize=7)
axes[2].set_title('Dept Risk Map\n(bubble size = headcount)', fontweight='bold')
axes[2].set_xlabel('Avg Monthly Salary (₹ Thousands)')
axes[2].set_ylabel('Attrition Rate (%)')

plt.tight_layout()
plt.savefig('02_deep_dive.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Chart 2 saved: Deep Dive Analysis")

# ════════════════════════════════════════════════════════════════
# FIGURE 3 — ML: Attrition Prediction
# ════════════════════════════════════════════════════════════════
print("\n--- Training Attrition Prediction Model ---")

le = LabelEncoder()
df_ml = df.copy()
for col in ['department', 'overtime', 'attrition', 'tenure_band', 'salary_band', 'age_band']:
    if col in df_ml.columns:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

features = ['age', 'department', 'job_level', 'monthly_salary', 'years_experience',
            'years_at_company', 'job_satisfaction', 'work_life_balance',
            'performance_rating', 'overtime', 'distance_from_home',
            'num_companies_worked', 'training_last_year']

X = df_ml[features]
y = df_ml['attrition_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42,
                                class_weight='balanced', max_depth=8)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print(f"Model AUC: {auc:.3f}")
print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Attrition Prediction — Random Forest Classifier', fontsize=14, fontweight='bold', color=DARK)

importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
feat_labels = {
    'age': 'Age', 'department': 'Department', 'job_level': 'Job Level',
    'monthly_salary': 'Monthly Salary', 'years_experience': 'Total Experience',
    'years_at_company': 'Tenure', 'job_satisfaction': 'Job Satisfaction',
    'work_life_balance': 'Work-Life Balance', 'performance_rating': 'Performance Rating',
    'overtime': 'Overtime', 'distance_from_home': 'Distance from Home',
    'num_companies_worked': 'Companies Worked', 'training_last_year': 'Training Hours'
}
importance.index = [feat_labels.get(f, f) for f in importance.index]
colors_imp = [RED if v > importance.quantile(0.7) else BLUE for v in importance.values]
axes[0].barh(importance.index, importance.values, color=colors_imp)
axes[0].set_title(f'Feature Importance (AUC = {auc:.3f})', fontweight='bold')
axes[0].set_xlabel('Importance Score')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
axes[1].set_title('Confusion Matrix', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('03_ml_attrition.png', bbox_inches='tight', dpi=150)
plt.close()
print("✅ Chart 3 saved: ML Attrition Model")

# ════════════════════════════════════════════════════════════════
# COST OF ATTRITION
# ════════════════════════════════════════════════════════════════
attrition_employees = df[df['attrition'] == 'Yes']
replacement_cost_factor = 1.5  # Industry standard: 1.5x annual salary
annual_cost = attrition_employees['monthly_salary'].sum() * 12 * replacement_cost_factor

print("\n" + "=" * 60)
print("HR INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

worst_dept = dept_attr.index[-1]
print(f"\n1. ATTRITION OVERVIEW")
print(f"   • Overall attrition: {df['attrition_flag'].mean()*100:.1f}%")
print(f"   • {df['attrition_flag'].sum()} employees left out of {len(df)}")
print(f"   • Estimated annual cost of attrition: ₹{annual_cost/1e7:.1f} Crore")
print(f"   • (Based on 1.5x annual salary replacement cost)")

print(f"\n2. HIGH-RISK DEPARTMENTS")
for dept in dept_attr[dept_attr > 25].index:
    count = len(df[(df['department']==dept) & (df['attrition']=='Yes')])
    print(f"   • {dept}: {dept_attr[dept]:.1f}% attrition — {count} employees left")

print(f"\n3. KEY ATTRITION DRIVERS")
print(f"   • Low satisfaction employees: {df[df['job_satisfaction']<=2]['attrition_flag'].mean()*100:.1f}% attrition rate")
print(f"   • Overtime employees: {df[df['overtime']=='Yes']['attrition_flag'].mean()*100:.1f}% attrition rate")
print(f"   • <30K salary band: {sal_attr.iloc[0]:.1f}% attrition — pay revision needed")
print(f"   • First year (0-1yr tenure): {tenure_attr.iloc[0]:.1f}% attrition — onboarding gap")

print(f"\n4. ML MODEL")
print(f"   • Random Forest predicts attrition with AUC = {auc:.3f}")
print(f"   • Top predictors: Salary, Job Satisfaction, Overtime, Tenure")
print(f"   • Model enables HR to flag at-risk employees proactively")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
