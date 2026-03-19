from flask import Flask, request, jsonify, render_template, send_file, session
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os
import io
import base64
import uuid
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'hr_dashboard_secret_2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

UPLOAD_FOLDER = 'static/uploads'
EXPORT_FOLDER = 'static/exports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# Global data store (in production use Redis/DB)
data_store = {}

COLUMN_ALIASES = {
    'employee_id': ['employee_id', 'emp_id', 'id', 'employeeid', 'staff_id'],
    'name': ['name', 'full_name', 'employee_name', 'emp_name'],
    'department': ['department', 'dept', 'division', 'team', 'group'],
    'gender': ['gender', 'sex'],
    'age': ['age', 'years_old'],
    'salary': ['salary', 'wage', 'pay', 'compensation', 'annual_salary', 'monthly_salary'],
    'experience': ['experience', 'years_experience', 'exp', 'tenure', 'years_of_experience', 'experience_years'],
    'performance_rating': ['performance_rating', 'performance', 'rating', 'perf_rating', 'score', 'performance_score'],
    'joining_date': ['joining_date', 'hire_date', 'start_date', 'date_joined', 'join_date'],
}

def normalize_columns(df):
    """Map dataset columns to standard names."""
    col_map = {}
    df_cols_lower = {c.lower().replace(' ', '_'): c for c in df.columns}
    for std_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df_cols_lower:
                col_map[df_cols_lower[alias]] = std_name
                break
    df = df.rename(columns=col_map)
    return df

def clean_dataframe(df):
    """Clean and preprocess the dataframe."""
    df = normalize_columns(df)
    # Fill numeric nulls with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    # Fill categorical nulls with mode
    for col in df.select_dtypes(include=['object']).columns:
        mode = df[col].mode()
        if len(mode) > 0:
            df[col].fillna(mode[0], inplace=True)
    # Parse dates
    if 'joining_date' in df.columns:
        df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
    return df

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', 
                facecolor='#0d1b3e', edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def set_dark_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1b3e',
        'axes.facecolor': '#112055',
        'axes.edgecolor': '#1e3a8a',
        'axes.labelcolor': '#93c5fd',
        'xtick.color': '#93c5fd',
        'ytick.color': '#93c5fd',
        'text.color': '#e0f2fe',
        'grid.color': '#1e3a8a',
        'grid.alpha': 0.4,
    })

def generate_charts(df):
    charts = {}
    set_dark_style()
    palette = ['#3b82f6', '#06b6d4', '#8b5cf6', '#f59e0b', '#10b981', '#ef4444', '#ec4899', '#f97316']

    # 1. Employees per Department
    if 'department' in df.columns:
        dept_counts = df['department'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(dept_counts.index, dept_counts.values, color=palette[:len(dept_counts)], 
                      edgecolor='#1e3a8a', linewidth=0.5, width=0.6)
        ax.set_title('Employees per Department', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        ax.set_xlabel('Department', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, dept_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', va='bottom', fontsize=9, color='#e0f2fe')
        plt.tight_layout()
        charts['dept_bar'] = fig_to_base64(fig)

    # 2. Gender Distribution
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            gender_counts.values, labels=gender_counts.index,
            autopct='%1.1f%%', colors=palette[:len(gender_counts)],
            wedgeprops={'linewidth': 2, 'edgecolor': '#0d1b3e'},
            textprops={'color': '#e0f2fe', 'fontsize': 11}
        )
        for at in autotexts:
            at.set_color('#0d1b3e')
            at.set_fontweight('bold')
        ax.set_title('Gender Distribution', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        plt.tight_layout()
        charts['gender_pie'] = fig_to_base64(fig)

    # 3. Avg Salary by Department
    if 'department' in df.columns and 'salary' in df.columns:
        dept_salary = df.groupby('department')['salary'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(dept_salary.index, dept_salary.values, 
                      color=[palette[i % len(palette)] for i in range(len(dept_salary))],
                      edgecolor='#1e3a8a', linewidth=0.5, width=0.6)
        ax.set_title('Average Salary by Department', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        ax.set_xlabel('Department', fontsize=10)
        ax.set_ylabel('Avg Salary', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, dept_salary.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + dept_salary.max()*0.01,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=8, color='#e0f2fe')
        plt.tight_layout()
        charts['salary_dept'] = fig_to_base64(fig)

    # 4. Age Distribution
    if 'age' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(df['age'].dropna(), bins=20, color='#3b82f6', edgecolor='#0d1b3e', 
                linewidth=0.5, alpha=0.85)
        ax.set_title('Age Distribution', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        ax.set_xlabel('Age', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        charts['age_hist'] = fig_to_base64(fig)

    # 5. Experience Distribution
    if 'experience' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(df['experience'].dropna(), bins=15, color='#06b6d4', edgecolor='#0d1b3e',
                linewidth=0.5, alpha=0.85)
        ax.set_title('Experience Distribution (Years)', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        ax.set_xlabel('Years of Experience', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        charts['exp_dist'] = fig_to_base64(fig)

    # 6. Correlation Heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = False
        sns.heatmap(corr, ax=ax, cmap=cmap, annot=True, fmt='.2f',
                    linewidths=0.5, linecolor='#0d1b3e',
                    annot_kws={'size': 9, 'color': 'white'},
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Correlation Heatmap', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        ax.tick_params(colors='#93c5fd')
        plt.tight_layout()
        charts['corr_heatmap'] = fig_to_base64(fig)

    # 7. Salary by Gender
    if 'gender' in df.columns and 'salary' in df.columns:
        gender_salary = df.groupby('gender')['salary'].mean()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(gender_salary.index, gender_salary.values,
                      color=palette[:len(gender_salary)], edgecolor='#1e3a8a', width=0.4)
        ax.set_title('Avg Salary by Gender', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        ax.set_ylabel('Average Salary', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, gender_salary.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + gender_salary.max()*0.01,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=10, color='#e0f2fe')
        plt.tight_layout()
        charts['gender_salary'] = fig_to_base64(fig)

    # 8. Performance Distribution
    if 'performance_rating' in df.columns:
        perf = df['performance_rating'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(perf.index.astype(str), perf.values,
                      color=palette[:len(perf)], edgecolor='#1e3a8a', width=0.5)
        ax.set_title('Performance Rating Distribution', fontsize=13, color='#93c5fd', pad=12, fontweight='bold')
        ax.set_xlabel('Performance Rating', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        charts['perf_dist'] = fig_to_base64(fig)

    return charts

def generate_eda_report(df, session_id):
    """Generate HTML EDA report manually."""
    report_path = os.path.join(EXPORT_FOLDER, f'eda_report_{session_id}.html')
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    stats = df.describe().to_html(classes='table table-dark table-striped table-sm', border=0)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_html = missing_df.to_html(classes='table table-dark table-striped table-sm', border=0)
    
    dtypes_df = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str), 'Unique Values': [df[c].nunique() for c in df.columns]})
    dtypes_html = dtypes_df.to_html(classes='table table-dark table-striped table-sm', border=0, index=False)

    cat_section = ""
    for col in cat_cols[:6]:
        vc = df[col].value_counts().head(10)
        vc_html = vc.to_frame().reset_index().rename(columns={col:'Value', 'count':'Count'}).to_html(classes='table table-dark table-sm', border=0, index=False)
        cat_section += f"<h4 style='color:#93c5fd'>{col}</h4>{vc_html}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HR EDA Report</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body {{ background:#0d1b3e; color:#e0f2fe; font-family:'Segoe UI',sans-serif; }}
.card {{ background:#112055; border:1px solid #1e3a8a; margin-bottom:24px; border-radius:12px; }}
.card-header {{ background:#1e3a8a; border-radius:12px 12px 0 0; padding:12px 20px; font-weight:700; color:#93c5fd; }}
table {{ color:#e0f2fe !important; }}
thead th {{ background:#1e3a8a !important; color:#93c5fd !important; }}
.stat-badge {{ background:#1e3a8a; border-radius:8px; padding:8px 16px; margin:4px; display:inline-block; }}
h1,h2,h3,h4 {{ color:#60a5fa; }}
</style>
</head>
<body>
<div class="container py-5">
<h1 class="text-center mb-2">🧠 HR Dataset EDA Report</h1>
<p class="text-center text-muted mb-5">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="card"><div class="card-header">📊 Dataset Overview</div>
<div class="card-body">
<div class="stat-badge">Rows: <strong>{len(df):,}</strong></div>
<div class="stat-badge">Columns: <strong>{len(df.columns)}</strong></div>
<div class="stat-badge">Numeric Cols: <strong>{len(num_cols)}</strong></div>
<div class="stat-badge">Categorical Cols: <strong>{len(cat_cols)}</strong></div>
<div class="stat-badge">Duplicates: <strong>{df.duplicated().sum()}</strong></div>
<div class="stat-badge">Total Missing: <strong>{df.isnull().sum().sum()}</strong></div>
</div></div>

<div class="card"><div class="card-header">📋 Column Types & Uniqueness</div>
<div class="card-body">{dtypes_html}</div></div>

<div class="card"><div class="card-header">❓ Missing Values</div>
<div class="card-body">{missing_html}</div></div>

<div class="card"><div class="card-header">📈 Statistical Summary</div>
<div class="card-body overflow-auto">{stats}</div></div>

<div class="card"><div class="card-header">🏷️ Categorical Column Distributions</div>
<div class="card-body">{cat_section}</div></div>

</div></body></html>"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return report_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400
    
    session_id = str(uuid.uuid4())[:8]
    
    try:
        df = pd.read_csv(file)
        df = clean_dataframe(df)
        
        # Store in memory
        data_store[session_id] = df
        
        # Generate charts
        charts = generate_charts(df)
        
        # Build analytics
        analytics = build_analytics(df)
        
        # Generate EDA report
        report_path = generate_eda_report(df, session_id)
        
        # Save cleaned CSV
        csv_path = os.path.join(EXPORT_FOLDER, f'cleaned_{session_id}.csv')
        df.to_csv(csv_path, index=False)
        
        return jsonify({
            'session_id': session_id,
            'analytics': analytics,
            'charts': charts,
            'columns': list(df.columns),
            'report_url': f'/static/exports/eda_report_{session_id}.html',
            'csv_url': f'/static/exports/cleaned_{session_id}.csv'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def build_analytics(df):
    analytics = {
        'overview': {
            'total_employees': len(df),
            'total_columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
        }
    }
    
    if 'salary' in df.columns:
        analytics['salary'] = {
            'mean': round(float(df['salary'].mean()), 2),
            'median': round(float(df['salary'].median()), 2),
            'min': round(float(df['salary'].min()), 2),
            'max': round(float(df['salary'].max()), 2),
            'std': round(float(df['salary'].std()), 2),
        }
    
    if 'age' in df.columns:
        analytics['age'] = {
            'mean': round(float(df['age'].mean()), 1),
            'min': int(df['age'].min()),
            'max': int(df['age'].max()),
        }
    
    if 'experience' in df.columns:
        analytics['experience'] = {
            'mean': round(float(df['experience'].mean()), 1),
            'max': round(float(df['experience'].max()), 1),
        }
    
    if 'department' in df.columns:
        dept_counts = df['department'].value_counts()
        analytics['departments'] = {
            'count': len(dept_counts),
            'largest': str(dept_counts.index[0]),
            'largest_count': int(dept_counts.iloc[0]),
            'distribution': dept_counts.to_dict()
        }
        if 'salary' in df.columns:
            analytics['departments']['avg_salary'] = df.groupby('department')['salary'].mean().round(2).to_dict()
    
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        analytics['gender'] = {
            'distribution': gender_counts.to_dict(),
        }
        if 'salary' in df.columns:
            analytics['gender']['avg_salary'] = df.groupby('gender')['salary'].mean().round(2).to_dict()
    
    if 'performance_rating' in df.columns:
        analytics['performance'] = {
            'mean': round(float(df['performance_rating'].mean()), 2),
            'distribution': df['performance_rating'].value_counts().to_dict()
        }
    
    # Data types
    analytics['dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    analytics['missing_per_col'] = df.isnull().sum().to_dict()
    
    return analytics

@app.route('/api/table', methods=['POST'])
def get_table():
    data = request.json
    session_id = data.get('session_id')
    page = int(data.get('page', 1))
    per_page = int(data.get('per_page', 20))
    search = data.get('search', '').lower()
    dept_filter = data.get('department', '')
    gender_filter = data.get('gender', '')
    sort_col = data.get('sort_col', '')
    sort_dir = data.get('sort_dir', 'asc')
    
    if session_id not in data_store:
        return jsonify({'error': 'Session expired. Please re-upload.'}), 404
    
    df = data_store[session_id].copy()
    
    if search:
        mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(search).any(), axis=1)
        df = df[mask]
    
    if dept_filter and 'department' in df.columns:
        df = df[df['department'] == dept_filter]
    
    if gender_filter and 'gender' in df.columns:
        df = df[df['gender'] == gender_filter]
    
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=(sort_dir == 'asc'))
    
    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.iloc[start:end]
    
    # Convert dates to string
    for col in page_df.select_dtypes(include=['datetime64']).columns:
        page_df[col] = page_df[col].dt.strftime('%Y-%m-%d')
    
    return jsonify({
        'data': page_df.fillna('').to_dict(orient='records'),
        'columns': list(df.columns),
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page,
        'departments': list(data_store[session_id]['department'].unique()) if 'department' in data_store[session_id].columns else [],
        'genders': list(data_store[session_id]['gender'].unique()) if 'gender' in data_store[session_id].columns else [],
    })

@app.route('/api/export_summary', methods=['POST'])
def export_summary():
    data = request.json
    session_id = data.get('session_id')
    if session_id not in data_store:
        return jsonify({'error': 'Session expired'}), 404
    df = data_store[session_id]
    analytics = build_analytics(df)
    
    lines = [
        "=" * 60,
        "HR EMPLOYEE ANALYTICS SUMMARY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "DATASET OVERVIEW",
        f"  Total Employees   : {analytics['overview']['total_employees']:,}",
        f"  Total Columns     : {analytics['overview']['total_columns']}",
        f"  Missing Values    : {analytics['overview']['missing_values']}",
        f"  Duplicate Rows    : {analytics['overview']['duplicates']}",
        "",
    ]
    
    if 'salary' in analytics:
        s = analytics['salary']
        lines += [
            "SALARY STATISTICS",
            f"  Mean Salary       : {s['mean']:,.2f}",
            f"  Median Salary     : {s['median']:,.2f}",
            f"  Min Salary        : {s['min']:,.2f}",
            f"  Max Salary        : {s['max']:,.2f}",
            f"  Std Deviation     : {s['std']:,.2f}",
            "",
        ]
    
    if 'departments' in analytics:
        d = analytics['departments']
        lines += [
            "DEPARTMENT ANALYSIS",
            f"  Total Departments : {d['count']}",
            f"  Largest Dept      : {d['largest']} ({d['largest_count']} employees)",
            "",
        ]
        for dept, count in d['distribution'].items():
            lines.append(f"    {dept:<25} : {count} employees")
        lines.append("")
    
    if 'gender' in analytics:
        lines += ["GENDER DISTRIBUTION"]
        for g, count in analytics['gender']['distribution'].items():
            lines.append(f"    {g:<15} : {count}")
        lines.append("")
    
    content = "\n".join(lines)
    report_path = os.path.join(EXPORT_FOLDER, f'summary_{session_id}.txt')
    with open(report_path, 'w') as f:
        f.write(content)
    return send_file(report_path, as_attachment=True, download_name='hr_summary_report.txt')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
