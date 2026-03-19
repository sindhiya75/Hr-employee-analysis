# HR Employee Analytics Dashboard

A professional Flask-based HR analytics platform with automated data analysis and visualization.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5050
```

## How to Use

1. **Upload Dataset** — Click "Upload Dataset" in the sidebar, upload any HR CSV file
2. **View Dashboard** — KPI cards and key charts appear automatically
3. **Analytics** — All 8 charts: department, salary, gender, age, experience, performance, correlation
4. **EDA Report** — Automated HTML report with statistical summary, missing values, distributions
5. **Employee Table** — Browse with search, filters, sorting, pagination
6. **Export** — Download cleaned CSV, EDA report, summary TXT, individual charts

## Supported Column Names

| Standard Name | Accepted Variations |
|---|---|
| department | dept, division, team |
| salary | wage, pay, compensation |
| experience | exp, tenure, years_experience |
| performance_rating | rating, performance, score |
| joining_date | hire_date, start_date |

A sample dataset (200 employees) is included at `static/uploads/sample_hr_data.csv`.
