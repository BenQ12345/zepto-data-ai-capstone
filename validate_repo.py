from pathlib import Path
import ast
import pandas as pd

ROOT = Path(__file__).resolve().parent
required = [
    "README.md",
    "requirements.txt",
    "data_pipeline/pipeline.py",
    "data_pipeline/README.md",
    "analytics/01_eda.py",
    "analytics/02_modeling.py",
    "analytics/titanic.csv",
    "analytics/titanic_best_pipeline.joblib",
    "analytics/README.md",
    "support_assistant/main.py",
    "support_assistant/graph.py",
    "support_assistant/ingest.py",
    "support_assistant/prompt_template.py",
    "support_assistant/Dockerfile",
]
for rel in required:
    assert (ROOT/rel).exists(), f"missing {rel}"

# Syntax validation for all Python source files.
for p in ROOT.rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'), filename=str(p))

# Analytics offline artifact checks.
titanic = pd.read_csv(ROOT/'analytics/titanic.csv')
assert titanic.shape == (891, 15)
assert set(['survived','pclass','age','sibsp','parch','fare','sex','embarked']) <= set(titanic.columns)

print('Repository structural validation: PASS')
