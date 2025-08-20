
"""Runner script for TelecomX churn pipeline (preprocessing + modeling).
Place TelecomX_cleaned.csv in data/ and run:
    python run_telecomx_pipeline.py
"""
import os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
try:
    from imblearn.over_sampling import SMOTE
except:
    SMOTE = None

DATA_CSV = 'data/TelecomX_cleaned.csv'
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError('Place your cleaned CSV as data/TelecomX_cleaned.csv')

df = pd.read_csv(DATA_CSV)
for id_col in ['customerID','CustomerID','id','ID']:
    if id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)

target_candidates = ['Churn','churn','Evasion','Evasión','Cancelacion','Cancelación','target']
target_col = next((c for c in df.columns if c in target_candidates), None)
if target_col is None:
    for c in df.columns:
        if df[c].nunique()==2:
            target_col = c; break
if target_col is None:
    raise ValueError('Cannot detect target column')

df[target_col] = df[target_col].astype(str).str.strip().str.lower().map({'no':0,'n':0,'false':0,'0':0,'si':1,'sí':1,'yes':1,'y':1,'true':1,'1':1}).astype(int)
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
cat_cols = [c for c in cat_cols if c!=target_col]
ohe_cols = [c for c in cat_cols if df[c].nunique()<30]
df_ohe = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
X = df_ohe.drop(columns=[target_col])
y = df_ohe[target_col]
if SMOTE is not None:
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
else:
    X_res, y_res = X, y
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42, stratify=y_res)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
log = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
y_pred_log = log.predict(X_test_scaled)
y_prob_log = log.predict_proba(X_test_scaled)[:,1]
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]
print("Logistic classification report:")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))
print("\nRandom Forest classification report:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
pd.DataFrame(X_res, columns=X_res.columns).assign(**{target_col: y_res.values}).to_csv('data/TelecomX_preprocessed.csv', index=False)
print('Saved data/TelecomX_preprocessed.csv')
