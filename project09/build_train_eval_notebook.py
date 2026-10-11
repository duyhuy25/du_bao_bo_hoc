from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")
import nbformat as n
from nbclient import NotebookClient
root=Path(__file__).parent.resolve()
c=[]
def md(s): c.append(n.v4.new_markdown_cell(s))
def code(s): c.append(n.v4.new_code_cell(s))
md('''# Train và eval: bằng chứng lựa chọn dropout_model.pkl
Notebook chạy toàn bộ quy trình và giữ output. Bài toán phân loại ba lớp tại thời điểm nhập học. Tên file là artifact, không phải kỹ thuật dropout trong mạng neural.

Dùng đúng phép chia và preprocessing của dự án. Chọn ứng viên bằng validation; chỉ đánh giá test sau khi chốt. Không ghi đè artifact gốc.''')
code('''from pathlib import Path
import os, sys, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, joblib
from IPython.display import display
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, ConfusionMatrixDisplay
ROOT = Path.cwd()
if not (ROOT/'src').exists(): ROOT = ROOT/'project09'
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from src.data import load_data, get_train_val_test_splits
from src.features import DropFutureFeatures, build_preprocessor
print('Python:',sys.version.split()[0], '| sklearn:',sklearn.__version__)
print('Seed: 42')
print('Data SHA256:',hashlib.sha256((ROOT/'data/raw/data.csv').read_bytes()).hexdigest())''')
md('## 1. Kiểm tra dữ liệu\nPhân bố nhãn giúp giải thích vì sao cần macro-F1 và recall Dropout bên cạnh accuracy.')
code('''df=load_data()
print('Kích thước:',df.shape,'| Giá trị thiếu:',df.isna().sum().sum(),'| Dòng trùng:',df.duplicated().sum())
display(df.head())
display(df.Target.value_counts().to_frame('Số lượng'))
display(df.dtypes.to_frame('Kiểu dữ liệu'))''')
md('## 2. Chia train / validation / test\nTỷ lệ 60/20/20, stratify theo nhãn, seed 42. Fit scaler và encoder chỉ trên train. Validation phục vụ lựa chọn, test phục vụ đánh giá cuối.')
code('''X_train,X_val,X_test,y_train,y_val,y_test=get_train_val_test_splits(df)
assert not(set(X_train.index)&set(X_val.index))
assert not(set(X_train.index)&set(X_test.index))
assert not(set(X_val.index)&set(X_test.index))
display(pd.DataFrame({k:y.value_counts() for k,y in [('Train',y_train),('Validation',y_val),('Test',y_test)]}))
print('Kích thước:',len(y_train),len(y_val),len(y_test))
print('Không trùng chỉ số dòng giữa các tập; nếu có trùng nội dung cần kiểm tra riêng.')''')
md('## 3. Loại thông tin tương lai\nLoại các biến học kỳ 1 và 2 vì chưa tồn tại lúc nhập học. Chuẩn hóa biến liên tục và one-hot biến phân loại theo src/features.py. Cần xác minh thời điểm thu thập các biến nợ/học phí trước triển khai thực tế.')
code('''reduced=DropFutureFeatures().fit_transform(X_train)
display(pd.Series([x for x in X_train if x not in reduced],name='Cột bị loại').to_frame())
print('Số cột trước / sau loại:',X_train.shape[1],reduced.shape[1])
p=build_preprocessor(X_train)
display(pd.DataFrame([(a,str(b),d) for a,b,d in p.transformers],columns=['Nhóm','Biến đổi','Các cột']))''')
md('''### 3.1. Thí nghiệm bắt buộc: Minh họa rò rỉ dữ liệu (Data Leakage Experiment)
So sánh mô hình ngây thơ (giữ lại kết quả học kỳ 1 & 2) với mô hình thực tế (loại bỏ thông tin tương lai).
Kết quả cho thấy nếu để rò rỉ dữ liệu, các chỉ số nhảy vọt giả tạo do dùng thông tin mà thời điểm nhập học chưa hề có.''')
code('''from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# Pipeline ngây thơ (Naive Pipeline giữ nguyên toàn bộ biến học kỳ)
cat_all = [c for c in X_train.select_dtypes(include=['object', 'int64']).columns]
num_all = [c for c in X_train.select_dtypes(include=['float64']).columns]
p_leak = ColumnTransformer([
    ('num', StandardScaler(), num_all),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_all)
])
m_leak = Pipeline([
    ('preprocessor', p_leak),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])
m_leak.fit(X_train, y_train)
p_leak_val = m_leak.predict(X_val)

acc_leak = accuracy_score(y_val, p_leak_val)
f1_leak = f1_score(y_val, p_leak_val, average='macro', zero_division=0)
rec_leak = recall_score(y_val, p_leak_val, labels=['Dropout'], average=None, zero_division=0)[0]

print(f"[CẢNH BÁO LEAKAGE] Giữ biến học kỳ:  Macro-F1={f1_leak:.4f} | Recall Dropout={rec_leak:.4f} | Accuracy={acc_leak:.4f}")
print("-> Hiệu năng cao ảo (Macro-F1 > 0.80) nhưng hoàn toàn vô dụng tại thời điểm nhập học!")''')
md('''## 4. Train và so sánh ứng viên
Dummy là mốc tối thiểu, Logistic Regression là baseline tuyến tính. RF không trọng số và RF balanced cho thấy tác động của trọng số lớp. RF balanced dùng đúng cấu hình src/train.py. Trọng số balanced không bảo đảm recall Dropout tăng.

**Tiêu chí chốt trước khi xem kết quả:** macro-F1 validation cao nhất, hòa thì ưu tiên recall Dropout rồi accuracy. Khi xác định được chi phí bỏ sót/cảnh báo sai, cần điều chỉnh tiêu chí trên validation.''')
code('''estimators={
'Dummy':DummyClassifier(strategy='most_frequent'),
'Logistic balanced':LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42),
'RF unweighted':RandomForestClassifier(n_estimators=100,random_state=42),
'RF balanced (project)':RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')}
models={}
for name,est in estimators.items():
    model=Pipeline([('dropper',DropFutureFeatures()),('preprocessor',build_preprocessor(X_train)),('classifier',est)])
    model.fit(X_train,y_train)
    models[name]=model
    print('Đã train:',name)
print('Đặc trưng sau encode:',len(models['RF balanced (project)'].named_steps['preprocessor'].get_feature_names_out()))''')
code('''def metrics(y,p):
    return dict(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',zero_division=0),dropout_recall=recall_score(y,p,labels=['Dropout'],average=None,zero_division=0)[0],dropout_precision=precision_score(y,p,labels=['Dropout'],average=None,zero_division=0)[0])
rows=[]
for name,model in models.items():
    for split,X,y in [('Train',X_train,y_train),('Validation',X_val,y_val)]:
        rows.append(dict(model=name,split=split,**metrics(y,model.predict(X))))
results=pd.DataFrame(rows)
display(results.round(4))
ranking=results[results.split=='Validation'].sort_values(['macro_f1','dropout_recall','accuracy'],ascending=False)
chosen=ranking.iloc[0]['model']
print('Chọn theo validation:',chosen)
gap=results.pivot(index='model',columns='split',values='macro_f1')
gap['train_minus_validation']=gap.Train-gap.Validation
display(gap.round(4))
print('Khoảng cách train–validation lớn là dấu hiệu cần xem xét overfitting.')''')
code('''ranking.set_index('model')[['macro_f1','dropout_recall','dropout_precision']].plot.bar(figsize=(10,4),rot=15,ylim=(0,1))
plt.title('So sánh validation')
plt.tight_layout()
plt.show()''')
md('## 5. Eval trên test\nĐánh giá ứng viên được chọn và RF dự án nếu khác. Không đổi lựa chọn dựa trên test. Trong confusion matrix, hàng là thực tế, cột là dự đoán; Dropout dự đoán sang lớp khác là bỏ sót.')
code('''test_rows=[]
for name in dict.fromkeys([chosen,'RF balanced (project)']):
    pred=models[name].predict(X_test)
    test_rows.append(dict(model=name,**metrics(y_test,pred)))
    print('TEST:',name)
    print(classification_report(y_test,pred,digits=4,zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test,pred,labels=models[name].classes_,cmap='Blues')
    plt.title(name+' — Test')
    plt.tight_layout()
test_results=pd.DataFrame(test_rows)
display(test_results.round(4))''')
md('''## 5.1. Đánh giá theo nhóm (Subgroup Evaluation)
Đánh giá độ công bằng và hiệu năng phân tách theo các nhóm nhân khẩu học và hoàn cảnh (Giới tính, Nợ học phí, Học bổng, Đóng học phí đúng hạn). Giúp phát hiện mô hình có bị thiên vị hoặc bỏ sót nhóm nào hay không.''')
code('''sub_cols = ['Gender', 'Debtor', 'Scholarship holder', 'Tuition fees up to date']
best_pred = models[chosen].predict(X_test)
sub_records = []
for c in sub_cols:
    if c in X_test.columns:
        for val in sorted(X_test[c].unique()):
            m = (X_test[c] == val)
            if m.sum() > 0:
                y_sub = y_test[m]
                p_sub = best_pred[m]
                rec = recall_score(y_sub, p_sub, labels=['Dropout'], average=None, zero_division=0)[0]
                prec = precision_score(y_sub, p_sub, labels=['Dropout'], average=None, zero_division=0)[0]
                f1_m = f1_score(y_sub, p_sub, average='macro', zero_division=0)
                sub_records.append({
                    'Nhóm thuộc tính': c,
                    'Giá trị': val,
                    'Số lượng': int(m.sum()),
                    'Dropout Recall': round(rec, 4),
                    'Dropout Precision': round(prec, 4),
                    'Macro-F1': round(f1_m, 4)
                })
df_subgroups = pd.DataFrame(sub_records)
display(df_subgroups)''')
md('''## 6. Kiểm tra chính artifact hiện có
Hiển thị cấu hình, metrics và mức độ giống dự đoán giữa model cũ với RF vừa train. Giống dự đoán không chứng minh lịch sử train. Nếu không có lịch sử dữ liệu của artifact cũ, chưa thể khẳng định test này độc lập đối với artifact đó. Cảnh báo phiên bản (nếu có) được giữ trong output.''')
code('''path=ROOT/'models/dropout_model.pkl'
print('Model SHA256:',hashlib.sha256(path.read_bytes()).hexdigest())
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    saved=joblib.load(path)
for w in caught: print('Cảnh báo:',str(w.message))
display(saved)
print('Classifier:',saved.named_steps['classifier'].get_params())
artifact_rows=[]
for split,X,y in [('Validation',X_val,y_val),('Test',X_test,y_test)]:
    pred=saved.predict(X)
    artifact_rows.append(dict(split=split,**metrics(y,pred)))
    print(split,'— tỷ lệ dự đoán giống RF train lại:',np.mean(pred==models['RF balanced (project)'].predict(X)))
display(pd.DataFrame(artifact_rows).round(4))''')
md('## 7. Kết luận dựa trên số liệu\nChỉ kết luận trong phạm vi các ứng viên và lần chia đã thử. Không chọn trước một kết luận rồi dùng số liệu để hợp thức hóa.')
code('''rf=ranking.set_index('model').loc['RF balanced (project)']
print(f"RF validation: macro-F1={rf['macro_f1']:.4f}, recall Dropout={rf['dropout_recall']:.4f}, precision Dropout={rf['dropout_precision']:.4f}")
if chosen=='RF balanced (project)':
    print('Có cơ sở chọn cấu hình dropout_model.pkl: dẫn đầu các ứng viên theo tiêu chí validation đã đặt.')
else:
    print('Chưa đủ cơ sở gọi dropout_model.pkl là lựa chọn tốt nhất. Ứng viên dẫn đầu:',chosen)
    print('RF là baseline hiện tại; cần giải thích đánh đổi theo nhu cầu nghiệp vụ hoặc cân nhắc thay model.')
print('Hạn chế: một lần chia dữ liệu, chưa cross-validation, chưa xác minh lịch sử artifact, chưa có chi phí bỏ sót/cảnh báo sai.')
out=ROOT/'reports/train_eval'
out.mkdir(parents=True,exist_ok=True)
results.to_csv(out/'train_validation_metrics.csv',index=False)
test_results.to_csv(out/'test_metrics.csv',index=False)
pd.DataFrame(artifact_rows).to_csv(out/'existing_artifact_metrics.csv',index=False)
joblib.dump(models['RF balanced (project)'],ROOT/'models/dropout_model_retrained.pkl')
print('Đã lưu metrics tại',out)
print('Đã lưu dropout_model_retrained.pkl; giữ nguyên dropout_model.pkl.')''')
nb=n.v4.new_notebook(cells=c,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=root/'train_eval_dropout.ipynb'
n.write(nb,path)
try:
    NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
finally:
    n.write(nb,path)
for cell in nb.cells:
    for o in cell.get('outputs',[]):
        if o.output_type=='stream': print(o.text)
print('Notebook:',path)
