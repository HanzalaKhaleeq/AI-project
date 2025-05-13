import os
import time
import pickle
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jinja2 import Environment, FileSystemLoader
import webbrowser

# 1. Create output directories
os.makedirs('plots', exist_ok=True)

# 2. Load test data
def load_test():
    df = pd.read_csv('data/isolet5.data', header=None)
    return df.iloc[:, :-1].values, df.iloc[:, -1].astype(int).values
X_test, y_test = load_test()
num_classes = 26

# 3. Load and evaluate models
model_paths = {
    'KNN': 'models/knn.pkl',
    'Naive Bayes': 'models/nb.pkl',
    'Random Forest': 'models/rf.pkl',
    'K-Means': 'models/kmeans.pkl'
}
results = []
inference_times = []
for name, path in model_paths.items():
    # load time
    start_load = time.time()
    with open(path, 'rb') as f:
        model = pickle.load(f)
    load_time = time.time() - start_load
    # inference time
    start_pred = time.time()
    preds = model.predict(X_test)
    pred_time = time.time() - start_pred
    # metrics
    if name != 'K-Means':
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
    else:
        acc = precision = recall = f1 = None
    results.append({'name': name, 'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'preds': preds})
    inference_times.append({'name': name, 'load_time': load_time, 'pred_time': pred_time})

# 4. Generate static plots
# 4.a Performance Metrics Bar
df_metrics = pd.DataFrame([r for r in results if r['accuracy'] is not None])
df_metrics = df_metrics.set_index('name')[['accuracy','precision','recall','f1']]
df_metrics.plot.bar(rot=0, figsize=(8,5))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Performance Metrics')
plt.tight_layout()
plt.savefig('plots/metrics.png')
plt.close()

# 4.b Confusion Matrix for Random Forest
rf_preds = next(r['preds'] for r in results if r['name']=='Random Forest')
cm = confusion_matrix(y_test, rf_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('plots/confusion_rf.png')
plt.close()

# 4.c t-SNE visualizations
tsne_proj = TSNE(n_components=2, random_state=42).fit_transform(X_test)
plt.figure(figsize=(6,5))
plt.scatter(tsne_proj[:,0], tsne_proj[:,1], c=y_test, s=10, alpha=0.8)
plt.title('t-SNE: True Labels')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig('plots/tsne_true.png')
plt.close()
plt.figure(figsize=(6,5))
km_preds = next(r['preds'] for r in results if r['name']=='K-Means')
plt.scatter(tsne_proj[:,0], tsne_proj[:,1], c=km_preds, s=10, alpha=0.8)
plt.title('t-SNE: K-Means Clusters')
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig('plots/tsne_kmeans.png')
plt.close()

# 4.d ROC curves for supervised
# binarize labels
y_bin = label_binarize(y_test, classes=list(range(1, num_classes+1)))
plt.figure(figsize=(8,6))
for r in results:
    if r['name'] != 'K-Means':
        model = pickle.load(open(model_paths[r['name']],'rb'))
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_test)
            # macro-average
            fpr = dict(); tpr = dict()
            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:,i], probas[:,i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= num_classes
            roc_auc = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, label=f"{r['name']} (AUC={roc_auc:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.title('Macro-Averaged ROC Curves')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('plots/roc_curves.png')
plt.close()

# 4.e Load vs Inference Time
df_times = pd.DataFrame(inference_times).set_index('name')
df_times.plot.bar(rot=0, figsize=(8,5))
plt.title('Load and Inference Time (s)')
plt.ylabel('Seconds')
plt.tight_layout()
plt.savefig('plots/times.png')
plt.close()

# 4.f Class-wise F1 heatmap (Random Forest)
pr = precision_recall_fscore_support(y_test, rf_preds, labels=range(1,num_classes+1))
f1_scores = pr[2].reshape(1, -1)
plt.figure(figsize=(10,2))
sns.heatmap(f1_scores, annot=True, cmap='viridis', cbar=False)
plt.title('Class-wise F1 Scores (Random Forest)')
plt.xlabel('Class Label (1–26)')
plt.yticks([])
plt.tight_layout()
plt.savefig('plots/f1_heatmap.png')
plt.close()



# 5. Prepare template variables
# summary
valid = [r for r in results if r['accuracy'] is not None]
top_acc = max(r['accuracy'] for r in valid)
top_f1 = max(r['f1'] for r in valid)
best_model = max(valid, key=lambda x: x['accuracy'])['name']
context = {
    'title':'ISOLET Analysis Dashboard',
    'subtitle':'Comparative evaluation of KNN, Naive Bayes, Random Forest, and K‑Means on the ISOLET spoken‑letter dataset.',
    'description':'The report examines model performance, error patterns, and timing metrics for recognizing spoken letters from the ISOLET dataset, highlighting the strengths and limitations of each algorithm.',
    'top_accuracy':f"{top_acc:.2%}",
    'top_f1':f"{top_f1:.2%}",
    'num_classes':num_classes,
    'num_features':X_test.shape[1],
    'metrics_img':'plots/metrics.png',
    'roc_img':'plots/roc_curves.png',
    'confusion_img':'plots/confusion_rf.png',
    'tsne_true_img':'plots/tsne_true.png',
    'tsne_kmeans_img':'plots/tsne_kmeans.png',
    'times_img':'plots/times.png',
    'f1_heatmap_img':'plots/f1_heatmap.png',
    'models': [
        {'name': r['name'], 'accuracy': r['accuracy'],
         'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1']}
        for r in results if r['accuracy'] is not None
    ],
    'insights':[
        f"{best_model} achieved highest accuracy of {top_acc:.2%}.",
        "ROC curves show strong separability for Random Forest.",
        "Class-wise F1 highlights harder-to-classify letters."
    ],
    'generation_date':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'footer_text':'©2025 ISOLET Analysis'
}

# 6. Render HTML
env = Environment(loader=FileSystemLoader('templates'), autoescape=True)
tmpl = env.get_template('report_template.html')
html = tmpl.render(**context)
with open('report.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('report.html generated')
webbrowser.open('report.html')
