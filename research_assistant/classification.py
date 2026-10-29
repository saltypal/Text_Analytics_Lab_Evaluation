"""Two fixed classifiers, identical splits, explicit evaluation sizes."""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .preprocessing import LABELS

def metrics(name, truth, predictions, scope):
    p, r, f, _ = precision_recall_fscore_support(truth, predictions, labels=LABELS, average='macro', zero_division=0)
    return {'System': name, 'Scope': scope, 'N': len(truth), 'Accuracy': accuracy_score(truth, predictions),
            'Precision (macro)': p, 'Recall (macro)': r, 'F1 (macro)': f}

def train_classifiers(name, x_train, x_test, train, test, seed=42, logistic_c=1.0, max_iter=3000,
                      rf_trees=200, rf_max_depth=None, rf_min_samples_leaf=1):
    results, predictions, models = [], {}, {}
    for classifier, estimator in [('Logistic Regression', LogisticRegression(C=logistic_c, max_iter=max_iter, random_state=seed)),
                                  ('Random Forest', RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_max_depth,
                                    min_samples_leaf=rf_min_samples_leaf, random_state=seed, n_jobs=-1))]:
        key = f'{name} + {classifier}'
        start = time.perf_counter()
        print(f'Training {key} ...', flush=True)
        model = make_pipeline(StandardScaler(with_mean=False), estimator) if classifier == 'Logistic Regression' else estimator
        model.fit(x_train, train.research_domain)
        pred = model.predict(x_test)
        results.append(metrics(key, test.research_domain, pred, 'Full test'))
        predictions[key], models[key] = pred, model
        print(f'Completed in {time.perf_counter()-start:.1f}s; accuracy={results[-1]["Accuracy"]:.3f}', flush=True)
    return results, predictions, models

def confusion_plots(predictions, test, output_dir, prefix='full'):
    if not predictions:
        print('No predictions available.')
        return
    fig, axes = plt.subplots(int(np.ceil(len(predictions)/2)), 2, figsize=(16, 5*int(np.ceil(len(predictions)/2))), squeeze=False)
    short = ['NLP', 'Health AI', 'Cybersec', 'Vision', 'Robotics']
    for ax, (name, pred) in zip(axes.flat, predictions.items()):
        invalid = any(label not in LABELS for label in pred)
        labels = LABELS + ['INVALID'] if invalid else LABELS
        ticks = short + ['INVALID'] if invalid else short
        cm = confusion_matrix(test.research_domain, pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ticks, yticklabels=ticks, ax=ax, cbar=False)
        ax.set(title=name, xlabel='Predicted domain', ylabel='True domain')
    for ax in list(axes.flat)[len(predictions):]:
        ax.set_visible(False)
    plt.tight_layout()
    fig.savefig(output_dir / f'{prefix}_confusion_matrices.png', dpi=140, bbox_inches='tight')
    plt.show()

def misclassified(predictions, test):
    frames = []
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        wrong = pred != test.research_domain.to_numpy()
        part = test.loc[wrong, ['paper_id', 'title', 'abstract', 'research_domain']].copy()
        part['Predicted'], part['System'] = pred[wrong], name
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
