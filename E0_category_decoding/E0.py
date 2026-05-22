# E0: Category Decoding with ERP Features using SVM 

import os
import mne
import argparse
import numpy as np
import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utilz.preprocessing_utilz import load_erp_features, load_stim_metadata

parser = argparse.ArgumentParser(description='ERP Category Decoding')
parser.add_argument('--data-dir', type=str, required=True, help='Path to the preprocessed ERP features directory')
parser.add_argument('--config', type=int, default=2, required=False, help='Configuration number for preprocessing ')
parser.add_argument('--topk-cats', type=int, default=5, required=False, help='Number of top categories to consider for decoding')
parser.add_argument('--out-dir', type=str, default='results', required=False, help='Directory to save results')
args = parser.parse_args()


def run_decoding(X: np.ndarray, y: np.ndarray, n_splits: int = 5, C: float = 0.01, random_state: int = 42):
    """
    X: (n_samples, n_features)
    y: string or integer labels
    n_splits: number of cross-validation splits
    C: SVM regularization parameter
    random_state: random state for reproducibility
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(C=C, dual=False, max_iter=10000))
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_val_score(clf, X, y_enc, cv=cv, scoring='accuracy')

    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_scores = cross_val_score(dummy_clf, X, y_enc, cv=cv, scoring='accuracy')
  

    return {
        'acc_mean': float(np.mean(scores)),
        'acc_std': float(np.std(scores)),
        'acc_folds': scores,
        'chance_mean': float(np.mean(dummy_scores)),
        'chance_std': float(np.std(dummy_scores)),
        'n_classes': len(le.classes_),
        'classes': list(le.classes_)
    }


def make_topk_mask(categories: list, topk: int):
    cats = pd.Series(categories)
    topk_categories = cats.value_counts().nlargest(topk+1).index.tolist() #+1 to account for 'nan' category
    topk_categories = [cat for cat in topk_categories if cat != 'nan']
    topk_mask = np.isin(categories, topk_categories)
    print(f"Top {topk} categories: {cats[topk_mask].value_counts().to_dict()}")
    return topk_mask, topk_categories

def main():

    result_dicts = []


    for subject_id in range(1, 6):
        print(f"Processing subject {subject_id}...")
        subject_dict = {'subject_id': subject_id}
        
        # load preprocessed ERP features
        erp_path = os.path.join(args.data_dir, f'config{args.config}', f"sub-{subject_id:02d}_train_erp_features.h5")
        erp_data = load_erp_features(erp_path)

        stim_metadata_path = os.path.join(args.data_dir, f'config{args.config}', f"sub-{subject_id:02d}_train_stim_metadata.h5")
        stim_metadata = load_stim_metadata(stim_metadata_path)
        
        topk_mask, topk_categories = make_topk_mask(stim_metadata['categories'], topk=args.topk_cats)

        # filter to top-k categories
        stim_metadata['categories'] = np.array(stim_metadata['categories'])[topk_mask]
        stim_metadata['stim_ids'] = np.array(stim_metadata['stim_ids'])[topk_mask]
        for comp in erp_data['components'].keys():
            erp_data['components'][comp]['vector'] = erp_data['components'][comp]['vector'][topk_mask, :]
            erp_data['components'][comp]['scalar'] = erp_data['components'][comp]['scalar'][topk_mask]

        subject_dict['n_samples'] = len(stim_metadata['categories'])
        subject_dict['n_classes'] = len(topk_categories)

        # run decoding for each component
        for comp in erp_data['components'].keys():
            print(f"  Decoding {comp} component...")
            X = erp_data['components'][comp]['vector']
            y = stim_metadata['categories']
            results = run_decoding(X, y)
            print(f"    Accuracy: {results['acc_mean']:.4f} ± {results['acc_std']:.4f} (chance: {results['chance_mean']:.4f}, n_classes: {results['n_classes']})")
            subject_dict[f'{comp}_acc_mean'] = results['acc_mean']
            subject_dict[f'{comp}_acc_std'] = results['acc_std']
            subject_dict[f'{comp}_chance_mean'] = results['chance_mean']

            # save n_ch and n_t for reference
            subject_dict[f'{comp}_n_channels'] = len(erp_data['components'][comp]['ch_used'])
            subject_dict[f'{comp}_n_timepoints'] = len(erp_data['components'][comp]['t_used'])
        result_dicts.append(subject_dict)

    results_df = pd.DataFrame(result_dicts)
    print("\nFinal Results:")
    print(results_df)

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, f'category_decoding_results_c{args.config}_top{args.topk_cats}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")










if __name__ == "__main__": main()