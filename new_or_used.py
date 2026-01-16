"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or 
used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in 
`build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was 
chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb
"""

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k_checked_v3.jsonlines")]
    def target(x): return x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test


def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary.

    Args:
        d (dict): Dictionary to flatten.
        parent_key (str): Key from parent level.
        sep (str): Separator between nested keys.
    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # If list has exactly one element, unwrap it
            if len(v) == 1:
                # If the single element is a dict, flatten it recursively
                if isinstance(v[0], dict):
                    items.extend(flatten_dict(v[0], new_key, sep=sep).items())
                else:
                    # Otherwise just keep the single value
                    items.append((new_key, v[0]))
            else:
                # Keep lists with 0 or multiple elements as-is
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)


def json_to_df(data):
    """
    Converts a list of JSON objects to a pandas DataFrame.
    """
    records = []
    for item in data:
        flattened = flatten_dict(item)
        records.append(flattened)
    return pd.DataFrame(records)


def fix_null_values(df):
    """
    Fix null values by replacing empty strings and empty lists with NaN
    """
    df_copy = df.copy()
    for column in df_copy.columns:
        if df_copy[column].dtype == 'object':
            df_copy[column] = df_copy[column].fillna(np.nan)
            # Replaces empty strings with only whitespace with NaN
            df_copy[column] = df_copy[column].replace(
                r'^\s*$', np.nan, regex=True)
            df_copy[column] = df_copy[column].apply(
                lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x
            )
        elif df_copy[column].dtype in ['int64', 'float64']:
            df_copy[column] = df_copy[column].fillna(np.nan)
        elif df_copy[column].dtype == 'bool':
            df_copy[column] = df_copy[column].fillna(False)
    return df_copy


def process_data(df):
    """
    Apply all data cleansing and transformation steps from EDA
    """
    # Fix null values
    df = fix_null_values(df)

    # Calculate null percentages
    null_pct = (df.isna().sum() / len(df)) * 100

    # Drop columns with 100% null values
    cols_100_null = null_pct[null_pct == 100].index.tolist()
    df = df.drop(columns=cols_100_null)

    # Recalculate null percentages
    null_pct = (df.isna().sum() / len(df)) * 100

    # Drop ID fields with >95% null values
    cols_95_ids = [
        col
        for col in null_pct[null_pct > 95].index
        if '_id_' in col or '_id' in col
    ]

    df = df.drop(columns=cols_95_ids)

    # Recalculate null percentages
    null_pct = (df.isna().sum() / len(df)) * 100

    # Handle fields with >95% null values
    # Convert to boolean indicators if all non-null belong to one category
    # Otherwise drop
    for column in null_pct[null_pct > 95].index:
        condition_dist = df[df[column].notna()]['condition'].value_counts()
        if len(condition_dist) == 1:
            # Single category - convert to boolean indicator
            df[f'{column}_indicator'] = df[column].notna().astype(bool)
            # Drop the original column
            df = df.drop(columns=[column])
        else:
            df = df.drop(columns=[column])

    # Drop fields with 60-95% null that are IDs
    null_60_95 = null_pct[(null_pct > 60) & (null_pct < 95)].index
    cols_to_drop = [col for col in null_60_95 if '_id_' in col or '_id' in col]
    df = df.drop(columns=cols_to_drop)

    # Recalculate null percentages
    null_pct = (df.isna().sum() / len(df)) * 100

    # Drop high null fields except variations_attribute_combinations
    null_90_plus = null_pct[(null_pct >= 90) & (null_pct < 95)].index.tolist()
    if 'variations_attribute_combinations' in null_90_plus:
        null_90_plus.remove('variations_attribute_combinations')
    df = df.drop(columns=null_90_plus)

    # Drop attributes field
    if 'attributes' in df.columns:
        df = df.drop(columns=['attributes'])

    # Create boolean features
    if 'non_mercado_pago_payment_methods_type' in df.columns:
        df['has_non_mercado_pago_methods'] = (
            df['non_mercado_pago_payment_methods_type'].notna().astype(bool)
        )
        df = df.drop(
            columns=[
                'non_mercado_pago_payment_methods_type',
                'non_mercado_pago_payment_methods_description',
                'non_mercado_pago_payment_methods'
            ],
            errors='ignore',
        )

    # Pictures feature
    if 'pictures' in df.columns:
        df['has_pictures'] = df['pictures'].notna().astype(bool)
        pics_cols = [
            col 
            for col in df.columns 
            if 'pictures' in col and col != 'has_pictures'
        ]
        df = df.drop(columns=pics_cols, errors='ignore')

    # Warranty feature
    if 'warranty' in df.columns:
        df['has_warranty'] = df['warranty'].notna().astype(bool)

    # Drop highly correlated columns (correlation > 0.8)
    # From EDA, we know these are highly correlated
    high_corr_cols = ['price', 'available_quantity']
    df = df.drop(columns=high_corr_cols, errors='ignore')

    # Drop remaining ID fields
    id_cols = [col for col in df.columns if '_id_' in col or '_id' in col]
    df = df.drop(columns=id_cols, errors='ignore')

    # Drop non-informative categorical fields
    non_info_cols = [
        'id',
        'descriptions',
        'thumbnail',
        'secure_thumbnail',
        'permalink'
    ]
    df = df.drop(columns=non_info_cols, errors='ignore')

    # Convert tags lists to strings
    if 'tags' in df.columns:
        df['tags'] = df['tags'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )

    # Handle variations_attribute_combinations
    if 'variations_attribute_combinations' in df.columns:
        df['has_variations_combinations'] = (
            df['variations_attribute_combinations'].notna().astype(bool)
        )
        df = df.drop(columns=['variations_attribute_combinations'])

    # Drop single-value columns
    single_val_cols = ['seller_address_country_name', 'international_delivery_mode']
    df = df.drop(columns=single_val_cols, errors='ignore')

    # Create time-based features
    if 'start_time' in df.columns and 'stop_time' in df.columns:
        df['listing_duration_days'] = (df['stop_time'] - df['start_time']) / (1000 * 60 * 60 * 24)
        current_time = pd.Timestamp.now().timestamp() * 1000
        df['days_since_listed'] = (current_time - df['start_time']) / (1000 * 60 * 60 * 24)
        df['listing_month'] = pd.to_datetime(df['start_time'], unit='ms').dt.month
        df['listing_day_of_week'] = pd.to_datetime(df['start_time'], unit='ms').dt.dayofweek

        # Convert to categorical
        df['listing_month'] = df['listing_month'].astype('category')
        df['listing_day_of_week'] = df['listing_day_of_week'].astype('category')

        # Drop original time columns
        df = df.drop(columns=['start_time', 'stop_time'])

    return df


def remove_outliers(df, y=None):
    """
    Remove outliers from base_price (1st and 99th percentiles) according to EDA
    Returns filtered df and corresponding y if provided
    """
    if 'base_price' in df.columns:
        threshold_low = 16.99
        threshold_high = 130000.0

        mask = (df['base_price'] >= threshold_low) & (df['base_price'] <= threshold_high)
        df = df[mask].reset_index(drop=True)

        if y is not None:
            y = [y[i] for i in range(len(mask)) if mask.iloc[i]]

    return (df, y) if y is not None else df


def transform_features(df):
    """
    Apply transformations to numeric features
    """
    # Create binary features
    if 'listing_duration_days' in df.columns:
        df['is_60day_listing'] = (df['listing_duration_days'] == 60.0).astype(bool)
        df = df.drop(columns=['listing_duration_days'])

    if 'sold_quantity' in df.columns:
        df['has_sold_quantity'] = (df['sold_quantity'] > 0).astype(bool)

    # Apply log transformations
    log_cols = ['base_price', 'initial_quantity', 'sold_quantity']
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Apply sqrt transformation
    if 'days_since_listed' in df.columns:
        df['days_since_listed'] = np.sqrt(df['days_since_listed'])

    return df


def scale_features(df_train, df_test, scaler=None):
    """
    Scale numeric features using StandardScaler
    """
    features_to_scale = [
        'base_price',
        'initial_quantity',
        'sold_quantity',
        'days_since_listed'
    ]
    features_to_scale = [f for f in features_to_scale if f in df_train.columns]

    if len(features_to_scale) > 0:
        if scaler is None:
            scaler = StandardScaler()
            df_train[features_to_scale] = scaler.fit_transform(df_train[features_to_scale])
        else:
            df_train[features_to_scale] = scaler.transform(df_train[features_to_scale])

        df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])

    return df_train, df_test, scaler


if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    print(f"Original train size: {len(X_train)}, test size: {len(X_test)}")

    # Convert to DataFrames
    print("Converting to DataFrames...")
    df_train = json_to_df(X_train)
    df_test = json_to_df(X_test)

    # Add target back to train for processing
    df_train['condition'] = y_train
    df_test['condition'] = y_test

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Process data
    print("Processing training data...")
    df_train = process_data(df_train)

    print("Processing test data...")
    df_test = process_data(df_test)

    # Remove outliers from training data only
    print("Removing outliers from training data...")
    df_train, y_train = remove_outliers(df_train, y_train)

    print(f"After outlier removal - Train size: {len(df_train)}")

    # Apply feature transformations
    print("Applying feature transformations...")
    df_train = transform_features(df_train)
    df_test = transform_features(df_test)

    # Extract target from train
    y_train = df_train['condition'].tolist()
    df_train = df_train.drop(columns=['condition'])
    y_test = df_test['condition'].tolist()
    df_test = df_test.drop(columns=['condition'])

    # Ensure both datasets have the same columns
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    # Columns only in train
    train_only = train_cols - test_cols
    if train_only:
        print(f"Dropping columns only in train: {train_only}")
        df_train = df_train.drop(columns=list(train_only))

    # Columns only in test
    test_only = test_cols - train_cols
    if test_only:
        print(f"Dropping columns only in test: {test_only}")
        df_test = df_test.drop(columns=list(test_only))

    # Reorder test columns to match train
    df_test = df_test[df_train.columns]

    # Scale numeric features
    print("Scaling numeric features...")
    df_train, df_test, scaler = scale_features(df_train, df_test)

    print(f"Final train shape: {df_train.shape}, test shape: {df_test.shape}")
    print(f"Final train labels: {len(y_train)}, test labels: {len(y_test)}")

    # Now df_train, y_train, df_test, y_test are ready for modeling
    print("\nData preprocessing complete!")
    print(f"Features ({len(df_train.columns)}): {df_train.columns.tolist()}")

    print("\n" + "="*50)
    print("CATBOOST MODEL TRAINING AND EVALUATION")
    print("="*50)

    # Identify categorical features
    categorical_features = df_train.select_dtypes(include=['category', 'object', 'bool']).columns.tolist()
    print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")

    # Convert categorical features to string type (handles NaN)
    print("\nConverting categorical features to string type...")
    for col in categorical_features:
        df_train[col] = df_train[col].astype(str)
        df_test[col] = df_test[col].astype(str)

    # Split training data into train and validation sets (80-20 split)
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        df_train, 
        y_train, 
        train_size=0.8, 
        random_state=42, 
        stratify=y_train,
    )

    print(f"\nTrain split size: {len(X_train_split)}")
    print(f"Validation split size: {len(X_valid_split)}")
    print(f"Test size: {len(df_test)}")

    # Best hyperparameters from EDA randomized search
    print("\nTraining CatBoost model with optimized hyperparameters...")
    model_CB_best = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.1,
        l2_leaf_reg=10,
        random_state=42,
        od_type="Iter",
        od_wait=100,
        verbose=False,
    )

    # Train the model
    model_CB_best.fit(
        X_train_split,
        y_train_split,
        cat_features=categorical_features,
        eval_set=(X_valid_split, y_valid_split),
        verbose=False,
    )

    print("Model training complete!")

    # Evaluate on validation set
    print("\n" + "-"*50)
    print("VALIDATION SET EVALUATION")
    print("-"*50)

    valid_data_CB = Pool(
        data=X_valid_split,
        label=y_valid_split,
        cat_features=categorical_features
    )

    predict_valid_CB = model_CB_best.predict(valid_data_CB)

    # Encode labels for metrics
    le = LabelEncoder()
    y_valid_encoded = le.fit_transform(y_valid_split)
    predict_valid_encoded = le.transform(predict_valid_CB)

    # Calculate metrics
    acc_valid = accuracy_score(y_valid_encoded, predict_valid_encoded)
    precision_valid = precision_score(y_valid_encoded, predict_valid_encoded)
    recall_valid = recall_score(y_valid_encoded, predict_valid_encoded)
    auc_valid = roc_auc_score(y_valid_encoded, predict_valid_encoded)

    print(f'\nValidation Accuracy  = {acc_valid:.4f}')
    print(f'Validation Precision = {precision_valid:.4f}')
    print(f'Validation Recall    = {recall_valid:.4f}')
    print(f'Validation AUC       = {auc_valid:.4f}')

    # Confusion Matrix
    print("\n" + "-"*50)
    print("Confusion Matrix (Validation Set)")
    print("-"*50)
    cm_valid = pd.crosstab(y_valid_split, predict_valid_CB, rownames=['True Label'], colnames=['Predicted'])
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(pd.DataFrame(cm_valid), annot=True, cmap="Blues", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion Matrix - Validation Set', y=1.1, fontsize=14)
    plt.savefig('charts/confusion_matrix_validation.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved as 'charts/confusion_matrix_validation.png'")
    plt.show()

    # ROC Curve
    print("\n" + "-"*50)
    print("ROC Curve (Validation Set)")
    print("-"*50)
    fpr_valid, tpr_valid, threshold_valid = metrics.roc_curve(y_valid_encoded, predict_valid_encoded)
    roc_auc_valid = metrics.auc(fpr_valid, tpr_valid)
    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic - Validation Set')
    plt.plot(fpr_valid, tpr_valid, 'b', label='AUC = %0.4f' % roc_auc_valid)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(alpha=0.3)
    plt.savefig('charts/roc_curve_validation.png', dpi=150, bbox_inches='tight')
    print("ROC curve saved as 'charts/roc_curve_validation.png'")
    plt.show()

    # Classification Report
    print("\n" + "-"*50)
    print("Classification Report (Validation Set)")
    print("-"*50)
    print(classification_report(y_valid_split, predict_valid_CB))

    # Evaluate on test set
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)

    test_data_CB = Pool(
        data=df_test,
        label=y_test,
        cat_features=categorical_features
    )
    predict_test_CB = model_CB_best.predict(test_data_CB)

    # Encode labels for metrics
    y_test_encoded = le.transform(y_test)
    predict_test_encoded = le.transform(predict_test_CB)

    # Calculate metrics
    acc_test = accuracy_score(y_test_encoded, predict_test_encoded)
    precision_test = precision_score(y_test_encoded, predict_test_encoded)
    recall_test = recall_score(y_test_encoded, predict_test_encoded)
    auc_test = roc_auc_score(y_test_encoded, predict_test_encoded)

    print(f'\nTest Accuracy  = {acc_test:.4f}')
    print(f'Test Precision = {precision_test:.4f}')
    print(f'Test Recall    = {recall_test:.4f}')
    print(f'Test AUC       = {auc_test:.4f}')

    # Confusion Matrix
    print("\n" + "-"*50)
    print("Confusion Matrix (Test Set)")
    print("-"*50)
    cm_test = pd.crosstab(y_test, predict_test_CB, rownames=['True Label'], colnames=['Predicted'])
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(pd.DataFrame(cm_test), annot=True, cmap="Blues", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion Matrix - Test Set', y=1.1, fontsize=14)
    plt.savefig('charts/confusion_matrix_test.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix saved as 'charts/confusion_matrix_test.png'")
    plt.show()

    # ROC Curve
    print("\n" + "-"*50)
    print("ROC Curve (Test Set)")
    print("-"*50)
    fpr_test, tpr_test, threshold_test = metrics.roc_curve(y_test_encoded, predict_test_encoded)
    roc_auc_test = metrics.auc(fpr_test, tpr_test)
    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic - Test Set')
    plt.plot(fpr_test, tpr_test, 'b', label='AUC = %0.4f' % roc_auc_test)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(alpha=0.3)
    plt.savefig('charts/roc_curve_test.png', dpi=150, bbox_inches='tight')
    print("ROC curve saved as 'charts/roc_curve_test.png'")
    plt.show()

    # Classification Report
    print("\n" + "-"*50)
    print("Classification Report (Test Set)")
    print("-"*50)
    print(classification_report(y_test, predict_test_CB))

    # Feature Importance
    print("\n" + "-"*50)
    print("Feature Importance (Top 15)")
    print("-"*50)
    feature_importance = pd.DataFrame(model_CB_best.get_feature_importance(prettified=True))
    print(feature_importance.head(15).to_string(index=False))

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#EFE9E6")
    sns.barplot(
        x="Importances",
        y="Feature Id",
        data=feature_importance.head(15),
        palette="cool",
        ax=ax
    )
    for index, row in feature_importance.head(15).iterrows():
        ax.text(
            x=row['Importances'] + 0.5,
            y=index,
            s=f"{row['Importances']:.2f}",
            color="k",
            ha="left",
            va="center",
            fontfamily='monospace'
        )
    plt.title('Top 15 Feature Importances - CatBoost Model', fontsize=14)
    plt.tight_layout()
    plt.savefig('charts/feature_importance.png', dpi=150, bbox_inches='tight')
    print("Feature importance plot saved as 'charts/feature_importance.png'")
    plt.show()

    # Summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"\n{'Metric':<20} {'Validation':<15} {'Test':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {acc_valid:<15.4f} {acc_test:<15.4f}")
    print(f"{'Precision':<20} {precision_valid:<15.4f} {precision_test:<15.4f}")
    print(f"{'Recall':<20} {recall_valid:<15.4f} {recall_test:<15.4f}")
    print(f"{'AUC':<20} {auc_valid:<15.4f} {auc_test:<15.4f}")
    print("=" * 50)

    if np.round(acc_test, 2) >= 0.86:
        print(f"\nSUCCESS: Test accuracy ({acc_test:.4f}) meets the minimum requirement of 0.86")
    else:
        print(f"\nWARNING: Test accuracy ({acc_test:.4f}) is below the minimum requirement of 0.86")
