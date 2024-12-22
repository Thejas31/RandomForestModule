from flask import Flask, render_template, request, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error, r2_score,
                              mean_absolute_error, confusion_matrix, roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.xls', '.tsv', '.txt']
app.config['UPLOAD_FOLDER'] = './uploads'
df = None

def advanced_preprocess_data(df, target_column, columns_to_remove, 
                              missing_strategy, scaling_method, 
                              encoding_method):
    """
    Advanced preprocessing method with multiple options
    
    Parameters:
    - df: Input DataFrame
    - target_column: Column to be used as target variable
    - columns_to_remove: List of columns to drop
    - missing_strategy: Strategy for handling missing values ('mean', 'median', 'most_frequent')
    - scaling_method: Feature scaling method ('standard', 'minmax', 'robust')
    - encoding_method: Categorical encoding method ('label', 'onehot')
    
    Returns:
    - Preprocessed features, target, and encoders/scalers
    """
    # Create a copy of the dataframe to avoid modifying the original
    data1 = df.copy()
    data=data1.drop(target_column, axis=1)
    # Replace problematic values
    data.replace(['?', 'None', ''], np.nan, inplace=True)
    
    # Drop specified columns
    if columns_to_remove:
        data.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    # Handle missing values
    numeric_imputer = SimpleImputer(strategy=missing_strategy)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Impute numeric columns
    if len(numeric_columns) > 0:
        data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    # Impute categorical columns
    if len(categorical_columns) > 0:
        data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
    
    # Encoding categorical variables
    label_encoders = {}
    if encoding_method == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    
    # Scaling features
    scaler = None
    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    
    # Apply scaling to numeric columns
    if scaler and len(numeric_columns) > 0:
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    # Separate target and features
    if target_column and target_column in data1.columns:
        features = data
        target = data1[target_column]
    else:
        features = data1
        target = None
    
    return features, target, label_encoders, scaler


def encode_categorical_columns(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders

def plot_confusion_matrix(y_true, y_pred, label_encoder=None):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    if label_encoder:
        class_names = label_encoder.classes_
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, 
                    yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    confusion_matrix_url = base64.b64encode(buf.getvalue()).decode("utf8")
    buf.close()
    plt.close()
    
    return confusion_matrix_url

def plot_roc_curves(y_true, y_prob, label_encoder=None):
    plt.figure(figsize=(10, 8))
    
    if y_prob.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
    else:
        class_index = 1
        fpr, tpr, _ = roc_curve(y_true == class_index, y_prob[:, class_index])
        roc_auc = auc(fpr, tpr)
        class_name = label_encoder.classes_[class_index] if label_encoder else f'Class {class_index}'
        plt.plot(fpr, tpr, color='darkorange', 
                 label=f'ROC curve of {class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    roc_curve_url = base64.b64encode(buf.getvalue()).decode("utf8")
    buf.close()
    plt.close()
    
    return roc_curve_url

@app.route("/", methods=["GET", "POST"])
def main():
    global df
    data_preview, column_types = None, None
    accuracy = None
    mse = None 
    mae = None
    report = None
    plot_url = None
    confusion_matrix_url = None
    roc_curve_url = None
    columns = []
    preprocess_message = ""
    auc_score = None
    regression_curve_url = None
    model_inference = ""
    r2=None
    statistical_summary = None

    try:
        if request.method == "POST":
            if "file" in request.files:
                file = request.files["file"]
                session['filename'] = file.filename
                file_extension = file.filename.split('.')[-1].lower()

                if f".{file_extension}" not in app.config['UPLOAD_EXTENSIONS']:
                    flash("Invalid file type. Please upload a CSV, Excel, TSV, or TXT file.")
                    return render_template("index.html", columns=columns, filename="No file chosen")

                try:
                    if file_extension == "csv":
                        df = pd.read_csv(file)
                    elif file_extension in ["xlsx", "xls"]:
                        df = pd.read_excel(file, engine='openpyxl')
                    elif file_extension in ["tsv", "txt"]:
                        df = pd.read_csv(file, sep='\t')

                    columns = df.columns.tolist()
                    session['columns'] = columns
                    data_preview = df.head(10).to_html(classes="table table-striped", index=False)
                    column_types = df.dtypes.to_dict()

                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_data.csv"), index=False)
                    df.dropna(inplace = True)
                    df = df.apply(pd.to_numeric,errors = 'ignore')
                    session['columns'] = columns 
                    
                except Exception as e:
                    flash(f"Error reading file: {str(e)}")
                    return render_template("index.html", columns=columns, filename="No file chosen")

            elif request.form.get("action") == "Generate Plot":
                if df is not None:
                    x_axis = request.form.get("x_axis")
                    y_axis = request.form.get("y_axis")
                    plot_type = request.form.get("plot_type")

                    if x_axis and y_axis and plot_type:
                        plt.figure(figsize=(8, 6))
                        if plot_type == "scatter":
                            sns.scatterplot(data=df, x=x_axis, y=y_axis)
                        elif plot_type == "line":
                            sns.lineplot(data=df, x=x_axis, y=y_axis)
                        elif plot_type == "bar":
                            sns.barplot(data=df, x=x_axis, y=y_axis)
                        elif plot_type == "hist":
                            sns.histplot(data=df[x_axis], kde=True)

                        buf = io.BytesIO()
                        plt.savefig(buf, format="png")
                        buf.seek(0)
                        plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
                        buf.close()
                        plt.close()
            elif request.form.get("action") == "Statistical Summary":
                if df is not None:
                    try:
                        # Generate a statistical summary for numerical columns
                        statistical_summary = {
                            col: {
                                "count": df[col].count(),
                                "mean": df[col].mean(),
                                "std": df[col].std(),
                                "min": df[col].min(),
                                "q25": df[col].quantile(0.25),
                                "q50": df[col].median(),
                                "q75": df[col].quantile(0.75),
                                "max": df[col].max(),
                            }
                            for col in df.select_dtypes(include=[np.number])
                        }
            
                        # Convert the summary to HTML
                        statistical_summary_html = (
                            pd.DataFrame(statistical_summary).transpose()
                            .to_html(classes="table table-striped", index=True)
                        )
                        statistical_summary = statistical_summary_html  # Update this line to pass the HTML to the template
                    except Exception as e:
                        flash(f"An error occurred while generating statistical inference: {str(e)}")
                        statistical_summary = None
                else:
                    flash("Please upload a dataset first.")


            elif request.form.get("action") == "Apply Advanced Preprocessing":
                target_column = request.form.get("target_column")
                columns_to_remove = request.form.getlist("columns_to_remove")
                missing_strategy = request.form.get("missing_strategy")
                scaling_method = request.form.get("scaling_method")
                encoding_method = request.form.get("encoding_method")

                features, target, label_encoders, scaler = advanced_preprocess_data(
                    df, 
                    target_column=target_column, 
                    columns_to_remove=columns_to_remove,
                    missing_strategy=missing_strategy,
                    scaling_method=scaling_method,
                    encoding_method=encoding_method
                )

                # Combine features and target
                preprocessed_df = features.copy()
                if target is not None:
                    preprocessed_df[target_column] = target

                # Save preprocessed data
                preprocessed_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "preprocessed_data.csv"), index=False)

                # Update global dataframe
                df = preprocessed_df
                session['columns'] = df.columns.tolist()  # Refresh session columns
                preprocess_message = f"Advanced preprocessing completed. Columns removed: {columns_to_remove}, "\
                                     f"Missing strategy: {missing_strategy}, "\
                                     f"Scaling method: {scaling_method}, "\
                                     f"Encoding method: {encoding_method}"
                
                # Update columns and preview
                columns = df.columns.tolist()
                data_preview = df.head(10).to_html(classes="table table-striped", index=False)
                column_types = df.dtypes.to_dict()
                
            elif request.form.get("action") == "Train Model" and df is not None:
                target_column = request.form.get("target")
                n_estimators = int(request.form.get("n_estimators", 100))
                max_depth = int(request.form.get("max_depth", None))  # None for unlimited depth
                cross_validation = request.form.get("cross_validation") == "on"
                optimization_method = request.form.get("optimization_method")

                X = df.drop(target_column, axis=1)
                y = df[target_column]
                X, label_encoders = encode_categorical_columns(X)
                is_classification = y.dtype == 'object' or len(y.unique()) < 20

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                if is_classification:
                    label_encoder_target = LabelEncoder()
                    y_train = label_encoder_target.fit_transform(y_train)
                    y_test = label_encoder_target.transform(y_test)
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

                param_grid = {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [None, 10, 20, 30]
                }

                if optimization_method == "grid_search":
                    search = GridSearchCV(model, param_grid, cv=5)
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                elif optimization_method == "random_search":
                    search = RandomizedSearchCV(model, param_grid, cv=5, n_iter=15)
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                if is_classification:
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred)
                    y_prob = model.predict_proba(X_test)
                    confusion_matrix_url = plot_confusion_matrix(y_test, y_pred, label_encoder_target)
                    roc_curve_url = plot_roc_curves(y_test, y_prob, label_encoder_target)
                    
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 2:
                        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    else:
                        auc_score = roc_auc_score(y_test, y_prob[:, 1])
                    if accuracy >= 0.85:
                        model_inference = "The model is performing excellently!"
                    elif 0.70 <= accuracy < 0.85:
                        model_inference = "The model is performing well, but there's room for improvement."
                    else:
                        model_inference = "The model needs improvement."
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    report = f"MSE: {mse}, MAE: {mae}"
                    if r2 >= 0.85:
                        model_inference = "The model is performing excellently in predicting the target values."
                    elif 0.60 <= r2 < 0.85:
                        model_inference = "The model is reasonably good, but it could be optimized further."
                    else:
                        model_inference = "The model requires significant improvement for reliable predictions."
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Actual vs Predicted')
                    min_val = min(min(y_test), min(y_pred))
                    max_val = max(max(y_test), max(y_pred))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                    plt.title('Actual vs Predicted Values')
                    plt.xlabel('Actual Values')
                    plt.ylabel('Predicted Values')
                    plt.legend()
                    plt.grid(True)

                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    regression_curve_url = base64.b64encode(buf.getvalue()).decode('utf-8')
                    buf.close()
                    plt.close()

    except Exception as e:
        flash(f"An error occurred: {str(e)}")
    print("Model Inference:", model_inference)
    

    return render_template("index.html",
                         columns=session.get('columns', []),
                         filename=session.get('filename', 'No file chosen'),
                         data_preview=data_preview,
                         column_types=column_types,
                         accuracy=accuracy,
                         mse=mse,
                         mae=mae,
                         r2 = r2,
                         report=report,
                         plot_url=plot_url,
                         confusion_matrix_url=confusion_matrix_url,
                         roc_curve_url=roc_curve_url,
                         auc_score=auc_score,
                         regression_curve_url=regression_curve_url,
                         model_inference = model_inference,
                         preprocess_message=preprocess_message,
                         statistical_summary=statistical_summary)

if __name__ == "__main__":
    app.run(debug=True)
    