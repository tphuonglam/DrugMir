from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def get_model_and_transform(model_type: str, random_state = 42):
    assert model_type in ["rf", "logreg", "gbm", "ada", "svm"]
    needs_standardization = False

    if model_type == "rf":
        model_class = RandomForestClassifier
        model_kwargs = {"n_estimators": 200, "max_depth": 5, "random_state": random_state}
    elif model_type == "logreg":
        model_class = LogisticRegression
        model_kwargs = {"solver": "liblinear", "random_state": random_state}
        needs_standardization = True 
    elif model_type == "gbm":
        model_class = GradientBoostingClassifier
        model_kwargs = {"n_estimators": 100, "max_depth": 3, "random_state": random_state}
    elif model_type == "ada":
        model_class = AdaBoostClassifier
        model_kwargs = {"n_estimators": 100, "random_state": random_state}
    elif model_type == "svm":
        model_class = SVC
        model_kwargs = {"probability": True, "kernel": "rbf", "random_state": random_state}
        needs_standardization = True
    else:
        # Matches the error handling in the original train_evaluate_ml
        raise ValueError(f"Unknown model_type ----> {model_type}")

    tf = StandardScaler if needs_standardization else None

    return model_class, model_kwargs, tf

# --- Example Usage ---
# model_class, needs_transform = get_model_and_transform("logreg")
# print(f"Model Class: {model_class.__name__}")
# print(f"Needs Standardization: {needs_transform}")