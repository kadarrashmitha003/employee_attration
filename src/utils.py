import pickle
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    report = {}
    for name, model in models.items():
        param = params.get(name, {})
        gs = GridSearchCV(model, param, cv=3, n_jobs=-1, scoring='r2')
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        from sklearn.metrics import r2_score
        score = r2_score(y_test, y_pred)
        report[name] = score
    return report
