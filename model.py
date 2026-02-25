import os
import json
import pickle
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from typing import Optional, Tuple, List, Dict, Any


class My_Classifier_Model:

    def __init__(self, model_dir: str = "./model"):

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.num_features = [
            "N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin", "Copper",
            "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", "Prothrombin"
        ]
        self.skewed = ["Alk_Phos", "Copper", "SGOT", "Tryglicerides"]
        self.cat_seeds = [13, 42, 2024, 777]
        self.xgb_seeds = [42, 2024, 777]
        self.n_splits = 7
        self.skf_random_state = 42
        self.MISSING = "__MISSING__"
        self.eps = 1e-6

        self.xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.02,
            "reg_lambda": 2.0,
            "n_estimators": 4000,
        }

        self.best_params: Optional[Dict[str, Any]] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.cat_cols: Optional[List[str]] = None
        self.xgb_feature_names: Optional[List[str]] = None
        self.best_w: Optional[float] = None
        self.catboost_model_paths: List[Dict[str, str]] = []
        self.xgboost_model_paths: List[Dict[str, str]] = []

    def _preprocess_train(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        df = df.copy()

        if "id" in df.columns:
            df.drop("id", axis=1, inplace=True)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df["Status"])
        df.drop("Status", axis=1, inplace=True)

        if "Stage" in df.columns:
            df["Stage"] = df["Stage"].astype(str)
        self.cat_cols = df.select_dtypes(include="object").columns.tolist()

        for col in self.skewed:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        for c in self.cat_cols:
            df[c] = df[c].fillna(self.MISSING).astype(str)

        if "Bilirubin" in df.columns and "Albumin" in df.columns:
            df["Bili_Alb"] = df["Bilirubin"] / (df["Albumin"] + self.eps)
        if "Alk_Phos" in df.columns and "SGOT" in df.columns:
            df["Alk_SGOT"] = df["Alk_Phos"] / (df["SGOT"] + self.eps)
        if "Platelets" in df.columns and "Prothrombin" in df.columns:
            df["Plat_Prot"] = df["Platelets"] / (df["Prothrombin"] + self.eps)

        for col in df.columns:
            if df[col].isna().sum() > 0 and col not in self.cat_cols:
                df[col + "_isna"] = df[col].isna().astype(int)

        return df, y

    def _preprocess_test(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "id" in df.columns:
            df.drop("id", axis=1, inplace=True)
        if "Stage" in df.columns:
            df["Stage"] = df["Stage"].astype(str)

        for col in self.skewed:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        for c in self.cat_cols:
            if c in df.columns:
                df[c] = df[c].fillna(self.MISSING).astype(str)

        if "Bilirubin" in df.columns and "Albumin" in df.columns:
            df["Bili_Alb"] = df["Bilirubin"] / (df["Albumin"] + self.eps)
        if "Alk_Phos" in df.columns and "SGOT" in df.columns:
            df["Alk_SGOT"] = df["Alk_Phos"] / (df["SGOT"] + self.eps)
        if "Platelets" in df.columns and "Prothrombin" in df.columns:
            df["Plat_Prot"] = df["Platelets"] / (df["Prothrombin"] + self.eps)

        for col in df.columns:
            if df[col].isna().sum() > 0 and col not in self.cat_cols:
                df[col + "_isna"] = df[col].isna().astype(int)
        return df

    def _objective(self, trial, X: pd.DataFrame, y: np.ndarray, cat_cols: List[str]) -> float:

        params = {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "iterations": trial.suggest_int("iterations", 1800, 4500),
            "learning_rate": trial.suggest_float("learning_rate", 0.012, 0.06, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 8.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 128),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 64),
            "bootstrap_type": "Bayesian",
            "random_seed": 42,
            "verbose": 200,
            "early_stopping_rounds": 300,
            "task_type": "GPU",
            "devices": "0",
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros((len(X), 3))

        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=(X_val, y_val),
                cat_features=cat_cols,
                use_best_model=True,
                verbose=False
            )
            oof[val_idx] = model.predict_proba(X_val)

        return log_loss(y, oof)

    def train(self, train_path: str, test_path: str):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train = self._preprocess_train(train_df)

        X_test = self._preprocess_test(test_df)

        print("Starting Optuna optimization...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, self.cat_cols),
            n_trials=30,
            show_progress_bar=True
        )

        self.best_params = study.best_params
        self.best_params.update({
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "bootstrap_type": "Bayesian",
            "verbose": 200,
            "early_stopping_rounds": 300,
            "task_type": "GPU",
            "devices": "0",
        })

        print("Training CatBoost ensemble...")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.skf_random_state)
        cat_oof = np.zeros((len(X_train), 3))
        cat_test = np.zeros((len(X_test), 3))

        for seed in self.cat_seeds:
            seed_oof = np.zeros((len(X_train), 3))
            seed_test = np.zeros((len(X_test), 3))
            seed_model_paths = {}

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                model = CatBoostClassifier(**self.best_params, random_seed=seed)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=(X_val, y_val),
                    cat_features=self.cat_cols,
                    use_best_model=True,
                    verbose=False
                )
                seed_oof[val_idx] = model.predict_proba(X_val)
                seed_test += model.predict_proba(X_test) / skf.n_splits

                model_path = os.path.join(self.model_dir, f"catboost_seed_{seed}_fold_{fold}.cbm")
                model.save_model(model_path)
                seed_model_paths[f"fold_{fold}"] = model_path

            cat_oof += seed_oof / len(self.cat_seeds)
            cat_test += seed_test / len(self.cat_seeds)
            self.catboost_model_paths.append({f"seed_{seed}": seed_model_paths})

        cat_cv_logloss = log_loss(y_train, cat_oof)
        print(f"CatBoost ensemble CV logloss: {cat_cv_logloss:.6f}")

        print("Preparing data for XGBoost...")
        X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        X_all = pd.get_dummies(X_all, columns=self.cat_cols, dummy_na=True)
        X_all = X_all.astype(np.float32)

        X_xgb = X_all.iloc[:len(X_train)].reset_index(drop=True)
        X_test_xgb = X_all.iloc[len(X_train):].reset_index(drop=True)
        self.xgb_feature_names = X_xgb.columns.tolist()

        print("Training XGBoost ensemble...")
        xgb_oof = np.zeros((len(X_xgb), 3))
        xgb_test = np.zeros((len(X_test_xgb), 3))

        for seed in self.xgb_seeds:
            seed_oof = np.zeros((len(X_xgb), 3))
            seed_test = np.zeros((len(X_test_xgb), 3))
            seed_model_paths = {}

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_xgb, y_train)):
                X_tr, X_val = X_xgb.iloc[tr_idx], X_xgb.iloc[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                model = xgb.XGBClassifier(
                    **self.xgb_params,
                    random_state=seed,
                    early_stopping_rounds=200,
                )
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                seed_oof[val_idx] = model.predict_proba(X_val)
                seed_test += model.predict_proba(X_test_xgb) / skf.n_splits

                model_path = os.path.join(self.model_dir, f"xgboost_seed_{seed}_fold_{fold}.json")
                model.save_model(model_path)
                seed_model_paths[f"fold_{fold}"] = model_path

            xgb_oof += seed_oof / len(self.xgb_seeds)
            xgb_test += seed_test / len(self.xgb_seeds)
            self.xgboost_model_paths.append({f"seed_{seed}": seed_model_paths})

        xgb_cv_logloss = log_loss(y_train, xgb_oof)
        print(f"XGBoost ensemble CV logloss: {xgb_cv_logloss:.6f}")

        print("Searching for optimal blending weight...")
        weights = np.linspace(0.0, 1.0, 101)
        best_w, best_score = 1.0, 10.0
        for w in weights:
            blend_oof = w * cat_oof + (1.0 - w) * xgb_oof
            score = log_loss(y_train, blend_oof)
            if score < best_score:
                best_score = score
                best_w = w

        self.best_w = best_w
        print(f"Optimal CatBoost weight: {best_w:.2f}")
        print(f"Blending CV logloss: {best_score:.6f}")

        self._save_artifacts()

    def _save_artifacts(self):
        """Сохранение всех необходимых объектов в model_dir."""
        artifacts = {
            "best_params.json": self.best_params,
            "cat_cols.json": self.cat_cols,
            "xgb_feature_names.json": self.xgb_feature_names,
            "best_w.json": self.best_w,
            "cat_seeds.json": self.cat_seeds,
            "xgb_seeds.json": self.xgb_seeds,
            "skf_params.json": {"n_splits": self.n_splits, "random_state": self.skf_random_state},
        }
        for filename, data in artifacts.items():
            with open(os.path.join(self.model_dir, filename), "w") as f:
                json.dump(data, f, indent=2, default=str)

        with open(os.path.join(self.model_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(self.model_dir, "model_paths.json"), "w") as f:
            json.dump({
                "catboost": self.catboost_model_paths,
                "xgboost": self.xgboost_model_paths
            }, f, indent=2, default=str)

        print(f"All artifacts saved to {self.model_dir}")

    def _load_artifacts(self):
        with open(os.path.join(self.model_dir, "best_params.json"), "r") as f:
            self.best_params = json.load(f)
        with open(os.path.join(self.model_dir, "cat_cols.json"), "r") as f:
            self.cat_cols = json.load(f)
        with open(os.path.join(self.model_dir, "xgb_feature_names.json"), "r") as f:
            self.xgb_feature_names = json.load(f)
        with open(os.path.join(self.model_dir, "best_w.json"), "r") as f:
            self.best_w = json.load(f)
        with open(os.path.join(self.model_dir, "cat_seeds.json"), "r") as f:
            self.cat_seeds = json.load(f)
        with open(os.path.join(self.model_dir, "xgb_seeds.json"), "r") as f:
            self.xgb_seeds = json.load(f)
        with open(os.path.join(self.model_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(os.path.join(self.model_dir, "model_paths.json"), "r") as f:
            paths = json.load(f)
            self.catboost_model_paths = paths["catboost"]
            self.xgboost_model_paths = paths["xgboost"]

    def predict(self, test_path: str, output_path: str = "submission.csv"):
        self._load_artifacts()

        test_df = pd.read_csv(test_path)
        if "id" in test_df.columns:
            test_ids = test_df["id"].copy().to_list()
        else:
            test_ids = list(range(len(test_df)))

        X_test_raw = self._preprocess_test(test_df)

        cat_preds = np.zeros((len(X_test_raw), 3))
        n_cat_models = 0

        for seed_dict in self.catboost_model_paths:
            for seed_key, fold_dict in seed_dict.items():
                for fold_key, model_path in fold_dict.items():
                    model = CatBoostClassifier()
                    model.load_model(model_path)
                    cat_preds += model.predict_proba(X_test_raw)
                    n_cat_models += 1

        cat_preds /= n_cat_models


        X_test_dummies = pd.get_dummies(X_test_raw, columns=self.cat_cols, dummy_na=True)
        X_test_dummies = X_test_dummies.astype(np.float32)

        for col in self.xgb_feature_names:
            if col not in X_test_dummies.columns:
                X_test_dummies[col] = 0
        X_test_dummies = X_test_dummies[self.xgb_feature_names]

        xgb_preds = np.zeros((len(X_test_dummies), 3))
        n_xgb_models = 0

        for seed_dict in self.xgboost_model_paths:
            for seed_key, fold_dict in seed_dict.items():
                for fold_key, model_path in fold_dict.items():
                    model = xgb.XGBClassifier()
                    model.load_model(model_path)
                    xgb_preds += model.predict_proba(X_test_dummies)
                    n_xgb_models += 1
        xgb_preds /= n_xgb_models

        final_preds = self.best_w * cat_preds + (1.0 - self.best_w) * xgb_preds

        submission = pd.DataFrame({
            "id": test_ids,
            "Status_C": final_preds[:, 0],
            "Status_CL": final_preds[:, 1],
            "Status_D": final_preds[:, 2],
        })
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1]
        model = My_Classifier_Model(model_dir="./model")
        
        if command == "train":
            train_path = sys.argv[2] if len(sys.argv) > 2 else "train.csv"
            test_path = sys.argv[3] if len(sys.argv) > 3 else "test.csv"
            model.train(train_path, test_path)
        elif command == "predict":
            test_path = sys.argv[2] if len(sys.argv) > 2 else "test.csv"
            output_path = sys.argv[3] if len(sys.argv) > 3 else "submission.csv"
            model.predict(test_path, output_path)
        else:
            print("Unknown command. Use 'train' or 'predict'")
    else:
        print("Usage: python model.py [train|predict] [options]")