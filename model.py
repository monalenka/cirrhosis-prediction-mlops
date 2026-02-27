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
import logging
import sys
from logging.handlers import RotatingFileHandler
import traceback
from clearml import Task

def get_logger(name: str = "cirrhosis") -> logging.Logger:
    os.makedirs("data", exist_ok=True)
    log_path = "./data/log_file.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class My_Classifier_Model:
    def __init__(self, model_dir: str = "./model"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.logger = get_logger()
        self.logger.info(f"INIT - model_dir={self.model_dir}")

        self.xgb_params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "predictor": "auto",
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.02,
            "reg_lambda": 2.0,
            "n_estimators": 4000,
        }

        self.use_cat_gpu = self._catboost_try_gpu()
        self.use_xgb_gpu = self._xgb_try_gpu()

        self.logger.info(f"DEVICE - CatBoost GPU={self.use_cat_gpu}, XGBoost GPU={self.use_xgb_gpu}")

        if self.use_xgb_gpu:
            self.xgb_params["tree_method"] = "gpu_hist"
            self.xgb_params["predictor"] = "gpu_predictor"
        else:
            self.xgb_params["tree_method"] = "hist"
            self.xgb_params["predictor"] = "auto"

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

        self.best_params: Optional[Dict[str, Any]] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.cat_cols: Optional[List[str]] = None
        self.xgb_feature_names: Optional[List[str]] = None
        self.best_w: Optional[float] = None
        self.catboost_model_paths: List[Dict[str, str]] = []
        self.xgboost_model_paths: List[Dict[str, str]] = []

    def _catboost_try_gpu(self) -> bool:
        try:
            probe = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                iterations=5,
                task_type="GPU",
                devices="0",
                verbose=False,
            )
            Xp = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            yp = [0, 1, 2]
            probe.fit(Xp, yp, cat_features=["b"])
            return True
        except Exception as e:
            self.logger.info(f"CatBoost GPU unavailable -> CPU. Reason: {e}")
            return False


    def _xgb_try_gpu(self) -> bool:
        try:
            Xp = np.random.randn(20, 5).astype(np.float32)
            yp = np.random.randint(0, 3, size=20)
            probe = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                n_estimators=10,
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                random_state=42,
            )
            probe.fit(Xp, yp, eval_set=[(Xp, yp)], verbose=False)
            return True
        except Exception as e:
            self.logger.info(f"XGBoost GPU unavailable -> CPU. Reason: {e}")
            return False


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
        }

        if self.use_cat_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"

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
        task = Task.init(
        project_name="Cirrhosis Project",
        task_name="Training run",
        output_uri=True
        )

        self.logger.info(f"TRAIN - started train_path={train_path} test_path={test_path}")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            self.logger.info(f"TRAIN - train shape={train_df.shape} test shape={test_df.shape}")
            self.logger.info(f"TRAIN - train columns={list(train_df.columns)}")
            
            task.upload_artifact("train_dataset", train_df)
            task.upload_artifact("test_dataset", test_df)

            X_train, y_train = self._preprocess_train(train_df)

            X_test = self._preprocess_test(test_df)

            self.logger.info("TRAIN - starting Optuna optimization")
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
            })

            if self.use_cat_gpu:
                self.best_params.update({
                    "task_type": "GPU", 
                    "devices": "0"
                })

            task.connect(self.best_params, name="Best Params")
            task.connect({"cat_seeds": self.cat_seeds, "xgb_seeds": self.xgb_seeds})                

            self.logger.info("TRAIN - Training CatBoost ensemble...")
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
            self.logger.info(f"TRAIN - CatBoost ensemble CV logloss: {cat_cv_logloss:.6f}")

            task.get_logger().report_scalar("CV Log Loss", "CatBoost", value=cat_cv_logloss, iteration=0)

            self.logger.info("TRAIN - Preparing data for XGBoost...")
            X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
            X_all = pd.get_dummies(X_all, columns=self.cat_cols, dummy_na=True)
            X_all = X_all.astype(np.float32)

            X_xgb = X_all.iloc[:len(X_train)].reset_index(drop=True)
            X_test_xgb = X_all.iloc[len(X_train):].reset_index(drop=True)
            self.xgb_feature_names = X_xgb.columns.tolist()

            self.logger.info("TRAIN - Training XGBoost ensemble...")
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
            self.logger.info(f"TRAIN - XGBoost ensemble CV logloss: {xgb_cv_logloss:.6f}")

            task.get_logger().report_scalar("CV Log Loss", "XGBoost", value=xgb_cv_logloss, iteration=0)

            self.logger.info("TRAIN - Searching for optimal blending weight...")
            weights = np.linspace(0.0, 1.0, 101)
            best_w, best_score = 1.0, 10.0
            for w in weights:
                blend_oof = w * cat_oof + (1.0 - w) * xgb_oof
                score = log_loss(y_train, blend_oof)
                if score < best_score:
                    best_score = score
                    best_w = w

            self.best_w = best_w
            self.logger.info(f"TRAIN - Optimal CatBoost weight: {best_w:.2f}")
            self.logger.info(f"TRAIN - Blending CV logloss: {best_score:.6f}")

            task.get_logger().report_scalar("CV Log Loss", "Blending", value=best_score, iteration=0)

            self._save_artifacts()

            task.upload_artifact("model_folder", self.model_dir)

            self.logger.info("TRAIN - finished successfully")
        except Exception:
            self.logger.exception("TRAIN - failed with exception")
            
            task.mark_failed()
            raise

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

        with open(os.path.join(self.model_dir, "best_w.json"), "w") as f:
            json.dump(float(self.best_w), f)

        with open(os.path.join(self.model_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(self.model_dir, "model_paths.json"), "w") as f:
            json.dump({
                "catboost": self.catboost_model_paths,
                "xgboost": self.xgboost_model_paths
            }, f, indent=2, default=str)

        self.logger.info(f"All artifacts saved to {self.model_dir}")

    def _load_artifacts(self):
        with open(os.path.join(self.model_dir, "best_params.json"), "r") as f:
            self.best_params = json.load(f)
        with open(os.path.join(self.model_dir, "cat_cols.json"), "r") as f:
            self.cat_cols = json.load(f)
        with open(os.path.join(self.model_dir, "xgb_feature_names.json"), "r") as f:
            self.xgb_feature_names = json.load(f)
        with open(os.path.join(self.model_dir, "best_w.json"), "r") as f:
            self.best_w = float(json.load(f))
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

    def predict(self, test_path: str, output_path: str = "./data/results.csv"):
        self.logger.info(f"PREDICT - started test_path={test_path} output_path={output_path}")
        try:
            self._load_artifacts()
            test_df = pd.read_csv(test_path)
            self.logger.info(f"PREDICT - test shape={test_df.shape}")
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

            self.logger.info(f"PREDICT - loaded cat models={n_cat_models}, xgb models={n_xgb_models}")

            final_preds = self.best_w * cat_preds + (1.0 - self.best_w) * xgb_preds
            self.logger.info(f"PREDICT - proba stats: min={final_preds.min():.6f} max={final_preds.max():.6f} mean={final_preds.mean():.6f}")
            submission = pd.DataFrame({
                "id": test_ids,
                "Status_C": final_preds[:, 0],
                "Status_CL": final_preds[:, 1],
                "Status_D": final_preds[:, 2],
            })
            submission.to_csv(output_path, index=False)
            self.logger.info(f"PREDICT - Submission saved to {output_path}")
            self.logger.info("PREDICT - finished successfully")
        except Exception:
            self.logger.exception("PREDICT - failed with exception")
            raise

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