from wafer.logger import lg
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BestModelSelection:
    lg.info(
        f'Entered the "{os.path.basename(__file__)[:-3]}.BestModelSelection" class')

    X_train: np.array
    y_train: np.array
    X_test: np.array
    y_test: np.array
    cv_for_eval: int
    cv_for_hypertuning: int

    # shortlisted models
    xgb_clf = XGBClassifier(objective='binary:logistic')
    random_clf = RandomForestClassifier(random_state=42)

    # booleans
    use_accuracy = None

    def _choose_best_candidate_model(self) -> str:
        try:
            lg.info("Quest for choosing the `best candidate model` begins..")
            lg.info(
                'Candidate models: "SVC", "RandomForestClassifier", "XGBClassifier"')

            ####################### Evaluation using Cross-Validation #########################
            cross_eval_test_scores = []
            
            if len(np.unique(self.y_train)) == 0:  # then can't use `roc_auc_score`, Will go ahead with `accuracy`
                self.use_accuracy = True

                # ******************** Evaluating the Random Forest Clf ***********************
                lg.info(
                    'Evaluating the "RandomForestClassifier" using cross-validation..')

                # Cross-Validation Accuracies
                random_clf_accs = cross_val_score(
                    self.random_clf, self.X_train, self.y_train, cv=self.cv_for_eval, verbose=self.cv_for_eval)
                lg.info(
                    f'"RandomForestClassifier" CV Accuracies: {random_clf_accs}')

                # Mean Accuracy
                random_clf_mean_acc = random_clf_accs.mean()
                lg.info("..cross-validation finished successfully!")
                lg.info(
                    f'"RandomForestClassifier" mean Accuracy: {random_clf_mean_acc}')

                # Performance on Test set
                lg.info(
                    "Evaluating the performance of `RandomForestClasssifier` on test set..")
                y_test_pred = cross_val_predict(
                    self.random_clf, self.X_test, self.y_test, cv=self.cv_for_eval)
                test_acc = accuracy_score(self.y_test, y_test_pred)
                lg.info(
                    f"RandomForestClassifier's Accuracy on test set: {test_acc}")
                cross_eval_test_scores.append(
                    (test_acc, "RandomForestClassifier"))

                # ******************* Evaluating the XGB Clf **********************************
                lg.info('Evaluating the "XGBClassifier" using cross-validation..')

                # Cross-Validation Accuracies
                xgb_clf_accs = cross_val_score(
                    self.xgb_clf, self.X_train, self.y_train, cv=self.cv_for_eval, verbose=self.cv_for_eval)
                lg.info("..cross-validation finished successfully!")
                lg.info(f'"XGBClassifier" CV Accuracies: {xgb_clf_accs}')

                # Mean Accuracy
                xgb_clf_mean_acc = xgb_clf_accs.mean()
                lg.info(f'"XGBClassifier" mean Accuracy: {xgb_clf_mean_acc}')

                # Performance on Test set
                lg.info(
                    'Evaluating the performance of "XGBClassifier" on test set..')
                y_test_pred = cross_val_predict(
                    self.xgb_clf, self.X_test, self.y_test, cv=self.cv_for_eval)
                test_acc = accuracy_score(self.y_test, y_test_pred)
                lg.info(f"XGBClassifier's Accuracy on test set: {test_acc}")
                cross_eval_test_scores.append(
                    (test_acc, "XGBClassifier"))

                ##################### Returning the best performing Model #########################
                cross_eval_test_scores.sort()
                # model with largest cross-validated Accuracy
                best_candidate = cross_eval_test_scores[-1]
                lg.info(
                    f'best performing classifier turns outta be "{best_candidate[1]}" with Accuracy={best_candidate[0]}!')

                return best_candidate[1]

            else:  # gonna go ahead with `roc_auc_score` as performance metric
                self.use_accuracy = False

                # ******************** Evaluating the Random Forest Clf ***********************
                lg.info(
                    'Evaluating the "RandomForestClassifier" using cross-validation..')
                # Cross-Validation AUCs
                random_clf_aucs = cross_val_score(
                    self.random_clf, self.X_train, self.y_train, scoring='roc_auc', cv=self.cv_for_eval, verbose=3)
                lg.info(f'"RandomForestClassifier" CV AUCs: {random_clf_aucs}')
                # Mean AUC
                random_clf_mean_auc = random_clf_aucs.mean()
                lg.info("..cross-validation finished successfully!")
                lg.info(
                    f'"RandomForestClassifier" mean AUC: {random_clf_mean_auc}')

                # AUC on the Test set
                lg.info(
                    "Evaluating the performance of `RandomForestClasssifier` on test set..")
                y_test_pred = cross_val_predict(
                    self.random_clf, self.X_test, self.y_test, cv=self.cv_for_eval)
                test_auc = roc_auc_score(self.y_test, y_test_pred)
                lg.info(
                    f"RandomForestClassifier's AUC on test set: {test_auc}")
                cross_eval_test_scores.append(
                    (test_auc, "RandomForestClassifier"))

                # ******************* Evaluating the XGB Clf **********************************
                lg.info('Evaluating the "XGBClassifier" using cross-validation..')
                # Cross-Validation AUCs
                xgb_clf_aucs = cross_val_score(
                    self.xgb_clf, self.X_train, self.y_train, scoring='roc_auc', cv=self.cv_for_eval, verbose=3)
                lg.info("..cross-validation finished successfully!")
                lg.info(f'"XGBClassifier" CV AUCs: {xgb_clf_aucs}')
                # Mean AUC
                xgb_clf_mean_auc = xgb_clf_aucs.mean()
                lg.info(f'"XGBClassifier" mean AUC: {xgb_clf_mean_auc}')

                # AUC on the Test set
                lg.info(
                    'Evaluating the performance of "XGBClassifier" on test set..')
                y_test_pred = cross_val_predict(
                    self.xgb_clf, self.X_test, self.y_test, cv=self.cv_for_eval)
                test_auc = roc_auc_score(self.y_test, y_test_pred)
                lg.info(f"XGBClassifier's AUC on test set: {test_auc}")
                cross_eval_test_scores.append(
                    (test_auc, "XGBClassifier"))

                ##################### Returning the best performing Model #########################
                cross_eval_test_scores.sort()
                # model with largest cross-validated AUC
                best_candidate = cross_eval_test_scores[-1]

                lg.info(
                    f'best performing classifier turned outta be "{best_candidate[1]}" with ROC-AUC score={best_candidate[0]}!')
                return best_candidate[1]

        except Exception as e:
            lg.exception(e)
            raise e

    def _hypertune_XGBClassifier(self) -> XGBClassifier:
        try:
            lg.info('Hypertuning the "XGBClassifier" using GridSearchCV..')

            # shortlist params for GridSearchCV
            grid_params = {
                'learning_rate': [.001, .01, .1, .5],
                'max_depth': [3, 5, 6],
                'n_estimators': [100, 200, 300, 500]
            }

            if self.use_accuracy:
                # commence GridSearchCV
                self.grid_search = GridSearchCV(
                    self.xgb_clf, param_grid=grid_params, cv=self.cv_for_hypertuning, verbose=3)
                self.grid_search.fit(self.X_train, self.y_train)
                lg.info(
                    f"..GridSearchCV finished successfully with best_score_={self.grid_search.best_score_}!")
                lg.info(
                    f"XGBClassifier's best params: {self.grid_search.best_params_}")

                lg.info('Returning the "XGBClassifier" trained using best_params_..')
                return self.grid_search.best_estimator_
            else:  # use the AUC as performance metric
                # commence GridSearchCV
                self.grid_search = GridSearchCV(
                    self.xgb_clf, param_grid=grid_params, cv=self.cv_for_hypertuning, verbose=3, scoring='roc_auc')
                self.grid_search.fit(self.X_train, self.y_train)
                lg.info(
                    f"..GridSearchCV finished successfully with best_score_={self.grid_search.best_score_}!")
                lg.info(
                    f"XGBClassifier's best params: {self.grid_search.best_params_}")

                lg.info('Returning the "XGBClassifier" trained using best_params_..')
                return self.grid_search.best_estimator_
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def _hypertune_RandomForestClassifier(self) -> RandomForestClassifier:
        try:
            lg.info('Hypertuning the `"RandomForestClassifier" using GridSearchCV..')

            # shortlist params for GridSearchCV
            grid_params = {
                "n_estimators": [300, 500, 1000],
                "criterion": ['gini', 'entropy'],
                "max_depth": [3, 5, 6],
                "max_features": ['auto', 'log2']
            }

            if self.use_accuracy:
                # commmence GridSearchCV
                self.grid_search = GridSearchCV(
                    self.random_clf, param_grid=grid_params, cv=self.cv_for_hypertuning, verbose=3)
                self.grid_search.fit(self.X_train, self.y_train)
                lg.info(
                    f"..GridSearchCV finished successfully with best_score_={self.grid_search.best_score_}!")
                lg.info(
                    f"RandomForestClassifier's best params: {self.grid_search.best_params_}")

                lg.info(
                    'Returning the "RandomForestClassifier" trained using best_params_..')
                return self.grid_search.best_estimator_
            else:  # use the AUC as performance metric
                # commmence GridSearchCV
                self.grid_search = GridSearchCV(
                    self.random_clf, param_grid=grid_params, cv=self.cv_for_hypertuning, verbose=3)
                self.grid_search.fit(self.X_train, self.y_train)
                lg.info(
                    f"..GridSearchCV finished successfully with best_score_={self.grid_search.best_score_}!")
                lg.info(
                    f"RandomForestClassifier's best params: {self.grid_search.best_params_}")

                lg.info(
                    'Returning the "RandomForestClassifier" trained using best_params_..')
                return self.grid_search.best_estimator_
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def get_best_model(self):
        try:
            lg.info(
                f'Entered the `get_best_model` of the "{os.path.basename(__file__)[:-3]}.BestModelSelection" class')

            ################################## Fetch Best Model #################################################
            lg.info("Fetching the best model, evaluated using cross-validation..")
            best_candidate = self._choose_best_candidate_model()
            lg.info(f'Best Model we\'ve got: "{best_candidate}"')

            ################################# Finetune Best Model ###############################################
            lg.info(f'finetuning the best model `{best_candidate}`..')
            if best_candidate == "RandomForestClassifier":
                best_mod = self._hypertune_RandomForestClassifier()
            else:
                best_mod = self._hypertune_XGBClassifier()
            lg.info("..finetuned the best model successfully!")

            lg.info(f'returning the best model "{best_candidate}"..')
            return best_candidate, best_mod
            ...
        except Exception as e:
            lg.exception(e)
            raise e
