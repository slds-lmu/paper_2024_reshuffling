import os

import numpy as np
import optuna
import pytest

from analyze.result_analyzer import ResultAnalyzer
from reshufflebench.algorithms import CatBoost, FunnelMLP, LogReg, XGBoost
from reshufflebench.algorithms.classifier import Classifier
from reshufflebench.learner import LearnerRandomHoldout
from reshufflebench.metrics import compute_metric


@pytest.fixture(params=["catboost", "funnel_mlp", "logreg", "xgboost"])
def classifier(request: pytest.FixtureRequest) -> Classifier:
    if request.param == "catboost":
        return CatBoost(seed=2906)
    elif request.param == "funnel_mlp":
        return FunnelMLP(seed=2906)
    elif request.param == "logreg":
        return LogReg(seed=2906)
    elif request.param == "xgboost":
        return XGBoost(seed=2906)


@pytest.fixture(params=[True, False])
def reshuffle(request: pytest.FixtureRequest) -> bool:
    return request.param


def test_learner_random_holdout(classifier: Classifier, reshuffle: bool) -> None:
    learner_random = LearnerRandomHoldout(
        classifier=classifier,
        data_id=1169,
        train_valid_size=100,
        reshuffle=reshuffle,
        valid_frac=0.2,
        test_size=200,
        add_valid_size=100,
        n_trials=2,
        seed=2906,
    )
    learner_random.run()

    analyzer = ResultAnalyzer(learner_random.results_path, seed=learner_random.seed)
    analyzer.load(additional_file_types="all")

    results = analyzer.results
    assert len(results) == 2

    columns = [
        f"user_attrs_{metric}_{predict_set}"
        for metric in learner_random.metrics
        for predict_set in [
            "train",
            "valid",
            "add_valid",
            "test",
            "valid_train",
            "test_retrained",
        ]
    ]
    assert all([column in results.columns for column in columns])

    predict_sets = [
        # "y_pred_train_proba_hist",
        "y_pred_valid_proba_hist",
        "y_pred_add_valid_proba_hist",
        "y_pred_test_proba_hist",
        # "y_pred_valid_train_proba_hist",
        "y_pred_test_proba_retrained_hist",
    ]
    map_predict_sets_to_y_sets = {
        # "y_pred_train_proba_hist": "y_train_hist",
        "y_pred_valid_proba_hist": "y_valid_hist",
        "y_pred_add_valid_proba_hist": "y_add_valid_hist",
        "y_pred_test_proba_hist": "y_test",
        # "y_pred_valid_train_proba_hist": "y_valid_train",
        "y_pred_test_proba_retrained_hist": "y_test",
    }
    map_predict_sets_to_orig_metrics = {
        # "y_pred_train_proba_hist": "train",
        "y_pred_valid_proba_hist": "valid",
        "y_pred_add_valid_proba_hist": "add_valid",
        "y_pred_test_proba_hist": "test",
        # "y_pred_valid_train_proba_hist": "valid_train",
        "y_pred_test_proba_retrained_hist": "test_retrained",
    }
    map_predict_sets_to_hist = {
        # "y_pred_train_proba_hist": True,
        "y_pred_valid_proba_hist": True,
        "y_pred_add_valid_proba_hist": True,
        "y_pred_test_proba_hist": False,
        # "y_pred_valid_train_proba_hist": False,
        "y_pred_test_proba_retrained_hist": False,
    }

    columns_checked = []

    for predict_set in predict_sets:
        y = getattr(analyzer, map_predict_sets_to_y_sets[predict_set])
        y_pred_proba = getattr(analyzer, predict_set)

        if (
            map_predict_sets_to_y_sets[predict_set] == "y_train_hist"
            or map_predict_sets_to_y_sets[predict_set] == "y_valid_hist"
            or map_predict_sets_to_y_sets[predict_set] == "y_add_valid_hist"
        ):
            if reshuffle:
                assert any(y[0] != y[1])

        for trial in range(len(results)):
            if map_predict_sets_to_hist[predict_set]:
                if reshuffle:
                    y_tmp = y[trial]
                else:
                    y_tmp = y[0]
            else:
                y_tmp = y
            y_pred_proba_tmp = y_pred_proba[trial]
            y_pred_tmp = np.argmax(y_pred_proba_tmp, axis=1)
            if predict_set == "y_pred_train_proba_hist":
                assert y_tmp.shape[0] == learner_random.train_size
                assert y_tmp.shape[0] == int(0.8 * 100)
            elif predict_set == "y_pred_valid_proba_hist":
                assert y_tmp.shape[0] == learner_random.valid_size
                assert y_tmp.shape[0] == int(0.2 * 100)
            elif predict_set == "y_pred_add_valid_proba_hist":
                assert (
                    y_tmp.shape[0]
                    == learner_random.valid_size + learner_random.add_valid_size
                )
                assert y_tmp.shape[0] == int(0.2 * 100) + 100
            elif predict_set == "y_pred_test_proba_hist":
                assert y_tmp.shape[0] == learner_random.test_size
                assert y_tmp.shape[0] == 200
            elif predict_set == "y_pred_valid_train_proba_hist":
                assert (
                    y_tmp.shape[0]
                    == learner_random.train_size + learner_random.valid_size
                )
                assert y_tmp.shape[0] == 100
            elif predict_set == "y_pred_test_proba_retrained_hist":
                assert y_tmp.shape[0] == learner_random.test_size
                assert y_tmp.shape[0] == 200
            assert y_tmp.shape[0] == y_pred_proba_tmp.shape[0]
            assert y_tmp.shape[0] == y_pred_tmp.shape[0]
            for metric in learner_random.metrics:
                recalculated_metric = compute_metric(
                    y_tmp,
                    y_pred=y_pred_tmp,
                    y_pred_proba=y_pred_proba_tmp,
                    metric=metric,
                    labels=learner_random.labels,
                    multiclass=learner_random.multiclass,
                )
                assert (
                    abs(
                        recalculated_metric
                        - results[
                            f"user_attrs_{metric}_{map_predict_sets_to_orig_metrics[predict_set]}"
                        ][trial]
                    )
                    < 1e-12
                )
                columns_checked.append(
                    f"user_attrs_{metric}_{map_predict_sets_to_orig_metrics[predict_set]}"
                )

    if learner_random.classifier.classifier_id in ["catboost", "xgboost"]:
        assert all(
            results["user_attrs_actual_iterations"].apply(lambda x: len(eval(x))).values
            == 1
        )
        columns_checked.append("user_attrs_actual_iterations")

    remaining_columns = [
        column for column in results.columns if column not in columns_checked
    ]
    remaining_columns = [
        column for column in remaining_columns if "_valid_train" not in column
    ]
    remaining_columns = [
        column for column in remaining_columns if "_train" not in column
    ]
    remaining_columns = [
        column
        for column in remaining_columns
        if "user_attrs_error_on_fit" not in column
    ]

    assert all(["user_attrs" not in column for column in remaining_columns])

    # remove all files
    for file in os.listdir(learner_random.results_path):
        os.remove(os.path.join(learner_random.results_path, file))

    # remove results_path
    os.rmdir(learner_random.results_path)

    # run again with n_trials = 1
    learner_random = LearnerRandomHoldout(
        classifier=classifier,
        data_id=1169,
        train_valid_size=100,
        reshuffle=True,
        valid_frac=0.2,
        test_size=200,
        add_valid_size=100,
        n_trials=1,
        seed=2906,
    )
    learner_random.run()

    study = optuna.load_study(
        study_name=learner_random.study_name, storage=learner_random.storage
    )
    results_rerun = study.trials_dataframe()

    assert len(results_rerun) == 1
    assert results["value"].values[0] == results_rerun["value"].values[0]

    # remove all files
    for file in os.listdir(learner_random.results_path):
        os.remove(os.path.join(learner_random.results_path, file))

    # remove results_path
    os.rmdir(learner_random.results_path)
