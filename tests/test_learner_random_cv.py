import os

import numpy as np
import optuna
import pytest

from analyze.result_analyzer import ResultAnalyzer
from reshufflebench.algorithms import CatBoost, FunnelMLP, LogReg, XGBoost
from reshufflebench.algorithms.classifier import Classifier
from reshufflebench.learner import LearnerRandomCV
from reshufflebench.metrics import compute_metric


@pytest.fixture(params=["catboost", "funnel_mlp", "logreg", "xgboost"])
def classifier(request: pytest.FixtureRequest) -> Classifier:
    if request.param == "catboost":
        return CatBoost(seed=2409)
    elif request.param == "funnel_mlp":
        return FunnelMLP(seed=2409)
    elif request.param == "logreg":
        return LogReg(seed=2409)
    elif request.param == "xgboost":
        return XGBoost(seed=2409)


@pytest.fixture(params=[True, False])
def reshuffle(request: pytest.FixtureRequest) -> bool:
    return request.param


def test_learner_random_cv(classifier: Classifier, reshuffle: bool) -> None:
    learner_random = LearnerRandomCV(
        classifier=classifier,
        data_id=1169,
        train_valid_size=100,
        reshuffle=reshuffle,
        n_splits=5,
        n_repeats=2,
        test_size=200,
        add_valid_size=100,
        n_trials=2,
        seed=2409,
    )
    learner_random.run()

    analyzer = ResultAnalyzer(
        learner_random.results_path,
        seed=learner_random.seed,
        n_repeats=learner_random.n_repeats,
    )
    analyzer.load(additional_file_types="all")

    results = analyzer.results
    assert len(results) == 2

    cv_metrics = ["accuracies", "balanced_accuracies", "loglosses", "aucs"]

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
    columns_cv = [
        f"user_attrs_{metric}_{predict_set}"
        for metric in cv_metrics
        for predict_set in ["train", "valid", "add_valid", "test"]
    ]
    columns_cv_ensemble = [
        f"user_attrs_{metric}_test_ensemble" for metric in learner_random.metrics
    ]

    cv_metric_to_metric = {
        "accuracies": "accuracy",
        "balanced_accuracies": "balanced_accuracy",
        "loglosses": "logloss",
        "aucs": "auc",
    }
    metric_to_cv_metric = {v: k for k, v in cv_metric_to_metric.items()}

    assert all([column in results.columns for column in columns])
    assert all([column in results.columns for column in columns_cv])
    assert all([column in results.columns for column in columns_cv_ensemble])

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
    map_predict_sets_to_refit = {
        # "y_pred_train_proba_hist": False,
        "y_pred_valid_proba_hist": False,
        "y_pred_add_valid_proba_hist": False,
        "y_pred_test_proba_hist": False,
        # "y_pred_valid_train_proba_hist": True,
        "y_pred_test_proba_retrained_hist": True,
    }
    map_predict_sets_to_y_vary = {
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
                assert all([any(y[0][i] != y[1][i]) for i in range(len(y[0]))])

        for trial in range(len(results)):
            if map_predict_sets_to_y_vary[predict_set]:
                if reshuffle:
                    y_tmp = y[trial]
                else:
                    y_tmp = y[0]
            else:
                y_tmp = y
            y_pred_proba_tmp = y_pred_proba[trial]
            if map_predict_sets_to_y_vary[predict_set]:
                assert (
                    len(y_pred_proba_tmp)
                    == learner_random.n_repeats * learner_random.n_splits
                )
            if predict_set == "y_pred_train_proba_hist":
                assert y_tmp[0].shape[0] == learner_random.train_size
                assert y_tmp[0].shape[0] == int(0.8 * 100)
                assert y_tmp[0].shape[0] == y_pred_proba_tmp[0].shape[0]
            elif predict_set == "y_pred_valid_proba_hist":
                assert y_tmp[0].shape[0] == learner_random.valid_size
                assert y_tmp[0].shape[0] == int(0.2 * 100)
                assert y_tmp[0].shape[0] == y_pred_proba_tmp[0].shape[0]
            elif predict_set == "y_pred_add_valid_proba_hist":
                assert y_tmp[0].shape[0] == learner_random.valid_size + int(
                    learner_random.add_valid_size / learner_random.n_splits
                )  # add_valid_size is the size of the total additional validation set, not the size of the validation set for each split
                assert y_tmp[0].shape[0] == int(0.2 * 100) + (100 / 5)
                assert y_tmp[0].shape[0] == y_pred_proba_tmp[0].shape[0]
            elif predict_set == "y_pred_test_proba_hist":
                assert y_tmp.shape[0] == learner_random.test_size
                assert y_tmp.shape[0] == 200
                assert y_tmp.shape[0] == y_pred_proba_tmp[0].shape[0]
            elif predict_set == "y_pred_valid_train_proba_hist":
                assert (
                    y_tmp.shape[0]
                    == learner_random.train_size + learner_random.valid_size
                )
                assert y_tmp.shape[0] == 100
                assert y_tmp.shape[0] == y_pred_proba_tmp.shape[0]
            elif predict_set == "y_pred_test_proba_retrained_hist":
                assert y_tmp.shape[0] == learner_random.test_size
                assert y_tmp.shape[0] == 200
                assert y_tmp.shape[0] == y_pred_proba_tmp.shape[0]
            for metric in learner_random.metrics:
                if map_predict_sets_to_refit[predict_set]:
                    y_pred_tmp = np.argmax(y_pred_proba_tmp, axis=1)
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
                else:
                    recalculated_metric = []
                    for i in range(len(y_pred_proba_tmp)):
                        y_tmp_i = (
                            y_tmp[i]
                            if map_predict_sets_to_y_vary[predict_set]
                            else y_tmp
                        )
                        y_pred_tmp = np.argmax(y_pred_proba_tmp[i], axis=1)
                        recalculated_metric.append(
                            compute_metric(
                                y_tmp_i,
                                y_pred=y_pred_tmp,
                                y_pred_proba=y_pred_proba_tmp[i],
                                metric=metric,
                                labels=learner_random.labels,
                                multiclass=learner_random.multiclass,
                            )
                        )
                    assert all(
                        abs(
                            np.array(recalculated_metric)
                            - np.array(
                                eval(
                                    results[
                                        f"user_attrs_{metric_to_cv_metric[metric]}_{map_predict_sets_to_orig_metrics[predict_set]}"
                                    ][trial]
                                )
                            )
                        )
                        < 1e-12
                    )
                    assert (
                        abs(
                            np.mean(recalculated_metric)
                            - results[
                                f"user_attrs_{metric}_{map_predict_sets_to_orig_metrics[predict_set]}"
                            ][trial]
                        )
                        < 1e-12
                    )
                    columns_checked.append(
                        f"user_attrs_{metric_to_cv_metric[metric]}_{map_predict_sets_to_orig_metrics[predict_set]}"
                    )
                    columns_checked.append(
                        f"user_attrs_{metric}_{map_predict_sets_to_orig_metrics[predict_set]}"
                    )

    # test_ensemble
    y = getattr(analyzer, "y_test")
    y_pred_proba = getattr(analyzer, "y_pred_test_proba_hist")
    for trial in range(len(results)):
        y_tmp = y
        y_pred_proba_tmp = y_pred_proba[trial]
        assert (
            len(y_pred_proba_tmp) == learner_random.n_repeats * learner_random.n_splits
        )
        assert y_tmp.shape[0] == 200
        assert y_tmp.shape[0] == y_pred_proba_tmp[0].shape[0]
        y_pred_proba_tmp_ensemble = np.mean(y_pred_proba_tmp, axis=0)
        y_pred_tmp_ensemble = np.argmax(y_pred_proba_tmp_ensemble, axis=1)
        for metric in learner_random.metrics:
            recalculated_metric = compute_metric(
                y_tmp,
                y_pred=y_pred_tmp_ensemble,
                y_pred_proba=y_pred_proba_tmp_ensemble,
                metric=metric,
                labels=learner_random.labels,
                multiclass=learner_random.multiclass,
            )
            assert (
                abs(
                    recalculated_metric
                    - results[f"user_attrs_{metric}_test_ensemble"][trial]
                )
                < 1e-12
            )
            columns_checked.append(f"user_attrs_{metric}_test_ensemble")

    if learner_random.classifier.classifier_id in ["catboost", "xgboost"]:
        assert all(
            results["user_attrs_actual_iterations"].apply(lambda x: len(eval(x))).values
            == learner_random.n_repeats * learner_random.n_splits
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

    # check cv_splits
    y_valid_train = getattr(analyzer, "y_valid_train")

    # cv_splits_hist_train = getattr(analyzer, "cv_splits_hist_train")
    # y_train_hist = getattr(analyzer, "y_train_hist")
    # assert len(cv_splits_hist_train) == 2
    # for i in range(len(cv_splits_hist_train)):
    #    assert (
    #        len(cv_splits_hist_train[i])
    #        == learner_random.n_repeats * learner_random.n_splits
    #    )
    #    for j in range(len(cv_splits_hist_train[i])):
    #        assert cv_splits_hist_train[i][j].shape[0] == learner_random.train_size
    #        train_indices = cv_splits_hist_train[i][j]
    #        assert all(y_valid_train[train_indices] == y_train_hist[i][j])

    cv_splits_hist_valid = getattr(analyzer, "cv_splits_hist_valid")
    y_valid_hist = getattr(analyzer, "y_valid_hist")
    if reshuffle:
        assert len(cv_splits_hist_valid) == 2
    else:
        assert len(cv_splits_hist_valid) == 1
    for i in range(len(cv_splits_hist_valid)):
        assert (
            len(cv_splits_hist_valid[i])
            == learner_random.n_repeats * learner_random.n_splits
        )
        for j in range(len(cv_splits_hist_valid[i])):
            assert cv_splits_hist_valid[i][j].shape[0] == learner_random.valid_size
            valid_indices = cv_splits_hist_valid[i][j]
            assert all(y_valid_train[valid_indices] == y_valid_hist[i][j])

    # Note: y_add_valid is the additional validation set that was then partitioned into n_splits parts and added to the validation set for each split
    #       y_add_valid_hist always contains the concatenation of the original validation set and the additional validation set for each split
    #       similarly cv_splits_add_valid_hist_valid always contains the concatenation of the indices of the original validation set and the additional validation set for each split
    y_add_valid_use = getattr(analyzer, "y_add_valid_use")
    cv_splits_add_valid_hist_valid = getattr(analyzer, "cv_splits_add_valid_hist_valid")
    y_add_valid_hist = getattr(analyzer, "y_add_valid_hist")
    if reshuffle:
        assert len(cv_splits_add_valid_hist_valid) == 2
    else:
        assert len(cv_splits_add_valid_hist_valid) == 1
    for i in range(len(cv_splits_add_valid_hist_valid)):
        assert (
            len(cv_splits_add_valid_hist_valid[i])
            == learner_random.n_repeats * learner_random.n_splits
        )
        for j in range(len(cv_splits_add_valid_hist_valid[i])):
            assert cv_splits_add_valid_hist_valid[i][j].shape[
                0
            ] == learner_random.valid_size + int(
                learner_random.add_valid_size / learner_random.n_splits
            )
            add_valid_indices = cv_splits_add_valid_hist_valid[i][j][
                learner_random.valid_size :
            ]
            assert all(
                y_add_valid_use[add_valid_indices]
                == y_add_valid_hist[i][j][learner_random.valid_size :]
            )

    # remove all files
    for file in os.listdir(learner_random.results_path):
        os.remove(os.path.join(learner_random.results_path, file))

    # remove results_path
    os.rmdir(learner_random.results_path)

    # run again with n_trials=1
    learner_random = LearnerRandomCV(
        classifier=classifier,
        data_id=1169,
        train_valid_size=100,
        reshuffle=True,
        n_splits=5,
        n_repeats=2,
        test_size=200,
        add_valid_size=100,
        n_trials=1,
        seed=2409,
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
