import argparse
import concurrent.futures
import json
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import tqdm
from scipy.stats import kendalltau

from analyze.post_selector import PostSelector, PostSelectorNaive
from analyze.utils import compute_overtuning
from reshufflebench.metrics import compute_metric
from reshufflebench.utils import (
    NumpyArrayEncoder,
    check_y_predict_proba,
    int_or_none,
    load_list_of_1d_arrays,
    load_list_of_list_of_1d_arrays,
    load_list_of_list_of_pd_arrays,
    load_list_of_pd_arrays,
    load_single_array,
    str2bool,
)


class ResultAnalyzer(object):
    """
    Class to analyze the results of a study.
    """

    def __init__(
        self,
        results_path: str,
        seed: int,
        n_repeats: Optional[int] = None,
        check_files: bool = True,
    ):
        self.results_path = os.path.abspath(results_path)
        self.seed = seed
        self.file_name = f"study_seed_{self.seed}"
        with open(
            os.path.join(self.results_path, f"{self.file_name}_params.json"), "r"
        ) as f:
            self.params = json.load(f)
        if self.params["valid_type"] in ["cv", "repeatedholdout"]:
            self.params.update(
                {
                    "metric_to_cv_metric": {
                        v: k for k, v in self.params["cv_metric_to_metric"].items()
                    }
                }
            )

        if self.params["valid_type"] == "holdout" and n_repeats is not None:
            raise ValueError("n_repeats must be None for validation type holdout")

        if self.params["valid_type"] in ["cv", "repeatedholdout"]:
            if n_repeats is None:
                raise ValueError(
                    "n_repeats must not be None for validation type cv or repeatedholdout"
                )
            if n_repeats < 1 or n_repeats > self.params["n_repeats"]:
                raise ValueError(
                    f"n_repeats must be between 1 and {self.params['n_repeats']}"
                )

        self.n_repeats = n_repeats
        self.check_files = check_files

        resampling = (
            f"{self.params['valid_type']}_{str(self.params['valid_frac']).replace('.', '')}_{self.params['reshuffle']}"
            if self.params["valid_type"] == "holdout"
            else (
                f"{self.params['valid_type']}_{self.params['n_splits']}_{self.n_repeats}_{self.params['reshuffle']}"
                if self.params["valid_type"] == "cv"
                else f"{self.params['valid_type']}_{str(self.params['valid_frac']).replace('.', '')}_{self.n_repeats}_{self.params['reshuffle']}"
            )
        )
        self.params.update({"resampling": resampling})

        self.valid_type = (
            "holdout"
            if self.params["valid_type"] == "holdout"
            else (
                "repeatedholdout"
                if self.params["valid_type"] == "repeatedholdout"
                else (
                    "cv"
                    if self.params["valid_type"] == "cv" and self.n_repeats == 1
                    else "cv_repeated"
                )
            )
        )
        self.reshuffle = self.params["reshuffle"]

        self.adjust_results_for_n_repeats_needed = (
            self.valid_type in ["cv", "cv_repeated", "repeatedholdout"]
            and self.n_repeats != self.params["n_repeats"]
        )

        files = os.listdir(self.results_path)
        self.files = [file for file in files if str(self.seed) in file]
        if self.valid_type in ["cv", "cv_repeated", "repeatedholdout"]:
            self.expected_files_raw = [
                # "y_train_hist",
                "y_valid_hist",
                "y_add_valid_hist",
                # "y_pred_train_proba_hist",
                "y_pred_valid_proba_hist",
                "y_pred_add_valid_proba_hist",
                "y_pred_test_proba_hist",
                # "y_pred_valid_train_proba_hist",
                "y_pred_test_proba_retrained_hist",
                "y_valid_train",
                "y_add_valid_use",
                "y_test",
                # "cv_splits_hist_train",
                "cv_splits_hist_valid",
                "cv_splits_add_valid_hist_valid",
            ]
            self.expected_files = [
                f"{self.file_name}_{file}.parquet" for file in self.expected_files_raw
            ]
        else:
            self.expected_files_raw = [
                # "y_train_hist",
                "y_valid_hist",
                "y_add_valid_hist",
                # "y_pred_train_proba_hist",
                "y_pred_valid_proba_hist",
                "y_pred_add_valid_proba_hist",
                "y_pred_test_proba_hist",
                # "y_pred_valid_train_proba_hist",
                "y_pred_test_proba_retrained_hist",
                # "y_valid_train",
                "y_add_valid_use",
                "y_test",
            ]
            self.expected_files = [
                f"{self.file_name}_{file}.parquet" for file in self.expected_files_raw
            ]
        self.expected_files.append(f"{self.file_name}.db")
        self.expected_files.append(f"{self.file_name}_params.json")
        if self.check_files:
            if set(self.files) != set(self.expected_files):
                raise ValueError(f"Expected files does not match found files")

        self.study = None
        self.results = None

        self.y_train_hist = None
        self.y_valid_hist = None
        self.y_add_valid_hist = None
        self.y_pred_train_proba_hist = None
        self.y_pred_valid_proba_hist = None
        self.y_pred_add_valid_proba_hist = None
        self.y_pred_test_proba_hist = None
        self.y_pred_valid_train_proba_hist = None
        self.y_pred_test_proba_retrained_hist = None
        self.y_valid_train = None
        self.y_test = None

        self.cv_splits_hist_train = None
        self.cv_splits_hist_valid = None
        self.cv_splits_add_valid_hist_valid = None

        self.results_raw = {}
        self.results_post_selection = {}

        self.kendalls_tau_valid_test = {}
        self.kendalls_tau_train_valid = {}
        self.test_type_internal = None

    @property
    def test_type(self):
        """
        Return the test type based on the internal test type.
        """
        if self.test_type_internal is not None:
            if self.test_type_internal == "test_retrained":
                return "test_retrained"
            elif self.test_type_internal == "test":
                if self.valid_type == "holdout":
                    return "test"
                else:
                    return "test_ensemble"
        else:
            raise ValueError("test_type_internal is None")

    def adjust_results_for_n_repeats(self) -> None:
        """
        Adjust the results for the number of repeats in the case that the number of repeats is smaller than the number of repeats that was used for the study when using cv or repeatedholdout.
        This allows for post hoc simulation of n-times repeated cv or holdout with n ranging from 1 to the number of repeats that was used for the study.
        """
        if self.n_repeats == self.params["n_repeats"]:
            pass
        else:
            # subset to the indices that are needed
            if self.valid_type in ["cv", "cv_repeated"]:
                all_indices = list(
                    range(self.params["n_splits"] * self.params["n_repeats"])
                )
                indices = all_indices[: self.params["n_splits"] * self.n_repeats]
            elif self.valid_type == "repeatedholdout":
                all_indices = list(range(self.params["n_repeats"]))
                indices = all_indices[: self.n_repeats]
            field_to_subset = [
                "y_valid_hist",
                "y_add_valid_hist",
                "y_pred_valid_proba_hist",
                "y_pred_add_valid_proba_hist",
                "y_pred_test_proba_hist",
                "cv_splits_hist_valid",
                "cv_splits_add_valid_hist_valid",
            ]
            for field in field_to_subset:
                subset = []
                for i in range(len(getattr(self, field))):
                    subset.append([getattr(self, field)[i][j] for j in indices])
                setattr(self, field, subset)

            # correct results for each metric based on the relevant subset
            # e.g. user_attrs_accuracies_add_valid, user_attrs_accuracies_test, user_attrs_accuracies_train, user_attrs_accuracies_valid,
            #      user_attrs_accuracy_add_valid, user_attrs_accuracy_test, user_attrs_accuracy_test_ensemble, user_attrs_accuracy_train, user_attrs_accuracy_valid
            for metric in self.params["metrics"]:
                cv_metric = self.params["metric_to_cv_metric"][metric]
                columns_cv_metric = [
                    f"user_attrs_{cv_metric}_add_valid",
                    f"user_attrs_{cv_metric}_test",
                    f"user_attrs_{cv_metric}_train",
                    f"user_attrs_{cv_metric}_valid",
                ]
                for column in columns_cv_metric:
                    self.results[column] = (
                        self.results[column]
                        .apply(eval)
                        .apply(lambda x: [x[j] for j in indices])
                        .apply(json.dumps, cls=NumpyArrayEncoder)
                    )

                columns_metric = [
                    f"user_attrs_{metric}_add_valid",
                    f"user_attrs_{metric}_test",
                    f"user_attrs_{metric}_train",
                    f"user_attrs_{metric}_valid",
                ]

                for i, column in enumerate(columns_metric):
                    cv_values = self.results[columns_cv_metric[i]].apply(eval)
                    if not all(cv_values.apply(len) == len(indices)):
                        raise ValueError(f"Not all cv values have the expected length")
                    self.results[column] = cv_values.apply(np.mean)

                test_ensemble_column = f"user_attrs_{metric}_test_ensemble"
                test_ensemble_values = []
                for i in range(len(self.results)):
                    if len(self.y_pred_test_proba_hist[i]) != len(indices):
                        raise ValueError(f"y_pred_test_proba_hist has the wrong length")
                    predictions_proba_test_ensemble = np.mean(
                        self.y_pred_test_proba_hist[i], axis=0
                    )
                    row_sums = predictions_proba_test_ensemble.sum(
                        axis=1, keepdims=True
                    )
                    predictions_proba_test_ensemble = (
                        predictions_proba_test_ensemble / row_sums
                    )
                    check_y_predict_proba(predictions_proba_test_ensemble)
                    predictions_test_ensemble = np.argmax(
                        predictions_proba_test_ensemble, axis=1
                    )
                    test_ensemble_values.append(
                        compute_metric(
                            y_true=self.y_test,
                            y_pred=predictions_test_ensemble,
                            y_pred_proba=predictions_proba_test_ensemble,
                            metric=metric,
                            labels=self.params["labels"],
                            multiclass=self.params["multiclass"],
                        )
                    )
                self.results[test_ensemble_column] = test_ensemble_values

    def load(self, additional_file_types: str = "none") -> None:
        """
        Load the results.
        If additional_file_types is "none", load no additional files.
        If additional_file_types is "valid", load only additional files that are needed for validation.
        If additional_file_types is "all", load all additional files.
        Note that if self.adjust_results_for_n_repeats_needed is True, additional_file_types must be "all" and will be set to "all" automatically.
        If self.adjust_results_for_n_repeats_needed is True, the results will be adjusted for the number of repeats, see self.adjust_results_for_n_repeats.
        """
        if additional_file_types not in ["none", "valid", "all"]:
            raise ValueError(f"Unknown additional_file_types: {additional_file_types}")

        if self.adjust_results_for_n_repeats_needed:
            additional_file_types = "all"

        self.study = optuna.study.load_study(
            study_name=self.params["study_name"], storage=self.params["storage"]
        )
        self.results = self.study.trials_dataframe()
        if (
            len(self.results[self.results["state"] == "COMPLETE"])
            != self.params["n_trials"]
        ):
            raise ValueError(
                f"Number of trials does not match expected number of trials"
            )
        if not all(self.results["number"].values == range(self.params["n_trials"])):
            raise ValueError(f"Trial numbers are not in order")

        if additional_file_types in ["all", "valid"]:
            files = self.expected_files_raw
            if additional_file_types == "valid":
                files = [
                    file
                    for file in files
                    if file
                    in [
                        "y_valid_hist",
                        "y_pred_valid_proba_hist",
                        "cv_splits_hist_valid",
                    ]
                ]
            for file in files:
                try:
                    if self.valid_type == "holdout":
                        if file in [
                            "y_train_hist",
                            "y_valid_hist",
                            "y_add_valid_hist",
                        ]:
                            setattr(
                                self,
                                file,
                                load_list_of_1d_arrays(
                                    os.path.join(
                                        self.results_path,
                                        f"{self.file_name}_{file}.parquet",
                                    )
                                ),
                            )
                        elif file in ["y_valid_train", "y_add_valid_use", "y_test"]:
                            setattr(
                                self,
                                file,
                                load_single_array(
                                    os.path.join(
                                        self.results_path,
                                        f"{self.file_name}_{file}.parquet",
                                    )
                                ),
                            )
                        else:
                            setattr(
                                self,
                                file,
                                load_list_of_pd_arrays(
                                    os.path.join(
                                        self.results_path,
                                        f"{self.file_name}_{file}.parquet",
                                    )
                                ),
                            )
                    else:
                        if file in [
                            "y_train_hist",
                            "y_valid_hist",
                            "y_add_valid_hist",
                            "cv_splits_hist_train",
                            "cv_splits_hist_valid",
                            "cv_splits_add_valid_hist_valid",
                        ]:
                            setattr(
                                self,
                                file,
                                load_list_of_list_of_1d_arrays(
                                    os.path.join(
                                        self.results_path,
                                        f"{self.file_name}_{file}.parquet",
                                    )
                                ),
                            )
                        elif file in ["y_valid_train", "y_add_valid_use", "y_test"]:
                            setattr(
                                self,
                                file,
                                load_single_array(
                                    os.path.join(
                                        self.results_path,
                                        f"{self.file_name}_{file}.parquet",
                                    )
                                ),
                            )
                        elif file in [
                            "y_pred_valid_train_proba_hist",
                            "y_pred_test_proba_retrained_hist",
                        ]:
                            setattr(
                                self,
                                file,
                                load_list_of_pd_arrays(
                                    os.path.join(
                                        self.results_path,
                                        f"{self.file_name}_{file}.parquet",
                                    )
                                ),
                            )
                        else:
                            setattr(
                                self,
                                file,
                                load_list_of_list_of_pd_arrays(
                                    os.path.join(
                                        self.results_path,
                                        f"{self.file_name}_{file}.parquet",
                                    )
                                ),
                            )
                except FileNotFoundError:
                    warnings.warn(f"File {file} not found")

        if self.adjust_results_for_n_repeats_needed:
            self.adjust_results_for_n_repeats()

    def create_results_table_raw(self) -> None:
        """
        Create a results table with the raw results.
        """
        for metric in self.params["metrics"]:
            results = self.results.copy()
            params = [
                column for column in results.columns if column.startswith("params_")
            ]
            columns = [
                f"user_attrs_{metric}_train",
                f"user_attrs_{metric}_valid",
                f"user_attrs_{metric}_test",
                f"user_attrs_{metric}_valid_train",
                f"user_attrs_{metric}_test_retrained",
            ]
            if self.valid_type in ["cv", "cv_repeated", "repeatedholdout"]:
                columns.append(f"user_attrs_{metric}_test_ensemble")
                columns.append(
                    f"user_attrs_{self.params['metric_to_cv_metric'][metric]}_valid"
                )
            results = results[columns]
            results.columns = (
                ["train", "valid", "test", "valid_train", "test_retrained"]
                if self.valid_type == "holdout"
                else [
                    "train",
                    "valid",
                    "test",
                    "valid_train",
                    "test_retrained",
                    "test_ensemble",
                    "cv_valid",
                ]
            )
            if self.params["metrics_direction"][metric] == "maximize":
                for column in results.columns:
                    if column == "cv_valid":
                        results[column] = (
                            results[column].apply(eval).apply(lambda x: [-y for y in x])
                        )
                    else:
                        results[column] = results[column].apply(lambda x: -x)
            else:
                if "cv_valid" in results.columns:
                    results["cv_valid"] = results["cv_valid"].apply(eval)
            results["iteration"] = range(1, len(results) + 1)
            results["seed"] = self.seed
            results["classifier"] = self.params["classifier"]
            results["data_id"] = self.params["data_id"]
            results["train_valid_size"] = self.params["train_valid_size"]
            results["resampling"] = self.params["resampling"]
            results["metric"] = metric
            # add params
            for param in params:
                results[param] = self.results[param]
            self.results_raw[metric] = results

    def calculate_kendalls_tau_valid_test(self) -> None:
        """
        Calculate Kendall's Tau between valid and test_type.
        """
        for metric in self.params["metrics"]:
            kendalls_tau = []
            for local in [False, True]:
                # if local, select the top 20% based on valid and calculate Kendall's Tau between valid and test_type
                if local:
                    kendalls_tau_tmp = []
                    # for i in range(len(self.results_raw[metric])):
                    for i in range(9, len(self.results_raw[metric]), 10):
                        results = self.results_raw[metric].copy()
                        results = results.iloc[: i + 1]
                        q20 = results["valid"].quantile(0.2)
                        results = results[results["valid"] <= q20]
                        kendalls_tau_tmp.append(
                            kendalltau(
                                results["valid"].values, results[self.test_type].values
                            )[0]
                        )

                else:
                    kendalls_tau_tmp = []
                    # for i in range(len(self.results_raw[metric])):
                    for i in range(9, len(self.results_raw[metric]), 10):
                        results = self.results_raw[metric].copy()
                        results = results.iloc[: i + 1]
                        kendalls_tau_tmp.append(
                            kendalltau(
                                results["valid"].values, results[self.test_type].values
                            )[0]
                        )
                kendalls_tau.append(kendalls_tau_tmp)
            kendalls_tau_pd = pd.DataFrame(
                {
                    "kendalls_tau": kendalls_tau,
                    "test_type": self.test_type,
                    "local": [False, True],
                    "seed": self.seed,
                    "classifier": self.params["classifier"],
                    "data_id": self.params["data_id"],
                    "train_valid_size": self.params["train_valid_size"],
                    "resampling": self.params["resampling"],
                    "metric": metric,
                }
            )
            kendalls_tau_pd = kendalls_tau_pd.explode("kendalls_tau")
            kendalls_tau_pd["iteration"] = (
                # list(range(1, len(self.results_raw[metric]))) * 2
                list(range(9 + 1, len(self.results_raw[metric]) + 1, 10))
                * 2
            )
            self.kendalls_tau_valid_test.update({metric: kendalls_tau_pd})

    def create_results_table_post_selection(
        self, post_selector: PostSelector, **kwargs
    ) -> None:
        """
        Create a results table with the results after post selection.
        """
        results_post_selection = {"test": {}, "test_retrained": {}}
        for metric in self.params["metrics"]:
            post_selector.reset()
            results = self.results_raw[metric].copy()
            selected = []
            if len(results) != self.params["n_trials"] or len(results) != N_TRIALS:
                raise ValueError(
                    f"Number of trials does not match expected number of trials"
                )
            if post_selector.resolution_sparse:
                iterations = list(range(9, len(results), 10))
            else:
                iterations = list(range(len(results)))
            for i in tqdm.tqdm(iterations):
                selected.append(
                    post_selector.select(iteration=i, metric=metric, **kwargs)
                )
            results = results.iloc[selected]
            results.reset_index(inplace=True, drop=True)
            results["method"] = (
                f"{results['resampling'].values[0]}_post_{post_selector.id}"
            )
            results["orig_iteration"] = results["iteration"]
            results["iteration"] = [x + 1 for x in iterations]

            for test_type in ["test", "test_retrained"]:
                self.test_type_internal = test_type
                results_tmp = results.copy()
                results_tmp["overtuning"] = compute_overtuning(
                    list(results_tmp[self.test_type])
                )
                results_tmp = results_tmp.loc[
                    :,
                    [
                        "iteration",
                        "orig_iteration",
                        "valid",
                        self.test_type,
                        "overtuning",
                        "seed",
                        "classifier",
                        "data_id",
                        "train_valid_size",
                        "resampling",
                        "metric",
                        "method",
                    ],
                ]
                results_tmp.rename(columns={self.test_type: "test"}, inplace=True)
                # add additional iterationwise results
                if post_selector.additional_iterationwise_results:
                    for field in post_selector.additional_iterationwise_results:
                        value = getattr(post_selector, field)
                        if not isinstance(value, pd.DataFrame):
                            raise ValueError(
                                f"Additional iterationwise results must be a pd.DataFrame"
                            )
                        if "iteration" not in value.columns:
                            raise ValueError(
                                f"Additional iterationwise results must have a column 'iteration'"
                            )
                        # merge value into results by iteration
                        results_tmp = results_tmp.merge(
                            value, how="left", on="iteration", validate="1:1"
                        )
                if test_type == "test":
                    results_post_selection["test"][metric] = results_tmp
                elif test_type == "test_retrained":
                    results_post_selection["test_retrained"][metric] = results_tmp
        self.results_post_selection.update({post_selector.id: results_post_selection})


class ResultAnalyzerSimulateRepeatedHoldout(ResultAnalyzer):
    """
    Class to analyze the results of a study.
    Simulates repeated holdout (Monte Carlo CV) from the results of repeated CV.
    """

    def __init__(
        self, results_path: str, seed: int, n_repeats: int, check_files: bool = True
    ):
        super().__init__(
            results_path=results_path,
            seed=seed,
            n_repeats=n_repeats,
            check_files=check_files,
        )
        self.adjust_results_for_n_repeats_needed = True  # always adjust results for n_repeats due to simulation of repeated holdout

    # Note: could be faster if adjust is not done during loading but separately with the option to reset the results
    def adjust_results_for_n_repeats(self) -> None:
        """
        Simulates Monte Carlo CV from the results of repeated CV.
        """
        indices = [
            0 + i * self.params["n_splits"] for i in range(self.n_repeats)
        ]  # [0, 5, 10, 15, 20] etc.

        field_to_subset = [
            "y_valid_hist",
            "y_add_valid_hist",
            "y_pred_valid_proba_hist",
            "y_pred_add_valid_proba_hist",
            "y_pred_test_proba_hist",
            "cv_splits_hist_valid",
            "cv_splits_add_valid_hist_valid",
        ]
        for field in field_to_subset:
            subset = []
            for i in range(len(getattr(self, field))):
                subset.append([getattr(self, field)[i][j] for j in indices])
            setattr(self, field, subset)

        # correct results for each metric based on the relevant subset
        # e.g. user_attrs_accuracies_add_valid, user_attrs_accuracies_test, user_attrs_accuracies_train, user_attrs_accuracies_valid,
        #      user_attrs_accuracy_add_valid, user_attrs_accuracy_test, user_attrs_accuracy_test_ensemble, user_attrs_accuracy_train, user_attrs_accuracy_valid
        for metric in self.params["metrics"]:
            cv_metric = self.params["metric_to_cv_metric"][metric]
            columns_cv_metric = [
                f"user_attrs_{cv_metric}_add_valid",
                f"user_attrs_{cv_metric}_test",
                f"user_attrs_{cv_metric}_train",
                f"user_attrs_{cv_metric}_valid",
            ]
            for column in columns_cv_metric:
                self.results[column] = (
                    self.results[column]
                    .apply(eval)
                    .apply(lambda x: [x[j] for j in indices])
                    .apply(json.dumps, cls=NumpyArrayEncoder)
                )

            columns_metric = [
                f"user_attrs_{metric}_add_valid",
                f"user_attrs_{metric}_test",
                f"user_attrs_{metric}_train",
                f"user_attrs_{metric}_valid",
            ]

            for i, column in enumerate(columns_metric):
                cv_values = self.results[columns_cv_metric[i]].apply(eval)
                if not all(cv_values.apply(len) == len(indices)):
                    raise ValueError(f"Not all cv values have the expected length")
                self.results[column] = cv_values.apply(np.mean)

            test_ensemble_column = f"user_attrs_{metric}_test_ensemble"
            test_ensemble_values = []
            for i in range(len(self.results)):
                if len(self.y_pred_test_proba_hist[i]) != len(indices):
                    raise ValueError(f"y_pred_test_proba_hist has the wrong length")
                predictions_proba_test_ensemble = np.mean(
                    self.y_pred_test_proba_hist[i], axis=0
                )
                row_sums = predictions_proba_test_ensemble.sum(axis=1, keepdims=True)
                predictions_proba_test_ensemble = (
                    predictions_proba_test_ensemble / row_sums
                )
                check_y_predict_proba(predictions_proba_test_ensemble)
                predictions_test_ensemble = np.argmax(
                    predictions_proba_test_ensemble, axis=1
                )
                test_ensemble_values.append(
                    compute_metric(
                        y_true=self.y_test,
                        y_pred=predictions_test_ensemble,
                        y_pred_proba=predictions_proba_test_ensemble,
                        metric=metric,
                        labels=self.params["labels"],
                        multiclass=self.params["multiclass"],
                    )
                )
            self.results[test_ensemble_column] = test_ensemble_values


def analyze_results_basic(
    results_path: str,
    results_subfolder: str,
    seed: int,
    n_repeats: Optional[int] = None,
    check_files: bool = True,
) -> [pd.DataFrame, pd.DataFrame]:
    """
    Analyze the results of a study.
    Create a results table with the raw results and further calculate Kendall's Tau between valid and test_type for test type "test" and "test_retrained".
    """
    results_raw_tmp = {}
    kendalls_tau_valid_test_tmp = {"test": [], "test_retrained": []}
    analyzer = ResultAnalyzer(
        results_path=os.path.join(results_path, results_subfolder),
        seed=seed,
        n_repeats=n_repeats,
        check_files=check_files,
    )
    analyzer.load(additional_file_types="none")
    analyzer.create_results_table_raw()
    for metric in METRICS:
        results_raw_tmp[metric] = analyzer.results_raw[metric]

    for test_type in TEST_TYPES:
        analyzer.test_type_internal = test_type
        analyzer.calculate_kendalls_tau_valid_test()
        for metric in METRICS:
            kendalls_tau_valid_test_tmp[test_type].append(
                analyzer.kendalls_tau_valid_test[metric]
            )
    return (
        results_raw_tmp,
        kendalls_tau_valid_test_tmp,
    )


def analyze_results_post(
    results_path: str,
    results_subfolder: str,
    seed: int,
    type: str,
    n_repeats: Optional[int] = None,
    check_files: bool = True,
) -> Dict[str, List[pd.DataFrame]]:
    """
    Analyze the results of a study after post selection.
    For test type "test" and "test_retrained", create a results table with the results after post selection.
    """
    results_tmp = {"test": {}, "test_retrained": {}}
    additional_files = (
        "none" if type in ["post_naive"] else ("valid" if type in [] else "all")
    )
    analyzer = ResultAnalyzer(
        results_path=os.path.join(results_path, results_subfolder),
        seed=seed,
        n_repeats=n_repeats,
        check_files=check_files,
    )
    analyzer.load(additional_file_types=additional_files)
    analyzer.create_results_table_raw()
    post_selectors = {
        "post_naive": PostSelectorNaive(analyzer),
    }
    post_selector = post_selectors[type]

    if (
        analyzer.valid_type in post_selector.supported_valid_types
        and analyzer.reshuffle in post_selector.supported_reshufflings
    ):
        analyzer.create_results_table_post_selection(post_selector=post_selector)
        for test_type in TEST_TYPES:
            for metric in METRICS:
                results_tmp[test_type][metric] = analyzer.results_post_selection[
                    post_selector.id
                ][test_type][metric]

    return results_tmp


def analyze_results_post_naive_simulate_repeatedholdout(
    results_path: str,
    results_subfolder: str,
    seed: int,
    type: str,
    n_repeats: int,
    check_files: bool = True,
) -> Dict[str, List[pd.DataFrame]]:
    """
    Analyze the results of a study after post selection.
    Simulate repeated holdout (Monte Carlo CV) from the results of repeated CV.
    For test type "test" and "test_retrained", create a results table with the results after naive post selection.
    """
    if type != "post_naive_simulate_repeatedholdout":
        raise ValueError(f"Unsupported type: {type}")
    results_tmp = {"test": {}, "test_retrained": {}}
    additional_files = "valid"
    analyzer = ResultAnalyzerSimulateRepeatedHoldout(
        results_path=os.path.join(results_path, results_subfolder),
        seed=seed,
        n_repeats=n_repeats,
        check_files=check_files,
    )
    analyzer.load(additional_file_types=additional_files)
    analyzer.create_results_table_raw()
    post_selector = PostSelectorNaive(analyzer)
    post_selector.id += f"_simulate_repeatedholdout_{n_repeats}"

    if (
        analyzer.valid_type in post_selector.supported_valid_types
        and analyzer.reshuffle in post_selector.supported_reshufflings
    ):
        analyzer.create_results_table_post_selection(post_selector=post_selector)
        for test_type in TEST_TYPES:
            for metric in METRICS:
                results_tmp[test_type][metric] = analyzer.results_post_selection[
                    post_selector.id
                ][test_type][metric]

    return results_tmp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument(
        "--type",
        type=str,
        default="basic",
        choices=[
            "basic",
            "post_naive",
            "post_naive_simulate_repeatedholdout",
        ],
    )
    parser.add_argument("--results_subfolder", type=str)
    parser.add_argument("--n_repeats", type=int_or_none)
    parser.add_argument("--check_files", type=str2bool, default=True)
    args = parser.parse_args()

    results_path = os.path.abspath("../results")
    if "hebo" in args.results_subfolder:
        OPTIMIZER = "hebo"
        METRICS = ["auc"]
        N_TRIALS = 250
    else:
        OPTIMIZER = "random"
        METRICS = ["accuracy", "balanced_accuracy", "logloss", "auc"]
        if "tabpfn" in args.results_subfolder or "default" in args.results_subfolder:
            N_TRIALS = 1
        else:
            N_TRIALS = 500
    SEEDS = list(range(42, 52))
    TEST_TYPES = ["test", "test_retrained"]

    results_subfolders = os.listdir(results_path)
    if args.results_subfolder not in results_subfolders:
        raise ValueError(f"Unknown results_subfolder: {args.results_subfolder}")
    if not os.path.exists(os.path.abspath("../csvs/raw")):
        os.makedirs(os.path.abspath("../csvs/raw"))

    partial_file_name = args.results_subfolder
    if args.n_repeats is not None:
        if "cv" in partial_file_name:
            resampling_abbreviation = [
                string for string in partial_file_name.split("_") if ("cv" in string)
            ][0]
        elif "repeatedholdout" in partial_file_name:
            splits = partial_file_name.split("_")
            resampling_abbreviation = [s for s in splits if "repeatedholdout" in s][0]
            resampling_abbreviation += (
                "_" + splits[splits.index(resampling_abbreviation) + 1]
            )
        partial_file_name = partial_file_name.replace(
            resampling_abbreviation, f"{resampling_abbreviation}ur{args.n_repeats}"
        )

    if args.type == "basic":
        results_raw = {"raw": []}
        kendalls_tau_valid_test = {"test": [], "test_retrained": []}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            tasks = []
            for seed in SEEDS:
                tasks.append(
                    executor.submit(
                        analyze_results_basic,
                        results_path,
                        args.results_subfolder,
                        seed,
                        args.n_repeats,
                        args.check_files,
                    )
                )

            for future in concurrent.futures.as_completed(tasks):
                try:
                    (
                        results_raw_tmp,
                        kendalls_tau_valid_test_tmp,
                    ) = future.result()
                    if results_raw_tmp:
                        results_raw["raw"].append(pd.concat(results_raw_tmp))
                    for test_type in TEST_TYPES:
                        if kendalls_tau_valid_test_tmp[test_type]:
                            kendalls_tau_valid_test[test_type].append(
                                pd.concat(kendalls_tau_valid_test_tmp[test_type])
                            )
                except Exception as e:
                    print(f"Exception occurred: {e}")

        if results_raw["raw"]:
            results_raw_tmp = pd.concat(results_raw["raw"])
            results_raw_tmp["optimizer"] = OPTIMIZER
            results_raw_tmp.to_csv(
                f"../csvs/raw/{partial_file_name}_raw.csv",
                index=False,
            )

        for test_type in TEST_TYPES:
            if kendalls_tau_valid_test[test_type]:
                kendalls_tau_valid_test_tmp = pd.concat(
                    kendalls_tau_valid_test[test_type]
                )
                kendalls_tau_valid_test_tmp["optimizer"] = OPTIMIZER
                kendalls_tau_valid_test_tmp.to_csv(
                    f"../csvs/raw/{partial_file_name}_kendalls_tau_valid_test_{test_type}.csv",
                    index=False,
                )

    elif args.type in [
        "post_naive",
    ]:
        results = {"test": [], "test_retrained": []}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            tasks = []
            for seed in SEEDS:
                tasks.append(
                    executor.submit(
                        analyze_results_post,
                        results_path,
                        args.results_subfolder,
                        seed,
                        args.type,
                        args.n_repeats,
                        args.check_files,
                    )
                )

            for future in concurrent.futures.as_completed(tasks):
                try:
                    results_tmp = future.result()
                    for test_type in TEST_TYPES:
                        if results_tmp[test_type]:
                            results[test_type].append(pd.concat(results_tmp[test_type]))
                except Exception as e:
                    print(f"Exception occurred: {e}")

        for test_type in TEST_TYPES:
            if results[test_type]:
                results_tmp = pd.concat(results[test_type])
                results_tmp["optimizer"] = OPTIMIZER
                results_tmp.to_csv(
                    f"../csvs/raw/{partial_file_name}_{args.type}_{test_type}.csv",
                    index=False,
                )
    elif args.type == "post_naive_simulate_repeatedholdout":
        results = {"test": [], "test_retrained": []}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            tasks = []
            for seed in SEEDS:
                tasks.append(
                    executor.submit(
                        analyze_results_post_naive_simulate_repeatedholdout,
                        results_path,
                        args.results_subfolder,
                        seed,
                        args.type,
                        args.n_repeats,
                        args.check_files,
                    )
                )

            for future in concurrent.futures.as_completed(tasks):
                try:
                    results_tmp = future.result()
                    for test_type in TEST_TYPES:
                        if results_tmp[test_type]:
                            results[test_type].append(pd.concat(results_tmp[test_type]))
                except Exception as e:
                    print(f"Exception occurred: {e}")

        for test_type in TEST_TYPES:
            if results[test_type]:
                results_tmp = pd.concat(results[test_type])
                results_tmp["optimizer"] = OPTIMIZER
                results_tmp.to_csv(
                    f"../csvs/raw/{partial_file_name}_{args.type}_{args.n_repeats}_{test_type}.csv",
                    index=False,
                )
