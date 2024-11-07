from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
from optuna.trial import Trial

from reshufflebench.algorithms.classifier import Classifier
from reshufflebench.utils import unify_missing_values


class TestClassifier(Classifier):
    def __init__(
        self,
        impute_x_cat: bool,
        impute_x_num: bool,
        encode_x: bool,
        scale_x: bool,
        seed: int,
    ):
        super().__init__(
            classifier_id="test",
            impute_x_cat=impute_x_cat,
            impute_x_num=impute_x_num,
            encode_x=encode_x,
            scale_x=scale_x,
            seed=seed,
        )
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.cat_features = None

    def get_hebo_search_space(self, **kwargs):
        return None

    def get_configspace_search_space(self, **kwargs):
        return None

    def get_internal_optuna_search_space(self, **kwargs):
        return None

    def construct_classifier(self, trial: Trial, **kwargs) -> None:
        pass

    def construct_classifier_refit(self, trial: Trial, **kwargs) -> None:
        pass

    def _fit(
        self,
        trial: Trial,
        x_train: np.array,
        y_train: np.array,
        x_valid: Optional[np.array] = None,
        y_valid: Optional[np.array] = None,
        cat_features: Optional[List[int]] = None,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.cat_features = cat_features


@pytest.fixture(params=[True, False])
def impute_x_cat(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[True, False])
def impute_x_num(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[True, False])
def encode_x(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[True, False])
def scale_x(request: pytest.FixtureRequest) -> bool:
    return request.param


def test_preprocessor(
    impute_x_cat: bool, impute_x_num: bool, encode_x: bool, scale_x: bool
):
    classifier = TestClassifier(
        impute_x_cat=impute_x_cat,
        impute_x_num=impute_x_num,
        encode_x=encode_x,
        scale_x=scale_x,
        seed=0,
    )

    assert classifier.classifier_id == "test"
    assert classifier.impute_x_cat == impute_x_cat
    assert classifier.impute_x_num == impute_x_num
    assert classifier.encode_x == encode_x
    assert classifier.scale_x == scale_x
    assert classifier.seed == 0
    assert classifier.classifier is None

    n = 1000

    feature_1 = np.repeat(1.0, n)

    categories_2 = ["2A", "2B", "2C"]
    feature_2 = np.random.choice(categories_2, size=n).astype(object)

    feature_3 = np.repeat(3, n)

    categories = ["4A", "4B", "4C"]
    feature_4 = np.random.choice(categories, size=n).astype(object)

    data = pd.DataFrame(
        {
            "NumericFeature1": feature_1,
            "CategoricalFeature1": feature_2,
            "IntegerFeature1": feature_3,
            "CategoricalFeature2": feature_4,
        }
    )

    cat_features = [1, 3]
    num_features = [0, 2]

    data.iloc[0:200, 0:4] = None

    data = unify_missing_values(data)

    y = np.random.choice([0, 1], size=n)

    x_train = data.iloc[:800, :]
    x_valid = data.iloc[800:, :]
    y_train = y[:800]
    y_valid = y[800:]

    classifier.construct_pipeline(
        trial=None, refit=False, cat_features=cat_features, num_features=num_features
    )
    classifier.fit(
        trial=None,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        cat_features=cat_features,
    )

    assert classifier.x_train.shape == (800, 4)
    assert classifier.y_train.shape == (800,)
    assert classifier.x_valid.shape == (200, 4)
    assert classifier.y_valid.shape == (200,)

    # catboost
    if not impute_x_cat and not impute_x_num and not encode_x and not scale_x:
        assert all(
            classifier.preprocessor.feature_names_in_
            == [
                "NumericFeature1",
                "CategoricalFeature1",
                "IntegerFeature1",
                "CategoricalFeature2",
            ]
        )
        assert all(
            classifier.preprocessor.get_feature_names_out()
            == [
                "passthrough__passthrough__passthrough__passthrough__NumericFeature1",
                "passthrough__passthrough__passthrough__passthrough__CategoricalFeature1",
                "passthrough__passthrough__passthrough__passthrough__IntegerFeature1",
                "passthrough__passthrough__passthrough__passthrough__CategoricalFeature2",
            ]
        )
        assert classifier.x_valid[0, 0] == 1.0
        assert classifier.x_valid[0, 1] in ["2A", "2B", "2C"]
        assert classifier.x_valid[0, 2] == 3.0
        assert classifier.x_valid[0, 3] in ["4A", "4B", "4C"]

    # logreg, funnel_mlp
    if impute_x_cat and impute_x_num and encode_x and scale_x:
        assert all(
            classifier.preprocessor.feature_names_in_
            == [
                "NumericFeature1",
                "CategoricalFeature1",
                "IntegerFeature1",
                "CategoricalFeature2",
            ]
        )
        assert all(
            classifier.preprocessor.get_feature_names_out()
            == [
                "scaler__encoder__remainder__imputer_cat__CategoricalFeature1",
                "scaler__encoder__remainder__imputer_cat__CategoricalFeature2",
                "scaler__remainder__imputer_num__remainder__NumericFeature1",
                "scaler__remainder__imputer_num__remainder__IntegerFeature1",
            ]
        )
        for i in range(4):
            assert all(classifier.x_valid[:, i] < 7) and all(
                classifier.x_valid[:, i] > -7
            )
            assert (
                len(np.unique(classifier.x_valid[:, i])) == 3
                if i in [0, 1]
                else len(np.unique(classifier.x_valid[:, i])) == 1
            )
        assert all(pd.isnull(classifier.x_train).sum(axis=0) == np.array([0, 0, 0, 0]))

    # xgboost, tabpfn
    if impute_x_cat and not impute_x_num and encode_x and not scale_x:
        assert all(
            classifier.preprocessor.feature_names_in_
            == [
                "NumericFeature1",
                "CategoricalFeature1",
                "IntegerFeature1",
                "CategoricalFeature2",
            ]
        )
        assert all(
            classifier.preprocessor.get_feature_names_out()
            == [
                "passthrough__encoder__passthrough__imputer_cat__CategoricalFeature1",
                "passthrough__encoder__passthrough__imputer_cat__CategoricalFeature2",
                "passthrough__remainder__passthrough__remainder__NumericFeature1",
                "passthrough__remainder__passthrough__remainder__IntegerFeature1",
            ]
        )
        for i in range(2):
            assert all(classifier.x_valid[:, i] < 7) and all(
                classifier.x_valid[:, i] > -7
            )
            assert len(np.unique(classifier.x_valid[:, i])) == 3
        assert all(
            pd.isnull(classifier.x_train).sum(axis=0) == np.array([0, 0, 200, 200])
        )
