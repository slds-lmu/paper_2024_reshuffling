from setuptools import find_packages, setup

setup(
    name="reshufflebench",
    version="0.0.9",
    author="",
    author_email="",
    packages=find_packages(),
    url="",
    license="",
    description="",
    long_description="",
    install_requires=[
        "numpy==1.25.2",
        "pandas==2.1.0",
        "scipy==1.11.2",
        "optuna==3.3.0",
        "scikit-learn==1.3.0",
        "torch==2.1.1",
        "tqdm==4.66.1",
        "setuptools==65.5.1",
        "pyarrow==13.0.0",
        "openml==0.14.1",
        "tabpfn==0.1.9",
        "xgboost==2.0.2",
        "catboost==1.2.1",
    ],
    extras_require={
        "test": ["pytest==7.4.3"],
        "HEBO": ["HEBO==0.3.5"],
    },
    python_requires=">=3.10",
    classifiers=[
        # Trove classifiers
        # Full list at https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
