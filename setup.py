# setup.py
from setuptools import setup, find_packages

setup(
    name="sms-spam-detection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "flask>=2.3.0",
        "jupyter>=1.0.0",
        "nltk>=3.8.0",
        "joblib>=1.3.0",
    ],
    python_requires=">=3.8",
)