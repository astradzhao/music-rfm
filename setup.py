from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="musicrfm",
    version="0.1.0",
    author="Music RFM Team",
    description="Representation Finetuning for Music Generation Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
        "soundfile",
        "transformers",
        "pandas",
        "xrfm",
        "datasets",
        "torchaudio",
    ],
)

