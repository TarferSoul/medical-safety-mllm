from setuptools import setup, find_packages

setup(
    name="radgraph",
    version="0.1.0.post1",
    description="RadGraph (local-fixed): offline-capable with local model paths",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
        "appdirs",
        "dotmap",
    ],
)
