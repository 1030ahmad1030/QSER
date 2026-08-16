"""
QSER Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="QSER",
    version="1.0.0",
    author="Ahmad Muhammad, Fatih K\"ulahcı",
    author_email="ahmad.muhammad@qu.edu.qa",
    description="A Data Physics Framework for Forward and Inverse Modeling of Physical Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1030ahmad1030/QSER",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics"
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0"
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "jax": ["jax>=0.4.0", "jaxlib>=0.4.0"],
        "openfoam": ["foamlib>=0.5.0"],
        "all": [
            "torch>=2.0.0",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "foamlib>=0.5.0",
            "QSignature>=1.0.6"
        ]
    },
)
