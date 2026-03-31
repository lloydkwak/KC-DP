# setup.py
from setuptools import setup, find_packages

setup(
    name="kc_dp",
    version="0.1.0",
    description="Kinematics-Conditioned Diffusion Policy for Cross-Embodiment Robot Manipulation",
    author="Your Name/Lab",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core mathematical and deep learning libraries
        "numpy",
        "torch",
        # High-performance rigid body kinematics library
        "pinocchio",
        # Configuration management (strictly required for our custom dataset/policy wrappers)
        "hydra-core",
        "omegaconf"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)