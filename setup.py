# setup.py
from setuptools import setup, find_packages

setup(
    name="remotebot",
    version="0.1.0",
    description="RemoteBot: Task-space diffusion policy with feasibility-guided denoising",
    author="Your Name/Lab",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core mathematical and deep learning libraries
        "numpy",
        "torch",
    # Legacy rigid body kinematics library
        "pinocchio",
    # Differentiable kinematics for feasibility guidance
    "pytorch-kinematics",
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