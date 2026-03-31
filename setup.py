from setuptools import setup, find_packages

setup(
    name="kc_dp",
    version="0.1.0",
    description="Kinematics-Conditioned Diffusion Policy",
    packages=find_packages(),
    install_requires=[
        # 추후 requirements.txt에서 관리하더라도 기본적인 패키지 명시
        "numpy",
        "torch",
        "pinocchio"
    ],
)
