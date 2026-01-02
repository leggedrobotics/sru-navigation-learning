#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="rsl_rl",
    version="2.3.4",
    packages=find_packages(),
    author="Fan Yang, ETH Zurich",
    maintainer="Fan Yang",
    maintainer_email="fanyang1@ethz.ch",
    url="https://github.com/leggedrobotics/sru-navigation-learning",
    license="BSD-3",
    description="SRU Navigation Learning - RL training framework with Spatially-Enhanced Recurrent Units for visual navigation",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "GitPython",
        "onnx",
    ],
)
