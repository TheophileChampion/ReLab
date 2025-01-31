from setuptools import setup, find_packages

# TODO Add support for C++ library
setup(
    name="relab",
    version="1.0.0",
    description="ReLab is a user-friendly library to streamline reinforcement learning experiments, offering prebuilt RL agents, Gym integration, and performance visualization.",
    keywords="pytorch, reinforcement learning, deep learning",
    author="Théophile Champion",
    author_email="theoph.champion@gmail.com",
    url="https://github.com/TheophileChampion/ReLab/",
    license="MIT License",
    packages=find_packages(),
    scripts=[
        "scripts/run_experiment",
        "scripts/run_training",
        "scripts/run_demo",
        "scripts/draw_graph",
        "scripts/describe_params",
        "scripts/display_checkpoint",
        "scripts/update_checkpoint",
        "scripts/test_install",
    ],
    install_requires=[
        "gymnasium[atari, other]==1.0.0",
        "torch==2.5.1",
        "torchrl==0.6.0",
        "tensorboard==2.18.0",
        "imageio==2.36.0",
        "pillow==10.4.0",
        "pandas==2.2.3",
        "seaborn==0.13.2",
        "matplotlib==3.9.2",
        "pytest==8.3.4",
        "torchvision==0.20.1",
        "pybind11==2.13.6",
        "invoke==2.2.0",
        "psutil==6.1.1",
        "numpy==2.2.1",
    ],
    python_requires="~=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
