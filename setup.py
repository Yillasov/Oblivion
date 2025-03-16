from setuptools import setup, find_packages

setup(
    name="oblivion-sdk",
    version="0.1.0",
    description="Neuromorphic SDK for UCAV systems",
    author="Yessine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pyyaml>=5.4.1",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
)