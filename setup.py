"""
KK-Utils - Core Utility Library

Setup script for installation and distribution.

Installation:
    # Development mode (editable)
    pip install -e .
    
    # Production mode
    pip install .
    
    # From PyPI (future)
    pip install kk-utils

Usage:
    from kk_utils import load_environment, setup_logging, ConfigLoader
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = [
    line.strip()
    for line in requirements_path.read_text().splitlines()
    if line.strip() and not line.startswith("#")
] if requirements_path.exists() else []

setup(
    name="kk-utils",
    version="1.0.0",
    author="KK",
    description="Core utility library for Python projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/kk-utils",
    packages=find_packages(),
    package_data={
        "kk_utils": ["ai/prompts/*.yaml"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={},
    include_package_data=True,
    zip_safe=False,
)
