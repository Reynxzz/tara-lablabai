"""Setup script for documentation generation agent"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="tara",
    version="1.0.0",
    description="AI-powered learning path generator for GitHub projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TARA Team",
    author_email="tara@example.com",
    url="https://github.com/Reynxzz/tara-lablabai",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'tara=scripts.run_documentation_agent:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Documentation",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="documentation ai crewai github automation openai",
    project_urls={
        "Bug Reports": "https://github.com/Reynxzz/tara-lablabai/issues",
        "Source": "https://github.com/Reynxzz/tara-lablabai",
    },
)
