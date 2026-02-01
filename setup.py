from setuptools import setup, find_packages

setup(
    name="project-plasma",
    version="0.1.0",
    description="Project Plasma - Next-gen Agent-Lightning with Generative World Models",
    author="Project Plasma Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "numpy>=1.24.0",
        "typing-extensions>=4.5.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
