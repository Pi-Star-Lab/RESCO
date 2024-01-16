import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resco-benchmark",
    version="1.0.0",
    author="James Ault",
    author_email="jault@tamu.edu",
    description="The Reinforcement Learning Benchmarks for Traffic Signal Control (RESCO)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pi-Star-Lab/RESCO",
    project_urls={
        "Bug Tracker": "https://github.com/Pi-Star-Lab/RESCO/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.16.5',
        'matplotlib>=3.5.2',
        'torch>=1.8.1',
        'gym<=0.22.0',  # PFRL needs an update for future versions
        'pfrl>=0.2.1',
        'traci'
    ],
    extras_require={
        "fma2c": ['tensorflow==1.15.5'],
        "libsumo": ['libsumo']
    },
    python_requires=">=3.7.4",
)
