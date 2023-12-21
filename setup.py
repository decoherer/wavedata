from setuptools import setup

setup(
    name="wavedata",
    version="0.1",
    py_modules=["wavedata", "plot"],
    # Additional metadata (optional but recommended)
    author="decoherer",
    author_email="63128649+decoherer@users.noreply.github.com",
    description="Tools for processing waveform-like data such as time series or xy data. Perform common scientific operations on xy data using numpy. Store, analyze, transform, and display xy data.",
    url="https://github.com/decoherer/wavedata",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Dependencies
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
    #   "another_package>=2.0",
    ],
)
