from setuptools import find_packages, setup
import versioneer

setup(
    name="xks",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Dougie Squire",
    url="https://github.com/dougiesquire/xks",
    description="A Python library for xarray-compatible Kolmogorov-Smirnov testing",
    long_description="A Python library for xarray-compatible Kolmogorov-Smirnov testing in multiple dimensions",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "xarray",
        "scipy",
    ],
)
