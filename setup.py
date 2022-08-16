from setuptools import find_packages, setup
import versioneer

setup(
    name="xstatstests",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Dougie Squire",
    url="https://github.com/dougiesquire/xstatstests",
    description="Statistical tests xarray objects",
    long_description="Statistical tests on xarray objects",
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
        "scipy>=1.8.0",
    ],
)
