from setuptools import setup, find_packages

setup(
    name = "bm2bbox",
    version = "0.1.0",
    description = "Converts a binary mask to a bounding box",
    author= "Juraj Zvolenský",
    author_email = "juro.zvolensky@gmail.com",
    packages=find_packages(include=["bm2bbox", "bm2bbox.*"]),
    install_requires=[
        "opencv-python>=4.7.0",
        "numpy>=1.25.0"
    ],
    entry_points={'console_scripts': ['bm2bbox=bm2bbox.main:main']},
)


