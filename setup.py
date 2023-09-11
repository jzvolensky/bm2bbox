from setuptools import setup, find_packages

setup(
    name = "bm2bbox",
    version = "1.0.3",
    description = "Converts a binary mask to a bounding box",
    author= "Juraj Zvolenský",
    author_email = "juro.zvolensky@gmail.com",
    packages=find_packages(include=["bm2bbox", "bm2bbox.*"]),
    install_requires=[
        'geojson>=3.0.1',
        'iniconfig>=2.0.0',
        'numpy>=1.25.2',
        'opencv-python>=4.8.0',
        'packaging>=23.1',
        'pluggy>=1.3.0',
        'pytest>=7.4.1',
    ],
    entry_points={'console_scripts': ['bm2bbox=bm2bbox.main:main']},
)


