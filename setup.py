import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='samscore',
     version='1.0',
     author="Yunxiang Li",
     author_email="yunxiang.li@utsouthwestern.edu",
     description="SAMScore Similarity Metric",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Kent0n-Li/SAMScore",
     packages=find_packages(),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
)
