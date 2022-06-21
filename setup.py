from setuptools import find_packages, setup

setup(
    name='funwithgans',
    version='0.1',
    packages=find_packages(),
    description="funwithgans repo",
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'torch==1.0.1',
        'torchvision==0.4.1',
        'numpy==1.22.0',
        'matplotlib==3.1.1',
    ]

)