from setuptools import setup, find_packages
from rvi_sampling import __version__

setup(
    name='rvi_sampling',
    version=__version__,
    description='Reinforced Variational Inference Sampling',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/zafarali/better-sampling', # temporary name
    author='Zafarali Ahmed',
    author_email='zafarali.ahmed@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
    python_requires='>=3.5'
)