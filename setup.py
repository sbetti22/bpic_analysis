from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# setup
setup(
    name='bpic_analysis',
    version='1.0',
    description='data analysis for JWST NIRSpec IFU data of B Pic',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sarah Betti',
    author_email='sbetti@stsci.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research ',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    keywords='jwst NIRSpec ifu BPic',
    packages=['bpic_analysis'],
    package_data={'bpic_analysis': ['*']},
    zip_safe=False
)