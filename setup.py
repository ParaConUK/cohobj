from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='cohobj',
    url='https://github.com/ReadingClouds/cohobj',
    author='Peter Clark',
    author_email='p.clark@reading.ac.uk',
    # Needed to actually package something
    packages=['cohobj', 
              ],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'scikit-image', 'dask', 'xarray', 'loguru'],
    # *strongly* suggested for sharing
    version='0.2.1',
    # The license can be anything you like
    license='MIT',
    description='python code to help identify and use objects in 3D LES data.',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
)