from distutils.core import setup

setup(
    name='stats_arrays',
    version='0.2',
    author='Chris Mutel',
    author_email='cmutel@gmail.com',
    url='https://bitbucket.org/cmutel/stats_arrays',
    packages=[
        'stats_arrays',
        'stats_arrays.distributions',
        'stats_arrays.tests'
    ],
    license='BSD 2-clause; LICENSE.txt',
    long_description=open('README.rst').read(),
)
