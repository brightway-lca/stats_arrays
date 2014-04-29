from distutils.core import setup

setup(
    name='stats_arrays',
    version='0.3',
    author='Chris Mutel',
    author_email='cmutel@gmail.com',
    url='https://bitbucket.org/cmutel/stats_arrays',
    install_requires=["numpy", "scipy", "nose"],
    packages=[
        'stats_arrays',
        'stats_arrays.distributions',
        'stats_arrays.tests',
        'stats_arrays.tests.distributions',
    ],
    license='BSD 2-clause; LICENSE.txt',
    long_description=open('README.rst').read(),
)
