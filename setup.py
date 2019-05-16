from setuptools import setup

setup(
    name='stats_arrays',
    version='0.6',
    author='Chris Mutel',
    author_email='cmutel@gmail.com',
    url='https://bitbucket.org/cmutel/stats_arrays',
    install_requires=["numpy", "scipy"],
    packages=[
        'stats_arrays',
        'stats_arrays.distributions',
    ],
    license="BSD 3-Clause License",
    description="Standard NumPy array interface for defining uncertain parameters",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
