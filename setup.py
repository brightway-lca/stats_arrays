from setuptools import setup

setup(
    name=u'stats_arrays',
    version=u'0.4',
    author=u'Chris Mutel',
    author_email=u'cmutel@gmail.com',
    url=u'https://bitbucket.org/cmutel/stats_arrays',
    install_requires=[u"numpy", u"scipy", u"nose"],
    packages=[
        u'stats_arrays',
        u'stats_arrays.distributions',
        u'stats_arrays.tests',
        u'stats_arrays.tests.distributions',
    ],
    license=u'BSD 2-clause; LICENSE.txt',
    long_description=open(u'README.rst').read(),
)
