from setuptools import setup

setup(name='funniest',
      version='0.1',
      description='Efficient Shapley performance attribution for least-squares problems',
      url='https://github.com/cvxgrp/ls-spa',
      license='MIT',
      packages=['ls_spa'],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
      ],
      zip_safe=False)
