from distutils.file_util import copy_file
from distutils.core import setup, Command
import os, glob


pjoin = os.path.join


setup(name='autogain',
      version='1.0a',
      description='find gain factors automatically',
      author='Marius Kriegerowski',
      author_email='marius.kriegerowski@uni-potsdam.de',
      packages=['autogain']
      )
