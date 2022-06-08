from setuptools import setup, find_packages


setup(name = 'cosmoconvert',
      version = '0.1',
      description = 'CosmoConvert: Convert between simulations because, why not!',
      url = 'https://github.com/rennehan/cosmoconvert',
      author = 'Doug Rennehan',
      author_email = 'douglas.rennehan@gmail.com',
      license = 'GPLv3',
      packages = find_packages(),
      zip_safe = False, install_requires=['numpy', 'scipy', 'h5py'])
