# Run from fetch3 directory, not this directory

rm -rf docs/_build
rm -rf docs/ftch
PYTHONPATH=. sphinx-autogen docs/gen_fetch3.rst
sphinx-build -b html docs docs/_build