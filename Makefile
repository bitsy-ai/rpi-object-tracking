.PHONY: clean clean-test clean-pyc clean-build docs help rpi-deps
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 rpi_deep_pantilt tests

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source rpi_deep_pantilt -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/rpi_deep_pantilt.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ rpi_deep_pantilt
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

sdist: clean ## builds source package
	python setup.py sdist
	ls -l dist

bdist_wheel: clean ## builds wheel package
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install

rpi-deps:
	sudo apt-get update && sudo apt-get install -y \
	cmake python3-dev libjpeg-dev libatlas-base-dev raspi-gpio libhdf5-dev python3-smbus
	
	##libjasper1 libilmbase-dev libopenexr-dev libgstreamer1.0-dev libqtgui4 libqt4-test libqtcore4 \
	##libatlas3-base libwebp6 libtiff5 libopenexr23
	
protoc:
	cd $$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') && protoc object_detection/protos/*.proto --python_out=.

edgetpu-image:
	docker build -t  EdgeTPUConverter -f tools/edgetpu.Dockerfile .

