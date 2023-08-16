# Installation

## For Windows

### Set up using Anaconda

The most straightforward way to install the requirements if you do not already have Python 3 installed is to download and install [Anaconda for Python 3](https://www.anaconda.com/download/). 

Download and install [graphviz](http://graphviz.org/download/) for Windows.

## For Mac OSX

### Set up using Anaconda

The most straightforward way to install the requirements if you do not already have Python 3 installed is to download and install [Anaconda for Python 3](https://www.anaconda.com/download/). 

Download and install [graphviz](http://graphviz.org/download/) for Mac OSX.

## For all, from your command-line

If it is the first time, download or `git clone` this repository to your computer.

    git clone https://github.com/UppsalaIM/2IS077

Create a new Conda environment from the provided `environment.yml` file. From the command-line, type:

    conda env create -f environment.yml

Activate the newly created Conda environment:

    conda activate lab1
    
You can then launch Jupyter:

    jupyter notebook

## Updating your local files

If you already installed everything and cloned the repository, you may periodically need to update your local files when an update as been made to the remote repository (like when new content has been made available):

    git pull
  