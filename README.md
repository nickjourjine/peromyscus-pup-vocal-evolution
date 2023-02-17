This repository contains code for reproducing results and figures from

Jourjine, N., Woolfolk, M.L., Sanguinetti-Scheck, J.I., Sabatini, J.E., McFadden, S., Lindholm, A.K., Hoekstra, H.E., 2022. Two pup vocalization types are genetically and functionally separable in deer mice (preprint). BioRxiv. https://doi.org/10.1101/2022.11.11.516230

# How to use

Code to reproduce analyses for all figures is in the `scripts` directory. This directory contains six jupyter notebooks and one R script, each of which is written to carry out a specific set of analyses described below. The notebooks use a set of helper functions located in the .py files in the `src` directory. You can install this code by navigating to your local copy of the repository and running `pip install -e .`

## conda environments

The conda environments used to produce the results in the paper are located in the files `article-env.yml`, which contains all of the python packages we used, and `r-env.yml`, which contains all of the R packages used for statistical analyses. You can reproduce these environments by running `conda env create -f article-env.yml` and `conda env create -f r-env.yml`, respectively. 

## Jupyter Notebooks

### Segmenting and UMAP.ipynb

This notebook carries out the segmenting and UMAP embedding in Figure 1 panel C and Supplemental Figure 1.

### Annotate from UMAP.ipynb

This notebook produces the annotations analyzed in Figure 2.

### warbleR_feature_extraction.R

This script uses the R package warbleR to calculate acoustic features of vocalization segments produced by the Segmenting and UMAP.ipynb notebook. It takes three inputs: species (the species to calculate features from); wavs.dir (the directory containing the segmented vocalizations from that species with one .wav file per vocalization); and analysis.dir (the directory where a csv with acoustic features calculated for each of those wav files will be saved). It is written to be called from the command line along with these inputs, e.g. `R path/to/warbleR_feature_extraction.R species wavs.dir analysis.dir`

### Prepare warbleR Job Scripts.ipynb

We calculated acoustic features for vocalizations using a computing cluster. This notebook produces the job scripts required to do this. If you have access to a computing cluster, this is the easiest way to calculate features for all the vocalizations of all the species.

### Train Models on Features.ipynb

This notebook trains the machine learning models in Figure 2 and Supplemental Figure 2 using acoustic features calculated by warbleR_feature_extraction.R

### Analyse Playback.ipynb

This notebook produces the analyses in Figure 3.

### Analyze Vocalizations.ipynb

This notebook produces the anlayses for all other main and supplemental Figures.

