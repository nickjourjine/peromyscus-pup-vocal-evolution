# *Peromyscus* pup vocal evolution

This repository contains code for reproducing results and figures from

Nicholas Jourjine, Maya L. Woolfolk, Juan I. Sanguinetti-Scheck, John E. Sabatini, Sade McFadden, Anna K. Lindholm, Hopi E. Hoekstra,
Two pup vocalization types are genetically and functionally separable in deer mice, Current Biology, 2023 https://doi.org/10.1016/j.cub.2023.02.045

## Scientific overview

![graphical_abstract](graphical_abstract.png)

The goal of this study was to better understand the genetic basis of vocal repertoire evolution. To do this, we focused on a group of North American rodents, the [deer mice](https://elifesciences.org/articles/06813), who have recently evolved large differences in their morphologies and behaviors. Focusing on a specific behavioral context in which these mice (and other mammals) vocalize - neonatal isolation - we characterized the evolution and ontogeny of isolation-induced calls, identifying two acoustically distinct call types: high-frequency ultrasonic vocalizations (USVs) similar to those made by isolated neonatal [house mice](https://pmc.ncbi.nlm.nih.gov/articles/PMC9744514/), and low frequency cries not found in the isolation induced calls of house mice, either in the labotory or in wild, free-living populations. We show in deer mice that cries and USVs are functionally distinct, as cries but not USVs trigger rapid maternal approach to a speaker, and also genetically distinct, as their features can be uncoupled in second generation hybrids between different deer mouse species. These findings provide insight into the genetic evolution of vocalization "types" within a species' vocal repertoire, and suggest that lack of genetic contraint between cries and USVs has facilitated their functional specialization in deer mice.

## How to use

Code to reproduce analyses for all figures is in the `scripts` directory. This directory contains six jupyter notebooks and one R script, each of which is written to carry out a specific set of analyses described below. The notebooks use a set of helper functions located in the .py files in the `src` directory. You can install this code by navigating to your local copy of the repository and running `pip install -e .`

The raw data analyzed by this code is located in a Dryad database at `https://doi.org/10.5061/dryad.g79cnp5ts` Please see the README file associated with that dataset for more information about the raw data.

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

## Anaconda Environments

The conda environments used to produce the results in the paper are located in `r-env.yml`, which contains all of the R packages used for statistical analyses, and `article-env.yml`, which contains python packages used to carry out all other analyses. You can reproduce these environments by running `conda env create -f r-env.yml` and `conda env create -f article-env.yml`, respectively.

## Two letter species codes

We use two-letter codes as shorthand to refer to the different taxa we analyze. They are:
|code|taxon                                |
|--- |-------------------------------------|
|BW  | *P. maniculatus bairdii*            |
|BK  | *P. maniculatus gambelli*           |
|SW  | *P. maniculatus rubidus*            |
|NB  | *P. maniculatus nubiterrae*         |
|PO  | *P. polionotus subgriseus*          | 
|LO  | *P. polionotus leucocephalus*       |
|GO  | *P. gossypinus*                     |
|LL  | *P. leucopus*                       |
|MU  | *Mus musculus domesticus* (C57Bl6/j)|
|MZ  | *Mus musculus domesticus* (wild)    |
