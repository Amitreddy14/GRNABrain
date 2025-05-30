# GRNABrain<br><sup>Locus Inference and Generative Adversarial Network for gRNA Design</sup>

### Abstract

The advent of Clustered Regularly Interspaced Short Palindromic Repeats (CRISPR) technology, notably the CRISPR-Cas9 system—an RNA-guided DNA endonuclease-introduces an exciting era of precise gene editing. Now, a central problem becomes the design of guide RNA (gRNA), the sequence of RNA responsible for locating a bind location in the genome for the CRISPR-Cas9 protein. While existing tools can predict gRNA activity, only experimental or algorithmic methods are used to generate gRNA specific to DNA subsequences. In this study, we propose LIGAND, a model which leverages a generative adversarial network (GAN) and novel attention-based architectures to simultaneously address on- and off- target gRNA activity prediction and gRNA sequence generation. LIGAND’s generator produces a plurality of viable, highly precise, and effective gRNA sequences with a novel objective function consideration for off-site activity minimization, while the discriminator maintains state-of-the-art performance in gRNA activity prediction with any DNA and epigenomic prior. This dual functionality positions LIGAND as a versatile tool with applications spanning medicine and research.

### Results

LIGAND can sucessfully generate and discriminate biologically-validated activity for arbitrary DNA/epigenomic/gRNA pairings, including the consideration of offsite effects in both generation and discrimination. For particular gene knockout regions, we can design gRNA to specifically cleave in those regions while maintaining minimal off-site effects.

![Screenshot 2025-04-21 020052](https://github.com/user-attachments/assets/65b971bf-fd30-494c-8075-c0150d3b7a2b)

*Average predicted activity over DNA region for validated gRNA*

![Screenshot 2025-04-21 020145](https://github.com/user-attachments/assets/a89894f8-6bbd-4516-8e1d-3fd07cf9baf7)


*Generation and top-5 evaluation of candidate gRNA for particular gene knockout regions*


---
