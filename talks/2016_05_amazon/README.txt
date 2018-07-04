Title:      All the OpenCL on GitHub: Teaching an AI to code, one
            character at a time
Authors:    Chris Cummins
Presented:  Amazon Development Centre, Edinburgh, Scotland, Thursday
            May 19th 2016.

Abstract:

I’ll be presenting work-in-progress research toward a novel approach
for generating benchmark programs. By mining a large corpus of
publicly available source code from GitHub, a deep learning neural
network is trained to learn the distribution of program code at the
character-sequence level. This learned distribution can be used to
provide some measure of the ‘humanness’ of a given source code, with
immediate applications for verifying the representativeness of
benchmark suites against ‘real world’ programs. However, if we instead
sample from this learned distribution, we can generate entirely new
character sequences with the eventual goal of creating compilable and
executable programs. If successful, this approach could have far
reaching implications for compiler testing, optimisations and
generative programming.
