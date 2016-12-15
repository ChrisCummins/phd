Title:      Deep Learning for Compilers
Authors:    Chris Cummins
Notes:      PhD First Year Review, November 2016

Abstract:

Continued advancements in machine learning have increasingly extended the
state-of-the art in language modelling for natural language processing. Coupled
with the increasing popularity of websites such as GitHub for hosting software
projects, this raises the potential for large scale language modelling over open
source code to build probabilistic models which capture both the semantics of a
programming language and its common usage in real world applications. This
document describes my work towards the development of systems for automatically
generating programs in a given programming language. The aim of this work is
improvements to predictive modelling for compiler optimisation and compiler
testing. In my first year I have applied LSTMs to large corpuses of code mined
from open source in order to generate executable OpenCL kernels. These generated
programs have been shown to improve the performance of state-of-the-art
predictive models; though the programs are typically short and limited to
operating only on scalars and vectors of numerical values. This document
describes the plans for future work to extend this initial proof-of-concept
through the use of formal grammars to generate programs in arbitrary languages,
capable of satisfying arbitrary properties of interest. This will enable the
automatic generation of programs in any language for a which a large corpus of
real world codes is available, with a range of applications including
exploration of unknown parts of program feature spaces, and identifying bugs in
compilers.
