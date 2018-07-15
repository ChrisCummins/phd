# Compiler Fuzzing through Deep Learning
[Chris Cummins](http://chriscummins.cc/),
[Pavlos Petoumenos](http://homepages.inf.ed.ac.uk/ppetoume/),
[Alastair Murray](http://www.alastairmurray.co.uk),
[Hugh Leather](http://homepages.inf.ed.ac.uk/hleather/).

**Winner of Distinguished Paper Award ISSTA'18**

<a href="https://chriscummins.cc/pub/2018-issta.pdf">
  <img src="https://chriscummins.cc/pub/2018-issta.png" height="325">
</a>

**Abstract**

> Random program generation - *fuzzing* - is an effective technique for
> discovering bugs in compilers but successful fuzzers require extensive
> development effort for every language supported by the compiler, and often
> leave parts of the language space untested.
>
> We introduce DeepSmith, a novel machine learning approach to accelerating
> compiler validation through the inference of generative models for compiler
> inputs. Our approach *infers* a learned model of the structure of real world
> code based on a large corpus of open source code. Then, it uses the model to
> automatically generate tens of thousands of realistic programs. Finally, we
> apply established differential testing methodologies on them to expose bugs
> in compilers. We apply our approach to the OpenCL programming language,
> automatically exposing bugs with little effort on our side. In 1,000 hours of
> automated testing of commercial and open source compilers, we discover bugs
> in all of them, submitting 67 bug reports. Our test cases are on average two
> orders of magnitude smaller than the state-of-the-art, require 3.03x less
> time to generate and evaluate, and expose bugs which the state-of-the-art
> cannot. Our random program generator, comprising only 500 lines of code, took
> 12 hours to train for OpenCL versus the state-of-the-art taking 9 man months
> to port from a generator for C and 50,000 lines of code. With 18 lines of
> code we extended our program generator to a second language, uncovering
> crashes in Solidity compilers in 12 hours of automated testing.

```
@inproceedings{cummins2018b,
  title={Compiler Fuzzing through Deep Learning},
  author={Cummins, Chris and Petoumenos, Pavlos and Murray, Alastair and Leather, Hugh},
  booktitle={ISSTA},
  year={2018},
  organization={ACM}
}
```

## Resources

See 
[//docs/2018_07_issta/artifact_evaluation](/docs/2018_07_issta/artifact_evaluation/)
for the supporting artifact of the paper.

See [//deeplearning/deepsmith](/deeplearning/deepsmith/) for the DeepSmith 
source code.
