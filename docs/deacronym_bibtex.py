"""Expand shorthand conference and journal names to their full version.

E.g. "CGO" -> "International Symposium on Code Generation and Optimization".

This uses a manually curated set of expansion rules.
"""
import sys

import bibtexparser

from labm8 import app

FLAGS = app.FLAGS

app.DEFINE_input_path('bibtex_path', None,
                      'Path of bibtex file to process in place')

# LaTeX-escaped strings (e.g. "\&" instead of "&").
expansions = {
    'inproceedings': {
        'AAAI': 'Conference on Artificial Intelligence (AAAI)',
        'ACL':
        'Annual Meeting of the Association for Computational Linguistics (ACL)',
        'ADAPT':
        'International Workshop on Adaptive Self-tuning Computing Systems (ADAPT)',
        'AINA':
        'International Conference on Advanced Information Networking and Applications (AINA)',
        'ASE':
        'IEEE/ACM International Conference on Automated Software Engineering (ASE)',
        'ASPLOS':
        'International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)',
        'Big Data': 'International Conference on Big Data',
        'CASES':
        'International Conference on Compilers, Architectures and Synthesis for Embedded Systems (CASES)',
        'CCS':
        'ACM SIGSAC Conference on Computer and Communications Security (CCS)',
        'CF': 'ACM International Conference on Computing Frontiers (CF)',
        'CGO':
        'International Symposium on Code Generation and Optimization (CGO)',
        'CHASE':
        'International Workshop on Cooperative and Human Aspects of Software Engineering (CHASE)',
        'CSCW':
        'ACM Conference on Computer Supported Cooperative Work \\& Social Computing (CSCW)',
        'CVPR': 'Conference on Computer Vision and Pattern Recognition (CVPR)',
        'e-Energy':
        'International Conference on energy-Efficient Computing and Networking (e-Energy)',
        'ECCV': 'European Conference on Computer Vision (ECCV)',
        'ESEC/FSE':
        'ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE)',
        'FSE':
        'ACM SIGSOFT International Symposium on the Foundations of Software Engineering (FSE)',
        'GPGPU':
        'Workshop on General-Purpose Computation on Graphics Processing Units (GPGPU)',
        'HiPC': 'International Conference on High Performance Computing (HiPC)',
        'HLPGPU':
        'High-Level Programming for Heterogeneous and Hierarchical Parallel Systems (HLPGPU)',
        'HotPar': 'USENIX conference on Hot topics in parallelism (HotPar)',
        'HPCA':
        'International Symposium on High-Performance Computer Architecture (HPCA)',
        'ICCS': 'International Conference on Computational Science (ICCS)',
        'ICLR': 'International Conference on Learning Representations (ICLR)',
        'ICML': 'International Conference on Machine Learning (ICML)',
        'ICPP': 'International Conference on Parallel Processing (ICPP)',
        'ICSE': 'International Conference on Software Engineering (ICSE)',
        'ICSTW':
        'International Conference on Software Testing, Verification and Validation Workshops (ICSTW)',
        'IISWC': 'International Symposium on Workload Characterization (IISWC)',
        'InPar': 'Innovative Parallel Computing (InPar)',
        'Interspeech':
        'Annual Conference of the International Speech Communication Association (Interspeech)',
        'IPDPSW':
        'International Parallel \\& Distributed Processing Symposium (IPDPSW)',
        'ISCA': 'International Symposium on Computer Architecture (ISCA)',
        'ISPASS':
        'International Symposium on Performance Analysis of Systems and software (ISPASS)',
        'ISSTA':
        'ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA)',
        'IWMSE':
        'International Workshop on Multicore Software Engineering (IWMSE)',
        'IWOCL': 'International Workshop on OpenCL (IWOCL)',
        'LCPC':
        'International Workshop on Languages and Compilers for Parallel Computing (LCPC)',
        'LCTES':
        'Languages, Compilers, Tools and Theory of Embedded Systems (LCTES)',
        'MAPL': 'Machine Learning and Programming Languages (MAPL)',
        'MCSoC':
        'International Symposium on Embedded Multicore/Many-core Systems-on-Chip (MCSoC)',
        'MICRO': 'International Symposium on Microarchitecture (MICRO)',
        'MSR': 'Working Conference on Mining Software Repositories (MSR)',
        'NeurIPS':
        'Conference on Neural Information Processing Systems (NeurIPS)',
        'NIPS': 'Conference on Neural Information Processing Systems (NIPS)',
        'OOPSLA':
        'Object-oriented Programming Systems, Languages and Applications (OOPSLA)',
        'OSDI':
        'USENIX Symposium on Operating Systems Design and Implementation (OSDI)',
        'PACT':
        'International Conference on Parallel Architectures and Compilation Techniques (PACT)',
        'PARCO': 'International Conference on Parallel Computing (PARCO)',
        'PLDI':
        'ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)',
        'POPL': 'Symposium on Principles of Programming Languages (POPL)',
        'PPoPP':
        'ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP)',
        'SANER':
        'International Conference on Software Analysis, Evolution and Reengineering (SANER)',
        'SASIMI':
        'Workshop on Synthesis And System Integration of Mixed Information Technologies (SASIMI)',
        'SC': 'Supercomputing Conference (SC)',
        'SCOPES':
        'International Workshop on Software and Compilers for Embedded Systems (SCOPES)',
        'SIGCOMM':
        'Conference of the ACM Special Interest Group on Data Communication (SIGCOMM)',
        'SIGMOD': 'International Conference on Management of Data (SIGMOD)',
        'SLE':
        'ACM SIGPLAN International Conference on Software Language Engineering (SLE)',
        'SP': 'Symposium on Security and Privacy (SP)',
        '{\\{}USENIX{\\}} Security':
        '{\\{}USENIX{\\}} Security Security Symposium',
    },
    'article': {
        'CSUR': 'ACM Computing Surveys (CSUR)',
        'DTJ': 'Digital Technical Journal (DTJ)',
        'IJHPCA':
        'International Journal of High Performance Computing Applications (IJHPCA)',
        'IJPP': 'International Journal of Parallel Programming (IJPP)',
        'TACO': 'ACM Transactions on Architecture and Code Optimization (TACO)',
    }
}

_ignored_probable_acronyms = set()


def RewriteName(entry, title, translation_table):
  """Re-write the 'title' key in 'entry' using the translation table."""
  if title not in entry:
    return

  if (entry[title].upper() == entry[title] and
      entry[title] not in translation_table and
      entry[title] not in _ignored_probable_acronyms):
    _ignored_probable_acronyms.add(entry[title])
    print(
        f'Ignoring probable acryonym {len(_ignored_probable_acronyms)}: '
        f'"{entry[title]}"',
        file=sys.stderr)
    return

  entry[title] = translation_table.get(entry[title], entry[title])


def ExpandBibtexShorthandInPlace(bibtex, expansions) -> None:
  """Expand shorthand names in-place."""
  for entry in bibtex.entries:
    if entry['ENTRYTYPE'] == 'inproceedings':
      RewriteName(entry, 'booktitle', expansions['inproceedings'])
    elif entry['ENTRYTYPE'] == 'article':
      RewriteName(entry, 'journal', expansions['article'])


def BibtexToString(bibtex) -> str:
  """Serialize a bibtex to string representation."""
  string = bibtexparser.dumps(bibtex)
  # Strip non-ASCII characters in serialized bibtex since LaTeX complains about
  # "Unicode char X not set up for use with LaTeX."
  string = string.encode('ascii', 'ignore').decode('ascii')
  string = string.replace(u"\u200B", "")
  return string


def main():
  """Main entry point."""

  # Read the input.
  with open(FLAGS.bibtex_path) as f:
    bibtex = bibtexparser.load(f)

  ExpandBibtexShorthandInPlace(bibtex, expansions)
  string = BibtexToString(bibtex)

  # Write the result.
  with open(FLAGS.bibtex_path, 'w') as f:
    f.write(string)


if __name__ == '__main__':
  app.Run(main)
