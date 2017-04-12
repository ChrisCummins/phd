// sc - Scheme compiler
#include <iostream>

void compile(std::ostream &o, const int x) {
  o << " .section __TEXT,__text,regular,pure_instructions" << std::endl;
  o << " .macosx_version_min 10, 11" << std::endl;
  o << " .globl _scheme_entry" << std::endl;
  o << " .align 4, 0x90" << std::endl;
  o << " _scheme_entry:" << std::endl;
  o << " .cfi_startproc" << std::endl;
  o << " pushq %rbp" << std::endl;
  o << " .cfi_def_cfa_offset 16" << std::endl;
  o << " .cfi_offset %rbp, -16" << std::endl;
  o << " movq %rsp, %rbp" << std::endl;
  o << " .cfi_def_cfa_register %rbp" << std::endl;

  // Actual code:
  o << " movl $" << x << ", %eax" << std::endl;
  o << " popq %rbp" << std::endl;
  o << " retq" << std::endl;

  o << " .cfi_endproc" << std::endl;
  o << " .subsections_via_symbols" << std::endl;
}

int main(int argc, char **argv) {
  compile(std::cout, 5);
  return 0;
}
