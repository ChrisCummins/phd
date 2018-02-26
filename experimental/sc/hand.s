	.text
	.p2align 4,,15
	.globl scheme_entry
	.type scheme_entry, @function
scheme_entry:
	movl $42, %eax
	ret
