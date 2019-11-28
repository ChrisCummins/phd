/* -*- c-basic-offset: 8; -*-
 *
 * cdecl.c - C declaration to pseudo-English translator.
 *
 * My attempt to solve the Chapter 3 programming challenge of "Expert
 * C".
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXTOKENLEN 128
#define MAXTOKENS 128

enum type_tag { QUALIFIER, TYPE, IDENTIFIER };

struct token_t {
  char type;
  char string[MAXTOKENLEN];
};

/* holds tokens we read before reaching first identifier */
struct token_t stack[MAXTOKENS];

/* holds the size of the stack */
int stack_size = 0;

/* stack operations */
#define pop stack[stack_size--]
#define push(s) stack[stack_size++] = (s)
#define head stack[stack_size - 1]

/* holds the token just read */
struct token_t this;

/* utility routines -------- */

void print_state(const char *src) {
  char type[MAXTOKENLEN];

  if (this.type == IDENTIFIER)
    strcpy(type, "identifier");
  else if (this.type == QUALIFIER)
    strcpy(type, "qualifier ");
  else if (this.type == TYPE)
    strcpy(type, "type      ");

  printf("this: %s %s           %s\n", type, this.string, src);
}

/* look at the current token and set this.type to: "type",
 * "qualifier", or "identifier". */
enum type_tag classify_string(const char *const string) {
  if (strcmp(string, "volatile") == 0)
    return QUALIFIER;
  else if (strcmp(string, "const") == 0)
    return QUALIFIER;
  else if (strcmp(string, "unsigned") == 0)
    return TYPE;
  else if (strcmp(string, "signed") == 0)
    return TYPE;
  else if (strcmp(string, "int") == 0)
    return TYPE;
  else if (strcmp(string, "float") == 0)
    return TYPE;
  else if (strcmp(string, "double") == 0)
    return TYPE;
  else if (strcmp(string, "char") == 0)
    return TYPE;
  else if (strcmp(string, "struct") == 0)
    return TYPE;
  else if (strcmp(string, "union") == 0)
    return TYPE;
  else /* unrecognised, assume identifier */
    return IDENTIFIER;
}

/* read the next token into this.string.
 * If it is alphanumeric, classify_string.
 * Else it must be a single character token.
 *   this.type = the token itself;
 * terminate this.string with a nul. */
const char *consume_token(const char *src) {
  char *p = this.string;

  /* ignore leading whitespace */
  while (*src == ' ' || *src == '\t') src++;

  if (isalnum(*src)) {
    while (isalnum(*p++ = *src++))
      ;
    *(p - 1) = '\0';
    src--;
    this.type = classify_string(this.string);
  } else if (*src == '*') {
    strcpy(p, "pointer to");
    this.type = *src++;
  } else {
    *p++ = *src;
    *p++ = '\0';
    this.type = *src++;
  }

  // print_state(src);

  return src++;
}

/* consume_token and push it onto the stack until the first identifier is read.
 * Print "identifier is", this.string */
const char *read_to_first_identifier(const char *src) {
  src = consume_token(src);

  while (this.type != IDENTIFIER) {
    push(this);
    src = consume_token(src);
  }

  printf("%s is ", this.string);

  return src;
}

/* parsing routines -------- */

/* read past closing ')' print out "function returning" */
const char *deal_with_function_args(const char *src) {
  while (this.type != ')') {
    src = consume_token(src);
  }

  src = consume_token(src);
  printf("function returning ");

  return src;
}

/* while you've got "[size]" print it out and read past it */
const char *deal_with_arrays(const char *src) {
  while (this.type == '[') {
    printf("an array ");
    src = consume_token(src);
    if (isdigit(this.string[0])) {
      printf("[0,%d] ", atoi(this.string) - 1);
      src = consume_token(src);
    }
    src = consume_token(src);
    printf("of");
  }

  return src;
}

/* while you've got "*" on the stack print "pointer to" and pop it */
const char *deal_with_pointers(const char *src) {
  while (head.type == '*') printf("%s ", pop.string);

  return src;
}

/* if this.type is '[' deal_with_arrays.
 * if this.type is '(' deal_with_function_args
 * deal_with_pointers
 * while there's stuff on the stack
 *   if it's a '('
 *     pop it and consume_token; it should be a closing ')'
 *     deal_with_declarator
 *   else pop it and print */
const char *deal_with_declarator(const char *src) {
  src = consume_token(src);

  if (this.type == '[')
    src = deal_with_arrays(src);
  else if (this.type == '(')
    src = deal_with_function_args(src);

  src = deal_with_pointers(src);

  while (stack_size >= 0) {
    if (head.type == '(') {
      (void)pop;
      src = consume_token(src);
      src = deal_with_declarator(src);
    } else {
      printf("%s ", pop.string);
    }
  }

  return src;
}

/* main routine -------- */

/* read_to_first_identifier
 * deal_with_declarator */
void decl(const char *src) {
  src = read_to_first_identifier(src);
  deal_with_declarator(src);
  printf("\n");
}

int main(int argc, char **argv) {
  decl("const int *bar[5]");
  decl("char *(*c[10])(int **p)");

  return 0;
}
