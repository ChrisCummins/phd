// A yes/no user input prompt in two styles.

#include <exception>
#include <iostream>
#include <string>

enum class UsingTheForce { yes, no, iDontKnow };

UsingTheForce decode(const std::string &input) {
  if (input == "y")
    return UsingTheForce::yes;
  else if (input == "n")
    return UsingTheForce::no;
  else
    return UsingTheForce::iDontKnow;
}

// Prompt user for (y/n). If invalid input, repeat.
void doOrDoNotThereIsNoTry() {
  bool done = false, useTheForce = false;
  std::string input;

  while (!done) {
    std::cout << "Use the force? (y/n) ";
    std::cin >> input;

    switch (decode(input)) {
      case UsingTheForce::yes:
        useTheForce = true;
        [[clang::fallthrough]];
      case UsingTheForce::no:
        done = true;
        break;
      default:
        std::cout << "Fool! Try again.\n";
    }
  }

  std::cout << (useTheForce ? "Very well. May the force be with you."
                            : "I never did like you.")
            << std::endl;
}

class InputException : std::exception {
} ex;

// Same as before, but exception is thrown if input is invalid.
void okTry() {
  std::cout << "User the force? (y/n) ";
  std::string input;
  std::cin >> input;

  switch (decode(input)) {
    case UsingTheForce::yes:
      std::cout << "Very well. May the force be with you." << std::endl;
      break;
    case UsingTheForce::no:
      std::cout << "I never did like you." << std::endl;
      break;
    default:
      throw ex;
  }
}

int main(int argc, char **argv) {
  doOrDoNotThereIsNoTry();

  try {
    okTry();
  } catch (InputException &) {
    std::cout << "You had one shot!" << std::endl;
    return 1;
  }

  return 0;
}
