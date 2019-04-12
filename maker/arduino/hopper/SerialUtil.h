#pragma once

namespace serial {

// Print a message.
void Print() {}

template <typename Arg, typename... Args>
void Print(Arg&& arg, Args&&... args) {
  Serial.print(std::forward<Arg>(arg));
  Print(std::forward<Args>(args)...);
}

// Same as Print(), but with newline appended to the message.
void PrintLn() { Serial.print('\n'); }

template <typename Arg, typename... Args>
void PrintLn(Arg&& arg, Args&&... args) {
  Serial.print(std::forward<Arg>(arg));
  PrintLn(std::forward<Args>(args)...);
}

}  // namespace serial
