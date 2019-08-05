// Utility code for working with protocol buffers.
#pragma once

#include "labm8/cpp/logging.h"

#include <functional>
#include <iostream>

namespace pbutil {

// Run a process_function callback that accepts a proto message and mutates
// it in place. The proto message is decoded from the given istream, and
// serialized to to the ostream.
template <typename Message>
void ProcessMessageInPlace(std::function<void(Message *)> process_function,
                           std::istream *istream = &std::cin,
                           std::ostream *ostream = &std::cout) {
  // The proto instance that we'll parse from istream.
  Message message;

  // Decode the proto from istream.
  CHECK(message.ParseFromIstream(istream));

  // Do the work.
  process_function(&message);

  // Write the message to ostream.
  CHECK(message.SerializeToOstream(ostream));
}

// Run a process_function callback that accepts a proto message and writes
// to an output proto message. The input proto message is decoded from the given
// istream, and the output proto is serialized to to the ostream.
template <typename InputMessage, typename OutputMessage>
void ProcessMessage(
    std::function<void(const InputMessage &, OutputMessage *)> process_function,
    std::istream *istream = &std::cin, std::ostream *ostream = &std::cout) {
  // The proto instance that we'll parse from istream.
  InputMessage input_message;
  // The proto instance that we'll store the result in.
  OutputMessage output_message;

  // Decode the proto from istream.
  CHECK(input_message.ParseFromIstream(istream));

  // Do the work.
  process_function(input_message, &output_message);

  // Write the message to ostream.
  CHECK(output_message.SerializeToOstream(ostream));
}

}  // namespace pbutil

// A convenience macro to run an in-place process_function as the main()
// function of a program.
#define PBUTIL_INPLACE_PROCESS_MAIN(process_function, message_type) \
  int main() {                                                      \
    pbutil::ProcessMessageInPlace<message_type>(process_function);  \
    return 0;                                                       \
  }

// A convenience macro to run an process_function as the main() function of a
// program.
#define PBUTIL_PROCESS_MAIN(process_function, input_message_type,    \
                            output_message_type)                     \
  int main() {                                                       \
    pbutil::ProcessMessage<input_message_type, output_message_type>( \
        process_function);                                           \
    return 0;                                                        \
  }
