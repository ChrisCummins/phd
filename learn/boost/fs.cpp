#include <future>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <openssl/md5.h>

namespace fs = boost::filesystem;

namespace file {

std::string md5sum(const fs::path& path) {
  auto file_descript = open(path.string().c_str(), O_RDONLY);
  if (file_descript < 0)
    throw std::runtime_error("failed to open file");

  auto file_size = fs::file_size(path);
  auto file_buffer = reinterpret_cast<unsigned char*>(
      mmap(0, file_size, PROT_READ, MAP_SHARED, file_descript, 0));
  unsigned char md5[MD5_DIGEST_LENGTH];
  MD5(file_buffer, file_size, md5);
  munmap(file_buffer, file_size);

  std::ostringstream os;
  os << std::hex << std::setfill('0');

  for (int i = 0; i < MD5_DIGEST_LENGTH; ++i)
    os << std::setw(2) << static_cast<int>(md5[i]);

  return os.str();
}


//
// Walk the files in a filesystem, applying unwary op to each regular
// file, starting at root.
//
template<typename UnaryOp>
void walk_files(const fs::path& root, UnaryOp op,
                const bool follow_symlinks = false) {
  if (fs::is_symlink(root) && !follow_symlinks)
    return;

  if (fs::exists(root)) {
    if (fs::is_regular_file(root)) {
      op(root);
    } else if (fs::is_directory(root)) {
      for (const auto& entry : fs::directory_iterator(root))
        if (fs::is_directory(entry.path())) {
          std::async(std::launch::async, [&]() {
              // std::cerr << "New thread!" << std::endl;
              walk_files(entry.path(), op);
          });
        } else {
          walk_files(entry.path(), op);
        }
    }
  } else {
    std::cerr << "warning: " << root << " not found.\n";
  }
}

}  // namespace file


#include <mutex>

// std::mutex cout_lock;

std::ostream& get_cout() {
  return std::cout;
}


int main(int argc, char** argv) {
  //
  // Our file operator. Prints the checksum and path.
  //
  auto op = [](const auto& path) {
    try {
      get_cout() << file::md5sum(path) << ' ' << path.string() << std::endl;
    } catch (std::runtime_error& e) {
      std::cerr << "error: failed to open " << path.string() << std::endl;
    }
  };

  if (argc > 1) {
    for (auto i = 1; i < argc; i++)
      file::walk_files(fs::path{argv[i]}, op, false);
  } else {
    file::walk_files(fs::path{"."}, op, false);
  }

  return 0;
}
