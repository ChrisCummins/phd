#include "programl/db/db.h"
#include "labm8/cpp/logging.h"

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;

namespace programl {
namespace db {

Status Database::Connect() {
  if (connection_) {
    return Status(error::Code::FAILED_PRECONDITION,
                  "Database already connected");
  }

  try {
    connection_ = new pqxx::connection(uri_);
    if (!connection_) {
      return Status(error::Code::RESOURCE_EXHAUSTED,
                    "Failed to allocate connection");
    }
    if (!connection_->is_open()) {
      return Status(error::Code::FAILED_PRECONDITION,
                    "Failed to connect to database");
    }
    return Status::OK;
  } catch (const std::exception& e) {
    return Status(error::Code::PERMISSION_DENIED,
                  "Failed to connect to database: {}", e.what());
  }
}

void Database::ConnectOrDie() {
  Status status = Connect();
  if (!status.ok()) {
    LOG(FATAL) << status.error_message();
  }
}

void Database::Disconnect() {
  CHECK(connection_);
  connection_->disconnect();
  connection_ = nullptr;
}

pqxx::connection* Database::connection() {
  CHECK(connection_);
  return connection_;
}

}  // namespace db
}  // namespace programl
