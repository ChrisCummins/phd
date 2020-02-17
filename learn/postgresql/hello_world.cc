#include <iostream>
#include <pqxx/pqxx>
#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"

using labm8::Status;
using labm8::StatusOr;
namespace error = labm8::error;

using std::unique_ptr;

namespace sqlutil {
namespace psql {

StatusOr<pqxx::connection*> Connect(const string& string) {
  try {
    pqxx::connection* con = new pqxx::connection(string);
    CHECK(con) << "Failed to allocate connection";
    CHECK(con->is_open()) << "Can't open database";
    return std::move(con);
  } catch (const std::exception& e) {
    return Status(error::Code::INVALID_ARGUMENT,
                  "Failed to connect to database");  //, e.what());
  }
}

}  // namespace psql
}  // namespace sqlutil

int main(int argc, char* argv[]) {
  auto C = sqlutil::psql::Connect("dbname = testdb").ValueOrDie();
  LOG(INFO) << "Connected to database";

  // Create a database.
  {
    pqxx::work txn(*C);
    txn.exec(
        "CREATE TABLE Company("
        "ID INT PRIMARY KEY     NOT NULL,"
        "NAME           TEXT    NOT NULL,"
        "AGE            INT     NOT NULL,"
        "ADDRESS        CHAR(50),"
        "SALARY         REAL );");
    txn.commit();
  }

  // Insert records.
  {
    pqxx::work txn(*C);
    txn.exec(
        "INSERT INTO Company (ID,NAME,AGE,ADDRESS,SALARY) "
        "VALUES (1, 'Paul', 32, 'California', 20000.00 ); "
        "INSERT INTO Company (ID,NAME,AGE,ADDRESS,SALARY) "
        "VALUES (2, 'Allen', 25, 'Texas', 15000.00 ); "
        "INSERT INTO Company (ID,NAME,AGE,ADDRESS,SALARY)"
        "VALUES (3, 'Teddy', 23, 'Norway', 20000.00 );"
        "INSERT INTO Company (ID,NAME,AGE,ADDRESS,SALARY)"
        "VALUES (4, 'Mark', 25, 'Rich-Mond ', 65000.00 );");
    txn.commit();
  }

  // Select records.
  {
    pqxx::nontransaction ntxn(*C);
    pqxx::result result(ntxn.exec("SELECT * FROM Company"));

    /* List down all the records */
    for (const auto& c : result) {
      std::cout << "ID = " << c[0].as<int>() << std::endl;
      std::cout << "Name = " << c[1].as<string>() << std::endl;
      std::cout << "Age = " << c[2].as<int>() << std::endl;
      std::cout << "Address = " << c[3].as<string>() << std::endl;
      std::cout << "Salary = " << c[4].as<float>() << std::endl;
    }
  }

  // Update a record.
  {
    pqxx::work txn(*C);
    pqxx::row r = txn.exec1(
        "SELECT id "
        "FROM Company "
        "WHERE name =" +
        txn.quote("Paul"));

    int employee_id = r[0].as<int>();
    txn.exec0(
        "UPDATE Company "
        "SET salary = salary + 1 "
        "WHERE id = " +
        txn.quote(employee_id));
    txn.commit();
  }
}
