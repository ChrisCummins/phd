// Working through the Quick Start guide at https://gorm.io/docs/
package main

import (
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/sqlite"
)

type Product struct {
	gorm.Model
	Name  string
	Price int32 `gorm:"default:500"`
}

var FLAGS_dbpath = flag.String("dbpath", "", "Path to the database")

func main() {
	flag.Parse()

	// Create tempdir.
	dbpath := *FLAGS_dbpath
	if dbpath == "" {
		tmpdir, err := ioutil.TempDir("", "phd_go_")
		if err != nil {
			log.Fatal(err)
		}

		// Tidy up temporary directory.
		defer os.RemoveAll(tmpdir)

		dbpath = filepath.Join(tmpdir, "test.db")
	}

	// Create database.
	db, err := gorm.Open("sqlite3", dbpath)
	if err != nil {
		log.Fatal("failed to connect to database")
	}

	log.Println("Connected to database", dbpath)

	// Close database connection.
	defer db.Close()

	db.AutoMigrate(&Product{})

	db.Create(&Product{Name: "A thing you can buy", Price: 1000})

	var product Product
	db.First(&product, 1)
	db.First(&product, "name = ?", "A thing you can buy")

	db.Model(&product).Update("Price", 2000)

	db.Delete(&product)
}
