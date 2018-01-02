# Me.csv [![Release 0.0.1.dev1](https://img.shields.io/badge/release-0.0.1.dev1-blue.svg?style=flat)](https://github.com/ChrisCummins/me.csv/releases) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://opensource.org/licenses/MIT)

## Usage

### Installation

```sh
$ virtualenv -p python3.6 build/me
$ source build/me/bin/activate
$ pip install -r requirements.txt
```

### Configuration

Create file `~/.me.json`:

```
{
  "sources": {
    "omnifocus": {
      "of2_path": "/Applications/bin/of2"
    },
    "healthkit": {
      "export_path": "~/Documents/Health/export.zip"
    }
  },
  "exports": {
    "csv": {
      "path": "~/Documents/me.csv/"
    },
    "gsheet": {
      "name": "me.csv",
      "keypath": "~/google-service-key.json",
      "share_with": "me@example.com"
    }
  }
}
```

#### Import from HealthKit

1. Select "Export Health Data" on iPhone's Health app.
2. Set path to `export.zip` in `~/.my.json` > "sources" > "healthkit" > "export_path".

#### Import from OmniFocus

1. Install [of2export](https://github.com/psidnell/ofexport2).
2. Set path to `of2` in `~/.my.json` > "sources" > "omnifocus" > "of2_path".

#### Export to Google Sheets

1. [Obtain OAuth2 credentials from Google Developers Console](http://gspread.readthedocs.io/en/latest/oauth2.html).
2. Set path to the service key file in `~/.my.json` > "exports" > "gsheet" > "keypath".
3. Set the name of the spreadsheet to export to in `~/.my.json` > "exports" > "gsheet" > "name".
4. Set the email address linked to your Drive account in `~/.my.json` > "exports" > "gsheet" > "share_with".

### Running

```
$ me.csv
```
