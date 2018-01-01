# me.csv - A life in spreadsheets

**Installation:**

```sh
$ virtualenv -p python3.6 build/me
$ source build/me/bin/activate
$ pip install -r requirements.txt
```

**HealthKit:**

1. Select "Export Health Data" on iPhone's Health app.
2. Extract CSVs using:
```sh
$ ./bin/healthkit2csv export.zip outdir
```

**OmniFocus:**

1. Export OmniFocus database to JSON using [of2export](https://github.com/psidnell/ofexport2):
```sh
$ of2 -o omnifocus.json
```
2. Extract CSVs using:
```sh
$ ./bin/omnifocus2csv omnifocus.json outdir
```
