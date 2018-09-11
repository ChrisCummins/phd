.PHONY: test

venv3.6_activate = python/3.6/bin/activate
venv3.6 = source $(venv3.6_activate) &&

$(venv3.6_activate):
	virtualenv -p python3.6 python/3.6

test: $(venv3.6_activate)
	mkdir -pv build/3.6
	$(venv3.6) pip install -r requirements.txt
	$(venv3.6) protoc -I=. --python_out=freefocus freefocus.proto
