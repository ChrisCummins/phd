CLSmith/build/cl_launcher:
	mkdir -pv CLSmith/build
	cd CLSmith/build && cmake ..
	cd CLSmith/build && make -j$(shell nproc)
	cp -v CLSmith/build/*.h CLSmith/runtime

clean:
	rm -rf CLSmith/build
