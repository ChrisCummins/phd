# Create a minimal Ubuntu environment which is capable of running
# //experimental/deeplearning/deepsmith/opencl_fuzz.
FROM chriscummins/phd_base
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Add and unpack the pre-trained corpus and config.
RUN mkdir -p /data/
ADD clgen.pbtxt /data
ADD clsmith.pbtxt /data
ADD model /data/model

# Add and unpack the CLgen build archive.
ADD opencl_fuzz_image-layer.tar /

ENTRYPOINT ["python", "/app/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image.binary", "--generator=clgen", "--generator_config=/data/clgen.pbtxt", "--interesting_results_dir=/out"]
