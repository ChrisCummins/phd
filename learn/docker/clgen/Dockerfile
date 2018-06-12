FROM ubuntu:18.04
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Set and configure the locale. This is necessary for CLgen, which uses en_GB.
RUN apt-get update && \
    apt-get install -y --no-install-recommends language-pack-en
RUN locale-gen --purge en_GB.UTF-8

# Install Python 3.6 and symlink it to default python version.
RUN apt-get update && \
    apt-get install -y python3.6 && \
    ln -s /usr/bin/python3.6 /usr/bin/python

# Add and unpack the binary archives.
RUN mkdir -p /usr/local/opt/clgen
ADD clang_test.tar.bz2 /usr/local/opt/clgen/

# Add the tiny corpus.
RUN mkdir /tmp/tiny
ADD corpus.tar.bz2 /tmp/tiny
ADD config.pbtxt /tmp/tiny

WORKDIR /usr/local/opt/clgen
