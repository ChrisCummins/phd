# Augment the standard base environment with Java.
FROM chriscummins/phd_base:latest
MAINTAINER Chris Cummins <chrisc.101@gmail.com>

# Switch back to root user to do the installation admin.
USER root

# Workaround for broken Java install.
RUN mkdir -p /usr/share/man/man1

# Install Java.
RUN apt-get update && apt-get install -y --no-install-recommends \
  openjdk-8-jdk

# Switch back to user account.
USER docker

ENTRYPOINT ["/bin/bash"]
