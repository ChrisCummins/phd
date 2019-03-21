#!/usr/bin/env bash
#
# Start the testlogs viewer.
set -eux

docker pull chriscummins/bazel_testlogs_viewer:latest
docker run \
  --detach \
  -p8011:8011 \
  -v/var/phd/db:/var/phd/db \
  -v/var/phd/shared/tools/continuous_integration/buildbot/coverage:/coverage \
  chriscummins/bazel_testlogs_viewer \
  --db='file:///var/phd/db/cc1.mysql?buildbot_linux_cpu_phd_priv?charset=utf8' \
  --port=8011 \
  --buildbot_url='http://cc1.inf.ed.ac.uk:8010/#/builders/2' \
  --hostname='cc1.inf.ed.ac.uk' \
  --nodebug_flask_server
