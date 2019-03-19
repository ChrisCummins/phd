# [Buildbot](https://buildbot.net/) for PhD repo

To set up the master:
```sh
$ mkdir -p ~/buildbot
$ cd !$
$ python -m venv buildbot
$ source buildbot/bin/activate
# Now in buildbot virtualenv
$ python -m pip install 'buildbot[bundle]' buildbot-worker
$ buildbot create-master master
$ ln -s $PHD/tools/continuous_integration/buildbot/master/master.cfg ~/buildbot/master/master.cfg
$ buildbot start master
# Go to http://localhost:8010/
```

To set up a worker:
```sh
$ mkdir -p ~/buildbot
$ cd !$
$ buildbot-worker create-worker linux_worker cc3 linux_worker $WORKER_PASSWORD
$ buildbot-worker start linux_worker
```

To run the testlogs viewer:

```sh
$ docker run -p8011:8011 chriscummins/bazel_testlogs_viewer \
    --db=DATABASE --port=8011 \
     --buildbot_url='http://BUILDBOT:8010/#/builders/2' \
     --hostname=$(hostname) 
```

To update the dockerhub testlogs import image:

```sh
$ bazel run //tools/continuous_integration/buildbot/report_generator:image
$ docker tag bazel/tools/continuous_integration/buildbot/report_generator:report_generator_image chriscummins/bazel_testlogs_import
$ docker push chriscummins/bazel_testlogs_import
```

To update the dockerhub testlogs viewer image:

```sh
$ bazel run //tools/continuous_integration/buildbot/report_generator:image
$ docker tag bazel/tools/continuous_integration/buildbot/report_generator:image chriscummins/bazel_testlogs_viewer
$ docker push chriscummins/bazel_testlogs_viewer
```
