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
