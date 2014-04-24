#!/usr/bin/env node

/*
 * watchr.js - Deploy a development server and automatically rebuild
 * sources.
 *
 * Copyright 2014 Chris Cummins.
 *
 * This file is part of pip-db.
 *
 * pip-db is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pip-db is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pip-db.  If not, see <http://www.gnu.org/licenses/>.
 */

var colors = require('colors');
var fs = require('fs');
var path = require('path');
var spawn = require('child_process').spawn;
var util = require('util');
var watch = require('node-watch');

/*
 * Get the project source route.
 */
var getProjectRoot = function(dir) {
  if (dir === '/') {
    console.log('fatal: Unable to locate project base directory!');
    process.exit(3);
  } else {
    if (fs.existsSync(dir + '/configure.ac'))
      return dir;
    else
      return getProjectRoot(path.resolve(dir + '/..'));
  }
};

/* Directories: */
var rootDir = getProjectRoot(__dirname);             // Project root
var srcDirs = [rootDir + '/src', rootDir + '/test']; // Clojure sources
var resourcesDirs = [rootDir + '/resources'];        // Web resources
var reportDir = [rootDir + '/Documentation/report']; // Report

// Print a message
var message = function (msg) {
  if (process.env.EMACS)
    console.log(msg); // Colour deficient Emacs
  else
    console.log(msg.green);
};

// Print an error message
var errorMessage = function (msg) {
  if (process.env.EMACS)
    console.log(msg); // Colour deficient Emacs
  else
    console.log(msg.red);
};

var run = function (cmd, opts) {
  try {
    var worker = spawn(cmd, opts);

    worker.stdout.on('data', function (data) {
      process.stdout.write(data);
    });

    worker.stderr.on('data', function (data) {
      process.stderr.write(data);
    });

    worker.on('exit', function (code) {
      if (code !== 0)
        console.log('Child process exited with code ' + code);
    });
  } catch (err) {
    console.log('error!');
    console.log(err);
  }
};

// Resources modified callback
var resourcesModified = function(filename) {
  message('Rebuilding resources...');
  run('make', ['-s', '-C', 'resources/']);
};

// Source code modified callback
var srcModified = function (filename) {
  message('Restarting server...');
  run('./scripts/run.sh');

  message('Rebuilding documentation...');
  run('make', ['-s', '-C', 'Documentation/']);
};

// Report modified callback
var reportModified = function(filename) {
  if (filename.match(/\.tex$/))
    run('make', ['-s', '-C', 'Documentation/report']);
};

process.chdir(rootDir);

// Register our handlers
watch(resourcesDirs, resourcesModified);
watch(srcDirs, srcModified);
watch(reportDir, reportModified);

// Run the handlers on startup
resourcesModified('');
srcModified('');
reportModified('');
