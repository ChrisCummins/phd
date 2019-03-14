#!/usr/bin/env node

/*
 * watchr.js - Automatically rebuild sources.
 *
 * Copyright 2014,2015 Chris Cummins.
 *
 * Forked from pip-db <https://github.com/ChrisCummins/pip-db>.
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

// Some file which is unique to the project root:
var ROOT_MARKER = "/.travis.yml";

// FIXME: Global mutable state - bad idea!
global.run_lock = false;

/*
 * Get the project source route.
 */
var getProjectRoot = function(dir) {
  if (dir === '/') {
    console.log('fatal: Unable to locate project base directory!');
    process.exit(3);
  } else {
    if (fs.existsSync(dir + ROOT_MARKER)) {
      return dir;
    } else {
      return getProjectRoot(path.resolve(dir + '/..'));
    }
  }
};

// Print a message
var message = function(msg) {
  if (process.env.EMACS) {
    console.log(msg);
  } // Colour deficient Emacs
  else {
    console.log(msg.green);
  }
};

// Print an error message
var errorMessage = function(msg) {
  if (process.env.EMACS) {
    console.log(msg);
  } // Colour deficient Emacs
  else {
    console.log(msg.red);
  }
};

var run = function(cmd, opts) {
  // If this is locked, do nothing.
  if (global.run_lock) {
    return;
  }

  try {
    // Lock:
    global.run_lock = true;
    var worker = spawn(cmd, opts);

    worker.stdout.on('data', function(data) {
      process.stdout.write(data);
    });

    worker.stderr.on('data', function(data) {
      process.stderr.write(data);
    });

    worker.on('exit', function(code) {
      // Unlock global state:
      global.run_lock = false;
      if (code !== 0) {
        console.log('Child process exited with code ' + code);
      }
    });
  } catch (err) {
    // Unlock global state:
    global.run_lock = false;
    console.log('error!');
    console.log(err);
  }
};

// Determine if file is ignored
var ignoredFile = function(filename) {
  return filename.match(/\/.git\//);
}

// Source code modified callback
var fileModified = function(filename) {
  if (!ignoredFile(filename)) {
    run('pmake');
  }
};

// Register our handler:
watch(getProjectRoot(__dirname), fileModified);
// Run the handler on startup:
fileModified('');