#!/bin/sh
ps -ef |grep Quad | grep -v grep | awk '{print $2}' | xargs kill -9