#!/bin/bash
# Daemon: refresh the evaluated-checkpoint feed every 60 s. Create /root/probes/FEED_STOP to end.
while [ ! -f /root/probes/FEED_STOP ]; do /root/venv-cpu/bin/python /root/probes/checkpoint_feed.py >> /root/probes/checkpoint_feed.log 2>&1; sleep 60; done
