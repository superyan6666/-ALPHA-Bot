#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

if __name__ == '__main__':
    os.environ.setdefault('GITHUB_EVENT_NAME', 'workflow_dispatch')
    os.environ.setdefault('RUN_MODE', 'normal')
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    exec(open('main.py').read())