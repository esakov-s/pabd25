#!/usr/bin/python3
import sys
import logging
logging.basicConfig(stream=sys.stderr)

from service.app import app as application
app = application