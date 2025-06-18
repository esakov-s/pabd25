#!/usr/bin/python3
import sys
import logging
import joblib
from service.app import app as application
logging.basicConfig(stream=sys.stderr)
model = 'models/decision_tree_reg_1.pkl'

app = application
app.config["model"] = joblib.load(model)
app.logger.info(f"Use model: {model}")
