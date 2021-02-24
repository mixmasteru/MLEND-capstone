# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame(
    {"bruises": pd.Series([False], dtype="bool"), "gill_attachment": pd.Series([False], dtype="bool"),
     "cap_shape_b": pd.Series([0], dtype="int64"), "cap_shape_c": pd.Series([0], dtype="int64"),
     "cap_shape_f": pd.Series([0], dtype="int64"), "cap_shape_k": pd.Series([0], dtype="int64"),
     "cap_shape_s": pd.Series([0], dtype="int64"), "cap_shape_x": pd.Series([0], dtype="int64"),
     "cap_surface_f": pd.Series([0], dtype="int64"), "cap_surface_g": pd.Series([0], dtype="int64"),
     "cap_surface_s": pd.Series([0], dtype="int64"), "cap_surface_y": pd.Series([0], dtype="int64"),
     "cap_color_b": pd.Series([0], dtype="int64"), "cap_color_c": pd.Series([0], dtype="int64"),
     "cap_color_e": pd.Series([0], dtype="int64"), "cap_color_g": pd.Series([0], dtype="int64"),
     "cap_color_n": pd.Series([0], dtype="int64"), "cap_color_p": pd.Series([0], dtype="int64"),
     "cap_color_r": pd.Series([0], dtype="int64"), "cap_color_u": pd.Series([0], dtype="int64"),
     "cap_color_w": pd.Series([0], dtype="int64"), "cap_color_y": pd.Series([0], dtype="int64"),
     "odor_a": pd.Series([0], dtype="int64"), "odor_c": pd.Series([0], dtype="int64"),
     "odor_f": pd.Series([0], dtype="int64"), "odor_l": pd.Series([0], dtype="int64"),
     "odor_m": pd.Series([0], dtype="int64"), "odor_n": pd.Series([0], dtype="int64"),
     "odor_p": pd.Series([0], dtype="int64"), "odor_s": pd.Series([0], dtype="int64"),
     "odor_y": pd.Series([0], dtype="int64"), "gill_spacing_c": pd.Series([0], dtype="int64"),
     "gill_spacing_w": pd.Series([0], dtype="int64"), "gill_size_b": pd.Series([0], dtype="int64"),
     "gill_size_n": pd.Series([0], dtype="int64"), "gill_color_b": pd.Series([0], dtype="int64"),
     "gill_color_e": pd.Series([0], dtype="int64"), "gill_color_g": pd.Series([0], dtype="int64"),
     "gill_color_h": pd.Series([0], dtype="int64"), "gill_color_k": pd.Series([0], dtype="int64"),
     "gill_color_n": pd.Series([0], dtype="int64"), "gill_color_p": pd.Series([0], dtype="int64"),
     "gill_color_r": pd.Series([0], dtype="int64"), "gill_color_u": pd.Series([0], dtype="int64"),
     "gill_color_w": pd.Series([0], dtype="int64"), "gill_color_y": pd.Series([0], dtype="int64"),
     "stalk_shape_e": pd.Series([0], dtype="int64"), "stalk_shape_t": pd.Series([0], dtype="int64"),
     "stalk_root_?": pd.Series([0], dtype="int64"), "stalk_root_b": pd.Series([0], dtype="int64"),
     "stalk_root_c": pd.Series([0], dtype="int64"), "stalk_root_e": pd.Series([0], dtype="int64"),
     "stalk_root_r": pd.Series([0], dtype="int64"), "stalk_surface_above_ring_f": pd.Series([0], dtype="int64"),
     "stalk_surface_above_ring_k": pd.Series([0], dtype="int64"),
     "stalk_surface_above_ring_s": pd.Series([0], dtype="int64"),
     "stalk_surface_above_ring_y": pd.Series([0], dtype="int64"),
     "stalk_surface_below_ring_f": pd.Series([0], dtype="int64"),
     "stalk_surface_below_ring_k": pd.Series([0], dtype="int64"),
     "stalk_surface_below_ring_s": pd.Series([0], dtype="int64"),
     "stalk_surface_below_ring_y": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_b": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_c": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_e": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_g": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_n": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_p": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_w": pd.Series([0], dtype="int64"),
     "stalk_color_above_ring_y": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_b": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_c": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_e": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_g": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_n": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_p": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_w": pd.Series([0], dtype="int64"),
     "stalk_color_below_ring_y": pd.Series([0], dtype="int64"), "veil_type_p": pd.Series([0], dtype="int64"),
     "veil_color_w": pd.Series([0], dtype="int64"), "veil_color_y": pd.Series([0], dtype="int64"),
     "ring_number_n": pd.Series([0], dtype="int64"), "ring_number_o": pd.Series([0], dtype="int64"),
     "ring_number_t": pd.Series([0], dtype="int64"), "ring_type_e": pd.Series([0], dtype="int64"),
     "ring_type_f": pd.Series([0], dtype="int64"), "ring_type_l": pd.Series([0], dtype="int64"),
     "ring_type_n": pd.Series([0], dtype="int64"), "ring_type_p": pd.Series([0], dtype="int64"),
     "spore_print_color_h": pd.Series([0], dtype="int64"), "spore_print_color_k": pd.Series([0], dtype="int64"),
     "spore_print_color_n": pd.Series([0], dtype="int64"), "spore_print_color_r": pd.Series([0], dtype="int64"),
     "spore_print_color_u": pd.Series([0], dtype="int64"), "spore_print_color_w": pd.Series([0], dtype="int64"),
     "population_a": pd.Series([0], dtype="int64"), "population_c": pd.Series([0], dtype="int64"),
     "population_n": pd.Series([0], dtype="int64"), "population_s": pd.Series([0], dtype="int64"),
     "population_v": pd.Series([0], dtype="int64"), "population_y": pd.Series([0], dtype="int64"),
     "habitat_d": pd.Series([0], dtype="int64"), "habitat_g": pd.Series([0], dtype="int64"),
     "habitat_l": pd.Series([0], dtype="int64"), "habitat_m": pd.Series([0], dtype="int64"),
     "habitat_p": pd.Series([0], dtype="int64"), "habitat_u": pd.Series([0], dtype="int64"),
     "habitat_w": pd.Series([0], dtype="int64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    json.dump()
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
