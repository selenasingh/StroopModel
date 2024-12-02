import numpy as np
import psyneulink as pnl
from setup import model_parameters, weights

# This implements the model by Cohen, J. D., & Huston, T. A. (1994). Progress in the use of interactive
# models for understanding attention and performance. In C. Umilta & M. Moscovitch(Eds.),
# AttentionandperformanceXV(pp.453-456). Cam- bridge, MA: MIT Press.
# The model aims to capute top-down effects of selective attention and the bottom-up effects of attentional capture.

#Modified to include emotional processing: Benjamin Li and Selena Singh, 2024

# Create mechanisms ---------------------------------------------------------------------------------------------------
#   Linear input units, colors: ('red', 'green'), words: ('RED','GREEN')
colors_input_layer = pnl.TransferMechanism(size=3, function=pnl.Linear, name='COLORS_INPUT')
words_input_layer = pnl.TransferMechanism(size=3, function=pnl.Linear, name='WORDS_INPUT')
task_input_layer = pnl.TransferMechanism(size=3, function=pnl.Linear, name='TASK_INPUT')
emotion_input_layer = pnl.TransferMechanism(size=3, function=pnl.Linear, name='EMOTION_INPUT')

# TASK LAYERS --------------------------------------------------------------------------------------------------------
#   Task layer, tasks: ('name the color', 'read the word') 
task_layer = pnl.RecurrentTransferMechanism(
    size=3, 
    function=pnl.Logistic(gain = model_parameters['task_layer']['gain'], 
                          x_0 = model_parameters['task_layer']['bias']),
    hetero=model_parameters['task_layer']['inhibition'],
    integrator_mode=True,
    integration_rate=model_parameters['task_layer']['rate'],
    name='TASK'
)

#   Hidden layer units, colors: ('red','green') words: ('RED','GREEN')
colors_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,
    function=pnl.Logistic(gain = model_parameters['colour_hidden']['gain'], 
                          x_0 = model_parameters['colour_hidden']['bias']),
    hetero=model_parameters['colour_hidden']['inhibition'],
    integrator_mode=True,
    integration_rate=model_parameters['colour_hidden']['rate'],
    name='COLORS HIDDEN'
)

words_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,
    function=pnl.Logistic(gain = model_parameters['word_hidden']['gain'], 
                          x_0 = model_parameters['word_hidden']['bias']),
    hetero=model_parameters['word_hidden']['inhibition'],
    integrator_mode=True,
    integration_rate=model_parameters['word_hidden']['rate'],
    name='WORDS HIDDEN'
)

emotion_hidden_layer = pnl.RecurrentTransferMechanism(
    size=3,
    function=pnl.Logistic(gain = model_parameters['emotion_hidden']['gain'], 
                          x_0 = model_parameters['emotion_hidden']['bias']),
    hetero=model_parameters['emotion_hidden']['inhibition'],
    integrator_mode=True,
    integration_rate=model_parameters['emotion_hidden']['rate'],
    name='EMOTION HIDDEN'
)

# RESPONSE LAYER ---------------------------------------------------------------------------------------------------------
#   Response layer, responses: ('red', 'green'): RecurrentTransferMechanism for self inhibition matrix
response_layer = pnl.RecurrentTransferMechanism(
    size=2, 
    function=pnl.Logistic(gain = model_parameters['response_layer']['gain'], 
                          x_0 = model_parameters['response_layer']['bias']),
    hetero=model_parameters['response_layer']['inhibition'],
    integrator_mode=True,
    integration_rate=model_parameters['response_layer']['rate'],
    name='RESPONSE'
)

# Connect mechanisms --------------------------------------------------------------------------------------------------
# (note that response layer projections are set to all zero first for initialization
# INPUTS TO LAYERS
color_input_weights = pnl.MappingProjection(matrix=weights['colour_input'])
word_input_weights = pnl.MappingProjection(matrix=weights['word_input'])
emotion_input_weights = pnl.MappingProjection(matrix=weights['emotion_input'])
task_input_weights = pnl.MappingProjection(matrix=weights['task_input'])

# LAYERS TO TASK, TASK TO LAYERS

color_task_weights = pnl.MappingProjection(matrix=weights['colour_task'])
task_color_weights = pnl.MappingProjection(matrix=weights['task_colour'])
word_task_weights = pnl.MappingProjection(matrix=weights['word_task'])
task_word_weights = pnl.MappingProjection(matrix=weights['task_word'])
emotion_task_weights = pnl.MappingProjection(matrix=weights['emotion_task'])
task_emotion_weights = pnl.MappingProjection(matrix=weights['task_emotion'])

# response weights
response_color_weights = pnl.MappingProjection(matrix=weights['response_colour'])
response_word_weights = pnl.MappingProjection(matrix=weights['response_word'])
response_emotion_weights = pnl.MappingProjection(matrix=weights['response_emotion'])
color_response_weights = pnl.MappingProjection(matrix=weights['colour_response'])
word_response_weights = pnl.MappingProjection(matrix=weights['word_response'])
emotion_response_weights = pnl.MappingProjection(matrix=weights['emotion_response'])

# Create pathways -----------------------------------------------------------------------------------------------------
# response pathways
color_response_process_1 = pnl.Pathway(
    pathway=[
        colors_input_layer,
        color_input_weights,
        colors_hidden_layer,
        color_response_weights,
        response_layer
    ],
    name='COLORS_RESPONSE_PROCESS_1'
)

color_response_process_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_color_weights,
        colors_hidden_layer
    ],
    name='COLORS_RESPONSE_PROCESS_2'
)

word_response_process_1 = pnl.Pathway(
    pathway=[
        words_input_layer,
        word_input_weights,
        words_hidden_layer,
        word_response_weights,
        response_layer
    ],
    name='WORDS_RESPONSE_PROCESS_1'
)

word_response_process_2 = pnl.Pathway(
    pathway=[
        (response_layer, pnl.NodeRole.OUTPUT), # this is related to logging the output results... probably
        response_word_weights,
        words_hidden_layer
    ],
    name='WORDS_RESPONSE_PROCESS_2'
)

# NOTE: emotion layers
emotion_response_process_1 = pnl.Pathway(
    pathway=[
        emotion_input_layer,
        emotion_input_weights,
        emotion_hidden_layer,
        emotion_response_weights,
        response_layer # NOTE: there was a comma at the end here, but I removed it. See if model breaks
    ],
    name='EMOTION_RESPONSE_PROCESS_1'
)

# NOTE:
emotion_response_process_2 = pnl.Pathway(
    pathway=[
        response_layer,
        response_emotion_weights,
        emotion_hidden_layer
    ],
    name='EMOTION_RESPONSE_PROCESS_2'
)

# task pathways
task_color_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_input_weights,
        task_layer,
        task_color_weights,
        colors_hidden_layer])

task_color_response_process_2 = pnl.Pathway(
    pathway=[
        colors_hidden_layer,
        color_task_weights,
        task_layer])

task_word_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_layer,
        task_word_weights,
        words_hidden_layer])

task_word_response_process_2 = pnl.Pathway(
    pathway=[
        words_hidden_layer,
        word_task_weights,
        task_layer])

task_emotion_response_process_1 = pnl.Pathway(
    pathway=[
        task_input_layer,
        task_input_weights,
        task_layer,
        task_emotion_weights,
        emotion_hidden_layer])

task_emotion_response_process_2 = pnl.Pathway(
    pathway=[
        emotion_hidden_layer,
        emotion_task_weights,
        task_layer])

# Create Composition --------------------------------------------------------------------------------------------------
Bidirectional_Stroop = pnl.Composition(
    pathways=[
        color_response_process_1,
        word_response_process_1,
        emotion_response_process_1,
        task_color_response_process_1,
        task_word_response_process_1,
        task_emotion_response_process_1,
        color_response_process_2,
        word_response_process_2,
        emotion_response_process_2,
        task_color_response_process_2,
        task_word_response_process_2,
        task_emotion_response_process_2
    ],
    reinitialize_mechanisms_when=pnl.Never(),
    name='Bidirectional Stroop Model'
)
