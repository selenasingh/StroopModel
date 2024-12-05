import numpy as np
import psyneulink as pnl
from setup import model_parameters, weights
import functools
from EmotionalGrain import colors_input_layer, words_input_layer, task_input_layer, emotion_input_layer, task_layer
from EmotionalGrain import colors_hidden_layer, words_hidden_layer, emotion_hidden_layer, response_layer 
from EmotionalGrain import color_input_weights, word_input_weights, emotion_input_weights, task_input_weights
from EmotionalGrain import color_task_weights, task_color_weights, word_task_weights, emotion_task_weights, task_emotion_weights 
from EmotionalGrain import response_color_weights, response_word_weights, response_emotion_weights, color_response_weights, word_response_weights 
from EmotionalGrain import emotion_response_weights, color_response_process_1, color_response_process_2, word_response_process_1, word_response_process_2
from EmotionalGrain import emotion_response_process_1, emotion_response_process_2, task_color_response_process_1, task_color_response_process_2
from EmotionalGrain import task_word_response_process_1, task_word_response_process_2, task_emotion_response_process_1, task_emotion_response_process_2

#from run_exp import colour_naming_stimuli

# Create Composition --------------------------------------------------------------------------------------------------
Bidirectional_Stroop_train = pnl.Composition(
    pathways=[
        (color_response_process_1, pnl.BackPropagation),
        (word_response_process_1, pnl.BackPropagation),
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
    #targets = [1,0],
    learning_rate = 5.0,
    reinitialize_mechanisms_when=pnl.Never(),
    name='Bidirectional Stroop Model'
)

stim_list = {colors_input_layer: [1, 0, 0], 
              words_input_layer: [0, 1, 0],
              emotion_input_layer: [0, 0, 0], 
              task_input_layer: [1, 0, 0]} 


target_list = {response_layer: [[0.6, 0.3]]}

color_input_weights.set_log_conditions('mod_matrix')
color_response_weights.set_log_conditions('mod_matrix')
word_input_weights.set_log_conditions('mod_matrix')
word_response_weights.set_log_conditions('mod_matrix')

def print_header(Bidirectional_Stroop_train, context):
    print("\n\n**** Time: ", Bidirectional_Stroop_train.scheduler.get_clock(context).simple_time)

 
def show_target(Bidirectional_Stroop_train):
    i = Bidirectional_Stroop_train.external_input_values
    t = Bidirectional_Stroop_train.pathways[0].target.input_ports[0].parameters.value.get(Bidirectional_Stroop_train)
    print('- Output:\n', response_layer.parameters.value.get(Bidirectional_Stroop_train))

 

Bidirectional_Stroop_train.learn(
    num_trials=50,
    inputs=stim_list,
    targets=target_list,
    clamp_input=pnl.SOFT_CLAMP,
    call_before_trial=functools.partial(print_header, Bidirectional_Stroop_train),
    call_after_trial=functools.partial(show_target, Bidirectional_Stroop_train),
    termination_processing={pnl.TimeScale.TRIAL: pnl.AfterNCalls(response_layer, 150)},
)

#print(color_response_weights.log.nparray(entries='mod_matrix', header=True))
print(word_response_weights.log.nparray(entries='mod_matrix', header=True))