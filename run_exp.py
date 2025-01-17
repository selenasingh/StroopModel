#run experiments 
import argparse
import numpy as np
import pandas as pd
import psyneulink as pnl

import logging
from EmotionalGrain import colors_input_layer, words_input_layer, task_input_layer, emotion_input_layer                          
from EmotionalGrain import task_layer, colors_hidden_layer, words_hidden_layer, emotion_hidden_layer, response_layer
from EmotionalGrain import response_color_weights, response_word_weights, response_emotion_weights, color_response_weights, word_response_weights, emotion_response_weights
from EmotionalGrain import Bidirectional_Stroop
from setup import model_parameters, exp_weights_reset, colour_naming_exp

parser = argparse.ArgumentParser()
parser.add_argument('--no-plot', action='store_false', help='Disable plotting', dest='enable_plot')
parser.add_argument('--threshold', type=float, help='Termination threshold for response output (default: %(default)f)', default=0.55)
parser.add_argument('--settle-trials', type=int, help='Number of trials for composition to initialize and settle (default: %(default)d)', default=50)
args = parser.parse_args()
settle_trials = args.settle_trials  # cycles until model settles

# Log mechanisms ------------------------------------------------------------------------------------------------------
task_layer.set_log_conditions('value')
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
emotion_hidden_layer.set_log_conditions('value') # NOTE: I have no idea what this does
response_layer.set_log_conditions('value')

input_dict = {colors_input_layer: [0, 0, 0],
              words_input_layer: [0, 0, 0],
              emotion_input_layer: [0, 0, 0], # NOTE: added emotion input to input dict. 
              task_input_layer: [0, 1, 0]} # NOTE: added extra emotion task input.
                # I believe the 1 would indiciate what task is being done. With color first, word second, emotion third
                # Similarly, I believe that the other inputs layers correspond to the condition (ex: negative, positive, congruent,incongruent)
#print("\n\n\n\n")
#print(Bidirectional_Stroop.run(inputs=input_dict))

#for node in Bidirectional_Stroop.mechanisms:
#    print(node.name, " Value: ", node.get_output_values(Bidirectional_Stroop))


# # LOGGING:
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
emotion_hidden_layer.set_log_conditions('value') # NOTE: added emotion layer logging.

# Create threshold function -------------------------------------------------------------------------------------------

terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.Or(
        pnl.Threshold(response_layer, 'value', model_parameters['threshold'], '>=', (0, 0)),
        pnl.Threshold(response_layer, 'value', model_parameters['threshold'], '>=', (0, 1)),
    )
}

# Create test trials function -----------------------------------------------------------------------------------------
# a BLUE word input is [1,0] to words_input_layer and GREEN word is [0,1]
# a blue color input is [1,0] to colors_input_layer and green color is [0,1]
# a color-naming trial is [1,0] to task_layer and a word-reading trial is [0,1]

def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, positive_emotion, negative_emotion, neutral_emotion, CN, WR, EP):
# CN = colour naming, WR = word reading, EP = emotion processing. 
    trialdict = {
        colors_input_layer: [red_color, green_color, neutral_color],
        words_input_layer: [red_word, green_word, neutral_word],
        emotion_input_layer: [positive_emotion, negative_emotion, neutral_emotion],
        task_input_layer: [CN, WR, EP]
    }
    return trialdict

# Define initialization trials separately
# order: red_color, green_color, neutral_color, red_word, green_word, neutral_word, positive_emotion, negative_emotion, neutral_emotion, CN, WR, EP
CN_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
WR_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
EP_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

# red_color, green_color, neutral_color, red_word, green_word, neutral_word, positive_emotion, negative_emotion, neutral_emotion, CN, WR, EP
CN_congruent_trial_input =   trial_dict(1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0) # red_colour + red_word
CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0) # red_colour + green_word
CN_control_trial_input =     trial_dict(1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0) # red_colour + no word (?)

WR_congruent_trial_input =   trial_dict(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0) # red_color + red_word
WR_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0) # red_colour + green_word
WR_control_trial_input =     trial_dict(0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0) # no color? + red word 

CN_positive_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)
CN_negative_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0)
CN_neutral_trial_input =  trial_dict(1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)

colour_naming_stimuli = [[CN_initialize_input, CN_control_trial_input],
                        [CN_initialize_input, CN_incongruent_trial_input],
                        [CN_initialize_input, CN_congruent_trial_input]]

word_reading_stimuli = [[WR_initialize_input, WR_control_trial_input],
                       [WR_initialize_input, WR_incongruent_trial_input],
                       [WR_initialize_input, WR_control_trial_input]]
             
emotion_colour_naming_stimuli = [[CN_initialize_input, CN_negative_trial_input],
                                [CN_initialize_input, CN_neutral_trial_input],
                                [CN_initialize_input, CN_positive_trial_input]]


# Create third stimulus? Technically we would only have colour naming trials to begin with. So I guess a third stimulus except it would be a colour naming one with emotional words activated, but not the actual task node.
    # It would be like the control CN task, but instead the emotional words are used.
conditions = 3
response_colournaming = []
response_colournaming2 = [] #what is the point of this??
# Run color naming trials ----------------------------------------------------------------------------------------------
for cond in range(conditions):

    #re-initialize weights to response layer 
    response_color_weights.parameters.matrix.set(exp_weights_reset['response_colour'], Bidirectional_Stroop)
    response_word_weights.parameters.matrix.set(exp_weights_reset['response_word'], Bidirectional_Stroop)
    response_emotion_weights.parameters.matrix.set(exp_weights_reset['response_emotion'], Bidirectional_Stroop)
    
    #run baseline
    Bidirectional_Stroop.run(inputs=colour_naming_stimuli[cond][0], num_trials=settle_trials)

    #change weights for experiment
    #task_layer.parameters.function.set(pnl.Logistic(gain = 0.5), Bidirectional_Stroop)
    task_layer.parameters.integration_rate.set(0.001, Bidirectional_Stroop)
    response_color_weights.parameters.matrix.set(colour_naming_exp['response_colour'], Bidirectional_Stroop)
    response_word_weights.parameters.matrix.set(colour_naming_exp['response_word'], Bidirectional_Stroop)
    response_emotion_weights.parameters.matrix.set(colour_naming_exp['response_emotion'], Bidirectional_Stroop)

    #run exp
    Bidirectional_Stroop.run(inputs=colour_naming_stimuli[cond][1], termination_processing=terminate_trial)

    # Store values from run -----------------------------------------------------------------------------------------------
    B_S = Bidirectional_Stroop.name
    r = response_layer.log.nparray_dictionary('value')      
    rr = r[B_S]['value']
    print(rr)
    n_r = rr.shape[0]
    #print(n_r)
    rrr = rr.reshape(n_r, 2) 
    #print(rrr)
 
    response_colournaming.append(rrr) 
    response_colournaming2.append(rrr.shape[0]) 

    # Clear log & reset ----------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    colors_hidden_layer.log.clear_entries()
    words_hidden_layer.log.clear_entries()
    emotion_hidden_layer.log.clear_entries() # NOTE: Clear emotion hidden layer logs
    task_layer.log.clear_entries()
    colors_hidden_layer.reset([[0, 0, 0]]) 
    words_hidden_layer.reset([[0, 0, 0]])
    emotion_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0, 0]]) # NOTE: task layer reset needs 3 nodes now.
    #print('response_colournaming: ', response_colournaming)
    #print('first trials')

"""
# Run color naming trials ----------------------------------------------------------------------------------------------
response_wordreading = []
response_wordreading2 = []
print('made the next responses')
for cond in range(conditions):
    #re-initialize weights to response layer 
    response_color_weights.parameters.matrix.set(exp_weights_reset['response_colour'], Bidirectional_Stroop)
    response_word_weights.parameters.matrix.set(exp_weights_reset['response_word'], Bidirectional_Stroop)
    response_emotion_weights.parameters.matrix.set(exp_weights_reset['response_emotion'], Bidirectional_Stroop)
    
    #run baseline
    Bidirectional_Stroop.run(inputs=word_reading_stimuli[cond][0], num_trials=settle_trials)

    #change weights for experiment
    response_color_weights.parameters.matrix.set(colour_naming_exp['response_colour'], Bidirectional_Stroop)
    response_word_weights.parameters.matrix.set(colour_naming_exp['response_word'], Bidirectional_Stroop)
    response_emotion_weights.parameters.matrix.set(colour_naming_exp['response_emotion'], Bidirectional_Stroop)

    #run exp
    Bidirectional_Stroop.run(inputs=word_reading_stimuli[cond][0], termination_processing=terminate_trial)

    # Store values from run -----------------------------------------------------------------------------------------------
    r2 = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
    rr2 = r2[Bidirectional_Stroop.name]['value']
    n_r2 = rr2.shape[0]
    rrr2 = rr2.reshape(n_r2, 2)
    response_wordreading.append(rrr2)  # .shape[0])
    response_wordreading2.append(rrr2.shape[0])

    # Clear log & reset ----------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    colors_hidden_layer.log.clear_entries()
    words_hidden_layer.log.clear_entries()
    emotion_hidden_layer.log.clear_entries() # NOTE: Clear emotion hidden layer logs
    task_layer.log.clear_entries()
    colors_hidden_layer.reset([[0, 0, 0]])
    words_hidden_layer.reset([[0, 0, 0]])
    emotion_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0, 0]]) # NOTE: again, 3 nodes now
    print('response_wordreading: ', response_wordreading)
    print('got to second trials')

"""
# Run color naming with emotion ----------------------------------------------------------------------------------------------

response_colouremotion = []
response_colouremotion2 = []
print('made the next responses')
for cond in range(conditions):
    
    #re-initialize weights to response layer 
    #task_layer.parameters.function.set(pnl.Logistic(gain = model_parameters['task_layer']['gain']), Bidirectional_Stroop)
    response_color_weights.parameters.matrix.set(exp_weights_reset['response_colour'], Bidirectional_Stroop)
    response_word_weights.parameters.matrix.set(exp_weights_reset['response_word'], Bidirectional_Stroop)
    response_emotion_weights.parameters.matrix.set(exp_weights_reset['response_emotion'], Bidirectional_Stroop)
    
    #run baseline
    Bidirectional_Stroop.run(inputs=emotion_colour_naming_stimuli[cond][0], num_trials=settle_trials)

    #TEST PARAMETERS/SELECTION
    #task_layer.parameters.function.set(pnl.Logistic(gain = 4), Bidirectional_Stroop)
    #task_layer.parameters.hetero.set(-4, Bidirectional_Stroop)

    #
    response_color_weights.parameters.matrix.set(colour_naming_exp['response_colour'], Bidirectional_Stroop)
    response_word_weights.parameters.matrix.set(colour_naming_exp['response_word'], Bidirectional_Stroop)
    response_emotion_weights.parameters.matrix.set(colour_naming_exp['response_emotion'], Bidirectional_Stroop)

    #run exp
    Bidirectional_Stroop.run(inputs=emotion_colour_naming_stimuli[cond][1], termination_processing=terminate_trial)

    # Store values from run -----------------------------------------------------------------------------------------------
    r3 = response_layer.log.nparray_dictionary('value')       # Log response output from special logistic function
    rr3 = r3[Bidirectional_Stroop.name]['value']
    n_r3 = rr3.shape[0]
    rrr3 = rr3.reshape(n_r3, 2)
    response_colouremotion.append(rrr3)  # .shape[0])
    response_colouremotion2.append(rrr3.shape[0])

    # Clear log & reset ----------------------------------------------------------------------------------------
    response_layer.log.clear_entries()
    colors_hidden_layer.log.clear_entries()
    words_hidden_layer.log.clear_entries()
    emotion_hidden_layer.log.clear_entries() # NOTE: Clear emotion hidden layer logs
    task_layer.log.clear_entries()
    colors_hidden_layer.reset([[0, 0, 0]])
    words_hidden_layer.reset([[0, 0, 0]])
    emotion_hidden_layer.reset([[0, 0, 0]])
    response_layer.reset([[0, 0]])
    task_layer.reset([[0, 0, 0]]) # NOTE: again, 3 nodes now
    print('response_colouremotion: ', response_colouremotion)
    print('got to third trials')


print('now we plot')
if args.enable_plot:
    import matplotlib.pyplot as plt
    # Plot results --------------------------------------------------------------------------------------------------------
    reg = np.dot(response_colournaming2, 5) + 115
    reg2 = np.dot(response_colournaming2, 5) + 115
    print(response_colournaming2)

    plt.figure()
    plt.rcParams["font.family"] = "Times"
    plt.rcParams["font.size"] = 20
    #plt.rcParams['figure.figsize'] = [10,12]
    plt.bar(["Incongruent", "Congruent"], reg[1:3], color = "gray")
    plt.ylabel('Reaction Time (ms)')
    #plt.ylim([500, 1500])
    plt.savefig("figures/standard_stroop/standard_TEST_IR_0.001.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Show emotional graph
    reg3 = np.dot(response_colouremotion2, 5) + 115
    plt.bar(["Negative", "Neutral", "Positive"], reg3, color = "gray")
    plt.xlabel('Valence')
    #plt.title('Simulated GRAIN data')
    #plt.xticks(np.arange(3), ('Negative', 'Neutral', 'Positive'))
    plt.ylabel('Reaction Time (ms)')
    #plt.ylim([500, 565])
    #plt.ylim([500, 1500])
    plt.savefig("figures/emotion_stroop/emotional_TEST_1.png", dpi=300, bbox_inches="tight")
    plt.close()

    ## datasaving 
    
    simulation_data = {
        'congruent': reg[2],
        'incongruent': reg[1],
        'negative': reg3[0],
        'neutral': reg3[1],
        'positive': reg3[2]
    }

    print(simulation_data)

    df = pd.DataFrame(simulation_data, index=[0]) 

    df.to_csv('simresults/baseline_IR_0.001.csv')
