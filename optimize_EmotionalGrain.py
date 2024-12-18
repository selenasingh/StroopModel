import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import Random
from inspyred import ec 

import argparse
import psyneulink as pnl
from setup import model_parameters, weights, exp_weights_reset, colour_naming_exp
from EmotionalGrain import colors_input_layer, words_input_layer, task_input_layer, emotion_input_layer, task_layer
from EmotionalGrain import colors_hidden_layer, words_hidden_layer, emotion_hidden_layer, response_layer 
from EmotionalGrain import color_input_weights, word_input_weights, emotion_input_weights, task_input_weights
from EmotionalGrain import color_task_weights, task_color_weights, task_word_weights, word_task_weights, emotion_task_weights, task_emotion_weights 
from EmotionalGrain import response_color_weights, response_word_weights, response_emotion_weights, color_response_weights, word_response_weights 
from EmotionalGrain import emotion_response_weights, color_response_process_1, color_response_process_2, word_response_process_1, word_response_process_2
from EmotionalGrain import emotion_response_process_1, emotion_response_process_2, task_color_response_process_1, task_color_response_process_2
from EmotionalGrain import task_word_response_process_1, task_word_response_process_2, task_emotion_response_process_1, task_emotion_response_process_2
from EmotionalGrain import Bidirectional_Stroop

#PARAMETER FITTING TO PARTICIPANT DATA

# import data
standard = pd.read_csv("data/compiled_1.csv")
n_participants = len(standard['prolific_ID'])

# ------------------------------MODELLING-----------------------------------------------------------
"""
TODO: Clean this up somehow.
"""
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
emotion_hidden_layer.set_log_conditions('value')
response_layer.set_log_conditions('value')

input_dict = {colors_input_layer: [0, 0, 0],
              words_input_layer: [0, 0, 0],
              emotion_input_layer: [0, 0, 0], # NOTE: added emotion input to input dict. 
              task_input_layer: [0, 1, 0]} # NOTE: added extra emotion task input.

# # LOGGING:
colors_hidden_layer.set_log_conditions('value')
words_hidden_layer.set_log_conditions('value')
emotion_hidden_layer.set_log_conditions('value') # NOTE: added emotion layer logging.

terminate_trial = {
    pnl.TimeScale.TRIAL: pnl.Or(
        pnl.Threshold(response_layer, 'value', model_parameters['threshold'], '>=', (0, 0)),
        pnl.Threshold(response_layer, 'value', model_parameters['threshold'], '>=', (0, 1)),
    )
}

# Create test trials function -----------------------------------------------------------------------------------------
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

CN_positive_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0)
CN_negative_trial_input = trial_dict(1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0)
CN_neutral_trial_input =  trial_dict(1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0)

colour_naming_stimuli = [[CN_initialize_input, CN_control_trial_input],
                        [CN_initialize_input, CN_incongruent_trial_input],
                        [CN_initialize_input, CN_congruent_trial_input]]
             
emotion_colour_naming_stimuli = [[CN_initialize_input, CN_negative_trial_input],
                                [CN_initialize_input, CN_neutral_trial_input],
                                [CN_initialize_input, CN_positive_trial_input]]

free_params = {
}

############################################# MODEL FITTING ################################################################
class optimize_stroop(object):
    def __init__(self, 
                 data, 
                 counterbalance,
                 participant,
                 pop_size = 10,
                 max_evaluations = 150,
                 num_selected = 10, 
                 mutation_rate = 0.02,
                ):
        self.data = data
        self.counterbalance = counterbalance,
        self.participant = participant,
        self.pop_size = pop_size
        self.max_evaluations = max_evaluations
        self.num_selected = num_selected
        self.mutation_rate = mutation_rate
        self.num_elites = 1
        self.flag = str('pt-') + str(self.participant[0]) + str('_cb-') + str(self.counterbalance[0])
        self.minParamValues = []
        self.maxParamValues = []
        self.num_inputs = 9
        self.initialParams = []
        #self.param = [colour_naming_exp['response_colour'][0,0], #baseline 1.5
        #              colour_naming_exp['response_word'][0,0], #baseline 2.5
         #             weights['colour_response'][0,0],
         #             weights['word_response'][0,0]
         #             ]
        #self.sd = [0.5, 0.5, 0.5, 0.5]
        #self.initialParams = [np.random.normal(self.param[i], self.sd[i]) for i in range(self.num_inputs)]

        self.plot_results()

    def extract_congruent(self): 
        self.data_congruent = self.data.loc[self.participant]['congruent']
        return self.data_congruent 

    def extract_incongruent(self):
        self.data_incongruent = self.data.loc[self.participant]['incongruent']
        return self.data_incongruent
    
    def extract_rrs(self):
        self.rrs = self.data.loc[self.participant]['RRS_score']
        return self.rrs

    def generate_params(self, random, args):
        self.initialParams = [random.uniform(self.minParamValues[i], self.maxParamValues[i]) for i in range(self.num_inputs)]
        return self.initialParams

    def evaluate_params(self, candidates, args):
        self.fitnessCandidates = []
        iteration = 0
        
        for cand in candidates:
            # Run color naming trials ----------------------------------------------------------------------------------------------
            conditions = 3
            response_colournaming = []
            response_colournaming2 = [] #what is the point of this??
            for cond in range(conditions):

                #re-initialize weights to response layer 
                response_color_weights.parameters.matrix.set(exp_weights_reset['response_colour'], Bidirectional_Stroop)
                response_word_weights.parameters.matrix.set(exp_weights_reset['response_word'], Bidirectional_Stroop)
                response_emotion_weights.parameters.matrix.set(exp_weights_reset['response_emotion'], Bidirectional_Stroop)
                color_response_weights.parameters.matrix.set(weights['colour_response'], Bidirectional_Stroop)
                word_response_weights.parameters.matrix.set(weights['word_response'], Bidirectional_Stroop)
                emotion_response_weights.parameters.matrix.set(weights['emotion_response'], Bidirectional_Stroop)
                task_layer.parameters.function.set(pnl.Logistic(gain = model_parameters['task_layer']['gain']), Bidirectional_Stroop)
                task_layer.parameters.hetero.set(model_parameters['task_layer']['inhibition'], Bidirectional_Stroop)
                task_layer.parameters.integration_rate.set(model_parameters['task_layer']['rate'], Bidirectional_Stroop)
                task_layer.parameters.function.set(pnl.Logistic(x_0 = model_parameters['task_layer']['bias']), Bidirectional_Stroop)
                task_color_weights.parameters.matrix.set(weights['task_colour'], Bidirectional_Stroop)
                task_word_weights.parameters.matrix.set(weights['task_word'], Bidirectional_Stroop)
                #task_emotion_weights.parameters.matrix.set(weights['task_emotion'], Bidirectional_Stroop)

                #run baseline
                Bidirectional_Stroop.run(inputs=colour_naming_stimuli[cond][0], num_trials=settle_trials)

                #change weights for experiment

                #THIS IS WHERE WE HAVE POTENTIAL FOR OPTIMIZATION!
                colour_naming_exp['response_colour'][0,0] = cand[0]
                colour_naming_exp['response_colour'][1,1] = cand[0]
                colour_naming_exp['response_word'][0,0] = cand[1]
                colour_naming_exp['response_word'][1,1] = cand[1]
                weights['colour_response'][0,0] = cand[2]
                weights['colour_response'][1,1] = cand[2]
                weights['word_response'][0,0] = cand[3]
                weights['word_response'][1,1] = cand[3]
                weights['task_colour'][0,:] = cand[4]
                weights['task_word'][1,:] = cand[4]
                #weights['task_emotion'][2,:] = cand[8]

                response_color_weights.parameters.matrix.set(colour_naming_exp['response_colour'], Bidirectional_Stroop)
                response_word_weights.parameters.matrix.set(colour_naming_exp['response_word'], Bidirectional_Stroop)
                response_emotion_weights.parameters.matrix.set(colour_naming_exp['response_emotion'], Bidirectional_Stroop)
                color_response_weights.parameters.matrix.set(weights['colour_response'], Bidirectional_Stroop)
                word_response_weights.parameters.matrix.set(weights['word_response'], Bidirectional_Stroop)
                emotion_response_weights.parameters.matrix.set(weights['emotion_response'], Bidirectional_Stroop)
                task_color_weights.parameters.matrix.set(weights['task_colour'], Bidirectional_Stroop)
                task_word_weights.parameters.matrix.set(weights['task_word'], Bidirectional_Stroop)
                task_emotion_weights.parameters.matrix.set(weights['task_emotion'], Bidirectional_Stroop)

                task_layer.parameters.function.set(pnl.Logistic(gain = cand[5]), Bidirectional_Stroop)
                task_layer.parameters.hetero.set(cand[6], Bidirectional_Stroop)
                task_layer.parameters.integration_rate.set(cand[7], Bidirectional_Stroop)
                task_layer.parameters.function.set(pnl.Logistic(x_0 = cand[8]), Bidirectional_Stroop)
                #run exp
                Bidirectional_Stroop.run(inputs=colour_naming_stimuli[cond][1], termination_processing=terminate_trial)

                # Store values from run -----------------------------------------------------------------------------------------------
                B_S = Bidirectional_Stroop.name
                r = response_layer.log.nparray_dictionary('value')      
                rr = r[B_S]['value']
                n_r = rr.shape[0]
                rrr = rr.reshape(n_r, 2) 
                response_colournaming.append(rrr) 
                response_colournaming2.append(rrr.shape[0]) 
                self.scaledrt = np.dot(response_colournaming2, 5) + 115

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

            # MSE computation 
            data_congruent = self.extract_congruent()
            data_incongruent = self.extract_incongruent()
            mse_congruent = (data_congruent - self.scaledrt[2]) ** 2
            mse_incongruent = (data_incongruent - self.scaledrt[1]) ** 2

            #difference between congruent and incongruent
            diff = data_incongruent - data_congruent 
            diff_sim = self.scaledrt[1] - self.scaledrt[2]

            mse_diff = (diff - diff_sim) ** 2

            fitness = (0.3*mse_congruent + 0.3*mse_incongruent + 0.4*mse_diff) 
            jokes = [data_incongruent, data_congruent] #indeed [rsv]
            print('data:', jokes)
            print('simulation:', self.scaledrt[1:3])
            print('fitness:', fitness)
            print('iteration:', iteration)
            iteration += 1

            self.fitnessCandidates.append(fitness)

        return self.fitnessCandidates

    def find_bestcandidate(self):
        rand = Random()
        rand.seed(0)

        from setup import weights, model_parameters, colour_naming_exp

        self.minParamValues = [colour_naming_exp['response_colour'][0,0] * 0.75, 
                            colour_naming_exp['response_word'][0,0] * 0.75, 
                            weights['colour_response'][0,0] * 0.75,
                            weights['word_response'][0,0] * 0.75,
                            weights['task_colour'][0,0] * 0.78,
                            model_parameters['task_layer']['gain'] * 0.1,
                            model_parameters['task_layer']['inhibition'] * 0.5 ,
                            model_parameters['task_layer']['rate'] * 0.03,
                            model_parameters['task_layer']['bias'], 
                           #weights['task_emotion'][2,0] * 0.9
                            ]
        

        self.maxParamValues= [colour_naming_exp['response_colour'][0,0] * 1.25, 
                            colour_naming_exp['response_word'][0,0] * 2, 
                            weights['colour_response'][0,0] * 1.25,
                            weights['word_response'][0,0] * 2,
                            weights['task_colour'][0,0] * 1.2,
                            model_parameters['task_layer']['gain'] * 4,
                            model_parameters['task_layer']['inhibition'] * 1.5,
                            model_parameters['task_layer']['rate'] * 1.5,
                            (model_parameters['task_layer']['bias'] + 1.0) * 2.1,
                            #weights['task_emotion'][2,0] * 1.5
                            ]


        # SET UP EVOLUTIONARY COMPUTATION ----------------------------------------------------------------------
        self.st_ec = ec.EvolutionaryComputation(rand)
        self.st_ec.selector = ec.selectors.truncation_selection  # purely deterministic
        self.st_ec.variator = [ec.variators.uniform_crossover, ec.variators.gaussian_mutation]
        self.st_ec.replacer = ec.replacers.generational_replacement
        self.st_ec.terminator = ec.terminators.evaluation_termination  # terminates after max number of evals is met
        self.st_ec.observer = ec.observers.plot_observer  # save to file, use observers.file_observer

        self.final_pop = self.st_ec.evolve(generator=self.generate_params,  # f'n for initializing params
                                            evaluator=self.evaluate_params,  # f'n for evaluating fitness values
                                            pop_size=self.pop_size,  # number of parameter sets per evaluation
                                            maximize=False,  # best fitness corresponds to minimum value
                                            bounder=ec.Bounder(  # set min/max param bounds
                                                self.minParamValues,
                                                self.maxParamValues
                                            ),
                                            max_evaluations=self.max_evaluations,
                                            num_selected=self.num_selected,
                                            mutation_rate=self.mutation_rate,
                                            num_inputs=self.num_inputs,
                                            num_elites=self.num_elites
                                            )

        self.final_pop.sort(reverse=True)  # sort final population so best fitness is first in list
        self.bestCand = self.final_pop[0].candidate  # bestCand <-- individual @ start of list

        plt.savefig('figures/op-output/observer_%s.pdf' % self.flag)  # save fitness vs. iterations graph
        plt.close()

        return self.bestCand

    def build_optimized_model(self):
        optparams = self.find_bestcandidate() 
        rrs = self.extract_rrs()
        print(optparams)
        conditions = 3
        response_colournaming = []
        response_colournaming2 = [] #what is the point of this??
        for cond in range(conditions):

                 #re-initialize weights to response layer 
                response_color_weights.parameters.matrix.set(exp_weights_reset['response_colour'], Bidirectional_Stroop)
                response_word_weights.parameters.matrix.set(exp_weights_reset['response_word'], Bidirectional_Stroop)
                response_emotion_weights.parameters.matrix.set(exp_weights_reset['response_emotion'], Bidirectional_Stroop)
                color_response_weights.parameters.matrix.set(weights['colour_response'], Bidirectional_Stroop)
                word_response_weights.parameters.matrix.set(weights['word_response'], Bidirectional_Stroop)
                emotion_response_weights.parameters.matrix.set(weights['emotion_response'], Bidirectional_Stroop)
                task_layer.parameters.function.set(pnl.Logistic(gain = model_parameters['task_layer']['gain']), Bidirectional_Stroop)
                task_layer.parameters.hetero.set(model_parameters['task_layer']['inhibition'], Bidirectional_Stroop)
                task_layer.parameters.integration_rate.set(model_parameters['task_layer']['rate'], Bidirectional_Stroop)
                task_layer.parameters.function.set(pnl.Logistic(x_0 = model_parameters['task_layer']['bias']), Bidirectional_Stroop)
                task_color_weights.parameters.matrix.set(weights['task_colour'], Bidirectional_Stroop)
                task_word_weights.parameters.matrix.set(weights['task_word'], Bidirectional_Stroop)
                #task_emotion_weights.parameters.matrix.set(weights['task_emotion'], Bidirectional_Stroop)

                #run baseline
                Bidirectional_Stroop.run(inputs=colour_naming_stimuli[cond][0], num_trials=settle_trials)

                #change weights for experiment

                #THIS IS WHERE WE HAVE POTENTIAL FOR OPTIMIZATION!
                colour_naming_exp['response_colour'][0,0] = optparams[0]
                colour_naming_exp['response_colour'][1,1] = optparams[0]
                colour_naming_exp['response_word'][0,0] = optparams[1]
                colour_naming_exp['response_word'][1,1] = optparams[1]
                weights['colour_response'][0,0] = optparams[2]
                weights['colour_response'][1,1] = optparams[2]
                weights['word_response'][0,0] = optparams[3]
                weights['word_response'][1,1] = optparams[3]
                weights['task_colour'][0,:] = optparams[4]
                weights['task_word'][1,:] = optparams[4]
                #weights['task_emotion'][2,:] = cand[8]

                response_color_weights.parameters.matrix.set(colour_naming_exp['response_colour'], Bidirectional_Stroop)
                response_word_weights.parameters.matrix.set(colour_naming_exp['response_word'], Bidirectional_Stroop)
                response_emotion_weights.parameters.matrix.set(colour_naming_exp['response_emotion'], Bidirectional_Stroop)
                color_response_weights.parameters.matrix.set(weights['colour_response'], Bidirectional_Stroop)
                word_response_weights.parameters.matrix.set(weights['word_response'], Bidirectional_Stroop)
                emotion_response_weights.parameters.matrix.set(weights['emotion_response'], Bidirectional_Stroop)
                task_color_weights.parameters.matrix.set(weights['task_colour'], Bidirectional_Stroop)
                task_word_weights.parameters.matrix.set(weights['task_word'], Bidirectional_Stroop)
                task_emotion_weights.parameters.matrix.set(weights['task_emotion'], Bidirectional_Stroop)

                task_layer.parameters.function.set(pnl.Logistic(gain = optparams[5]), Bidirectional_Stroop)
                task_layer.parameters.hetero.set(optparams[6], Bidirectional_Stroop)
                task_layer.parameters.integration_rate.set(optparams[7], Bidirectional_Stroop)
                task_layer.parameters.function.set(pnl.Logistic(x_0 = optparams[8]), Bidirectional_Stroop)
                #run exp
                Bidirectional_Stroop.run(inputs=colour_naming_stimuli[cond][1], termination_processing=terminate_trial)

                # Store values from run -----------------------------------------------------------------------------------------------
                B_S = Bidirectional_Stroop.name
                r = response_layer.log.nparray_dictionary('value')      
                rr = r[B_S]['value']
                n_r = rr.shape[0]
                rrr = rr.reshape(n_r, 2) 
                response_colournaming.append(rrr) 
                response_colournaming2.append(rrr.shape[0]) 
                self.scaledrt_built = np.dot(response_colournaming2, 5) + 115

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

        colnames = ['pid', 'cb', 'rrs', 'incongruent_data', 'congruent_data', 'incongruent_sim', 'congruent_sim']
        param_names = ['response_colour', 'response_word', 'colour_response', 'word_response', 'task_colour_w', 
                       'task_gain', 'task_inhib', 'task_intg', 'task_bias'
                       ] #'task_word',
        columns = colnames + param_names

        #assemble stuff 
        data = [self.participant[0], self.counterbalance[0], rrs, self.data_incongruent, self.data_congruent, self.scaledrt_built[1], self.scaledrt_built[2]]
        data_fordf = data + optparams

        df = pd.DataFrame([data_fordf], columns = columns)
        df.to_csv('simresults/opt-results/params_%s.csv' % self.flag)

        return self.scaledrt_built

    def plot_results(self):
        simulation = self.build_optimized_model()
        # Data for the bar plot
        groups = ['Incongruent', 'Congruent']
        values1 = [simulation[1], simulation[2]]
        values2 = [self.data_incongruent, self.data_congruent]

        # Creating the bar plot
        x = range(len(groups))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x, values1, width, label='Simulation', color = 'gray')
        ax.bar([i + width for i in x], values2, width, label='Experiment', color = 'cornflowerblue')

        # Adding labels, title, and legend
        ax.set_ylabel('Reaction Time')
        ax.set_xlabel('Condition')
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels(groups)
        ax.legend()

        # Displaying the plot
        fig.savefig('figures/op-output/participant_%s.png' % self.flag)
        plt.close(fig)


#n_test = 5
#for p in range(73, 76):
#    optimize_stroop(standard, 1, p, max_evaluations= 180)


#failed_to_converge = [24] #26, 29, 39, 41, 43, 46, 48, 49, 51, 55, 56, 65, 75] #3, 19 
#for p in failed_to_converge

#optimize_stroop(standard, 1, 7, max_evaluations= 200, mutation_rate= 0.03)
