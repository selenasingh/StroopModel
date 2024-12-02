# Params and set-up
import numpy as np

# Individual Unit Variables ----------------------------------------------------------------------------------------------------
model_parameters = { 

    'default_units' : {
        'default_bias': 0.0,
        'default_gain': 1.0,
        'threshold': 0.55 #threshold for decision
    } , 

    #task layer 
    'task_layer' : {
        'gain': 1.0,
        'bias': 0.0,
        'rate': 0.03,
        'inhibition': -2.0
    } ,

    #colour hidden layer 
    'colour_hidden' : {
        'gain': 1.0,
        'bias': 4.0,
        'rate': 0.03,
        'inhibition': -2.0
    } ,

    #word hidden layer
    'word_hidden' : {
        'gain': 1.0,
        'bias': 4.0,
        'rate': 0.03,
        'inhibition': -2.0
    } ,

    #emotion hidden layer
    'emotion_hidden' : {
        'gain': 1.0,
        'bias': 4.0,
        'rate': 0.03,
        'inhibition': -2.0
    } ,

    #response layer 
    'response_layer' : {
        'gain': 1.0,
        'bias': 0.0,
        'rate': 0.03,
        'inhibition': -2.0
    } , 

    'threshold': 0.55, #threshold for decision
}

# Weights Matrices 
weights = {
    'colour_input' : np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ]), 

    'word_input' : np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ]),

    'emotion_input' : np.array([
        [1.0, 0.0, 0.0], # positive
        [0.0, 1.0, 0.0], # negative
        [0.0, 0.0, 0.0]  # neutral
    ]),

    'task_input' : np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0] # added third emotion layer
    ]), 

    'colour_task' : np.array([
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0] # third column is emotion layer
    ]),

    'task_colour' : np.array([
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0] # third layer is emotion layer
    ]), 

    'word_task' : np.array([
        [0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 4.0, 0.0]
    ]), 

    'task_word' : np.array([
        [0.0, 0.0, 0.0],
        [4.0, 4.0, 4.0],
        [0.0, 0.0, 0.0]
    ]), 

    'emotion_task' : np.array([
        [0.0, 0.0, 4], #scale to 8 for emotional induction
        [0.0, 0.0, 4], # Assume that negative node has a stronger bias
        [0.0, 0.0, 4] # in theory, emotion leads to slowing of other processes in terms of rumination
    ]), 

    'task_emotion' : np.array([
        [0.0, 0.0, 0.0], 
        [0.0, 0.0, 0.0], 
        [4.0, 4.0, 4.0]
    ]), 

    'response_colour' : np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]), 

    'response_word' : np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]), 

    'response_emotion' : np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        # NOTE: the word and colour have no return value from the response. Does that make sense for emotion?
    ]), 

    'colour_response' : np.array([
        [1.5, 0.0],
        [0.0, 1.5],
        [0.0, 0.0]
    ]), 

    'word_response' : np.array([
        [2.5, 0.0], # this would be red to red. This would be red to green (so rows = own node, columns = task node... duh that's why it's 3x2)
        [0.0, 2.5], # this would be green to green
        [0.0, 0.0]
    ]), 

    'emotion_response' : np.array([
        # Our hypothesis predicts that both positive and negative cause interference. 
        # Assumes that negative words cause greater interference
        [0.0, 0.5], #positive, BL: 1.5
        [0.0, 1.5], #negative, BL: 2.5
        [0.0, 0.5] #neutral 
    ]),

}

exp_weights_reset = {
    'response_colour' : np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]), 

    'response_word' : np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
    
    'response_emotion' : np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ]),
}

colour_naming_exp = {
    #scaling weights for colour naming experiment
    'response_colour' : np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]
        ]), 

    'response_word' : np.array([
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0]
        ]), 

    'response_emotion': np.array([
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0]
        ]), 
}

