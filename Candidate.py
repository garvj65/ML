import numpy as np
import pandas as pd

dataset = pd.read_csv('2.csv')
concepts = np.array(dataset.iloc[:, :-1])
target = np.array(dataset.iloc[:, -1])

def candidate_elimination(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [['?'] * len(specific_h) for _ in range(len(specific_h))]

    for i, instance in enumerate(concepts):
        if target[i] == 'yes':
            specific_h = [spec if spec == val else '?' for spec, val in zip(specific_h, instance)]
        elif target[i] == 'no':
            for j in range(len(specific_h)):
                if specific_h[j] != instance[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = '?'

    general_h = [gh for gh in general_h if gh.count('?') != len(gh)]
    return specific_h, general_h

s_final, g_final = candidate_elimination(concepts, target)
print("Final Specific Hypothesis:", s_final)
print("Final General Hypotheses:", g_final)
