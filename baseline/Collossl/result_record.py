
import numpy as np
## shot = 1, 5, 10, 50
result_shot = [[0.111470, 0.154535, 0.162171, 0.161124, 0.171608],
               [0.405674, 0.4088089, 0.48846136, 0.487568806, 0.228241145],
               [0.471657, 0.487892, 0.4925255, 0.5734091, 0.587485],
               [0.522793, 0.647509, 0.6734978, 0.6492550, 0.729884]]

result_shot = np.array(result_shot)
result_shot = result_shot * 100
print(np.mean(result_shot, axis=1))
print(np.std(result_shot, axis=1))

## portion = 40 60 80 100
result_portion = [[0.522793, 0.6475094, 0.673498, 0.6492551, 0.7298848],
                  [0.6547525, 0.735800, 0.63784770, 0.59971352, 0.75070415],
                  [0.6231585, 0.688584, 0.64969, 0.6835898, 0.73940221],
                  [0.64935256, 0.7220736, 0.6796298, 0.73541418, 0.71678580]]
result_portion = np.array(result_portion)
result_portion = result_portion * 100
print(np.mean(result_portion, axis=1))
print(np.std(result_portion, axis=1))


result_alpha = [[0.471657, 0.487892, 0.4925255, 0.5734091, 0.587485],
                [0.529912, 0.609710, 0.518806, 0.5628067, 0.568746],
                [0.549757, 0.5120234, 0.549420, 0.552302, 0.500625],
                [0.4897875, 0.652854, 0.483546, 0.602877, 0.5188453],]
result_alpha = np.array(result_alpha)
result_alpha = result_alpha * 100
print(np.mean(result_alpha, axis=1))
print(np.std(result_alpha, axis=1))
