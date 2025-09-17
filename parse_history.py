import json

nums_rows = [2, 4, 8, 16, 24, 32]
class_metric = 'accuracy'
autor_metric = 'root_mean_squared_error'


def print_keys(data):
    print('Keys: [ ', end='')
    for k in data.keys():
        print(f'{k}, ', end='')
    print(']')


if __name__ == '__main__':
    prefix = 'runs_'
    suffix = '/model-classifier.json'
    for rows in nums_rows:
        filename = f'{prefix}{rows}{suffix}'
        # Opening JSON file
        with open(filename, 'r') as f:
            data = json.load(f)
            history = data['history']
            # In every three, the first element is the trace of the training,
            # and it is ignored. The second and third elements contain
            # the metric and loss for the classifier and autoencoder,
            # respectively
            print(f'History lenght: {len(history)}')
            for i in range(0, len(history), 3):
                class_value = history[i + 1][class_metric]
                autor_value = history[i + 2][autor_metric]
                print(f'{rows},{class_value},{autor_value}')
