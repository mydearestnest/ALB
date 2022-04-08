import pickle

from base import System, Elem, Node

test_class = System(1, 1, 1, 1, 1, 1)

with open('1.pkl', 'wb') as file:
    save_class_str = pickle.dumps(test_class)
    file.write(save_class_str)

with open("1.pkl", 'rb') as file:
    new_class = pickle.loads(file.read())
