## PySemEval2010
This is a yet another dataset of SemEval2010 task 8.  
Data is tuple of words, entity1 indexes and entity2 indexes.  
Target is tuple of relation class and direction.


## Example

```py
import pickle

with open('result/task8_train.pickle', 'rb') as fd:
    train_data, train_target = pickle.load(fd)

print(train_data[0], train_target[0])
# (([12],
#   [15],
#   ['The',
#    'system',
#    'as',
#    'described',
#    'above',
#    'has',
#    'its',
#    'greatest',
#    'application',
#    'in',
#    'an',
#    'arrayed',
#    'configuration',
#    'of',
#    'antenna',
#    'elements']),
#  ('Component-Whole', '(e2,e1)'))

print(train_data[1], train_target[1])
#  (([1],
#   [9],
#   ['The',
#    'child',
#    'was',
#    'carefully',
#    'wrapped',
#    'and',
#    'bound',
#    'into',
#    'the',
#    'cradle']),
#  ('Other', None))
```

## Build

### Requirements
- [SemEval 2010 Task 8](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview 
)
  - Download ```TEST_FILE_FULL.TXT``` and ```TRAIN_FILE.TXT``` from SemEval-2010 website.  
  - And save these files in this project's data directiory.
- Stanford CoreNLP with py-corenlp
  - [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started)
  - [py-corenlp](https://github.com/smilli/py-corenlp)

### Make pickle files
```
$ python3 src/task8.py
$ tree result/
result/
├── task8_test.pickle
└── task8_train.pickle
```
