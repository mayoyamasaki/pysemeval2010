import sys
import pickle
import itertools
from html.parser import HTMLParser

from pycorenlp import StanfordCoreNLP


CNLP = StanfordCoreNLP('http://localhost:9000')


class EntityParser(HTMLParser):
    TAG_E1 = 'e1'
    TAG_E2 = 'e2'

    def __init__(self):
        super().__init__()
        self.cnt = 0
        self.e1 = []
        self.e2 = []
        self.s = ''

    def handle_starttag(self, tag, attrs):
        if tag == self.TAG_E1:
            self.e1.append(self.cnt)
        elif tag == self.TAG_E2:
            self.e2.append(self.cnt)
        else:
            raise Exception('Tag error : Required entity tag is e1 or e2 only.')

    def handle_endtag(self, tag):
        if tag == self.TAG_E1:
            self.e1.append(self.cnt-1)
        elif tag == self.TAG_E2:
            self.e2.append(self.cnt-1)
        else:
            raise Exception('Tag error : Required entity tag is e1 or e2 only.')

    def handle_data(self, data):
        self.cnt += len(data)
        self.s += data

    def get_data(self):
        if not (len(self.e1) == len(self.e2) == 2):
            raise Exception('Tag error : Required two entities in a sentence but found multiple entities.')

        return tuple(self.e1), tuple(self.e2), self.s


def parse_sentence(s):
    p = EntityParser()
    p.feed(s)
    return p.get_data()


def search_sdp(e1, e2, sentence, dependencies):

    g = [[] for _ in range(len(sentence))]
    edges = {}
    for dep in dependencies:
        src, dst = dep['dependent']-1, dep['governor']-1
        if src == -1 or dst == -1: continue
        g[src].append(dst)
        g[dst].append(src)
        edges[tuple(sorted([src, dst]))] = dep

    def bfs(start, end):
        queue = [[start]]
        while queue:
            path = queue.pop(0)
            n = path[len(path) - 1]
            if n == end:
                return path
            else:
                for n2 in g[n]:
                    if n2 not in path:
                        new_path = path.copy()
                        new_path.append(n2)
                        queue.append(new_path)

    sdp = None
    stkn, dtkn = None, None
    for t1, t2 in itertools.product(e1, e2):
        dp = bfs(t2, t1)
        if sdp is None or len(sdp) > len(dp):
            sdp = dp
            stkn = sentence[t2]
            dtkn = sentence[t1]

    result = []
    for src, dst in zip(sdp, sdp[1:]):
        dep = edges[tuple(sorted([src, dst]))]
        if dep['governorGloss'] == stkn:
            result.append(dep['governorGloss'])
            stkn = dep['dependentGloss']
        elif dep['dependentGloss'] == stkn:
            result.append(dep['dependentGloss'])
            stkn = dep['governorGloss']
        else:
            raise Exception('Failed back trace.')
        result.append(dep['dep'])
    result.append(dtkn)
    return result


def annotate(e1, e2, s):
    res = CNLP.annotate(s, properties={
        'annotators': 'tokenize,depparse',
        'outputFormat': 'json'
    })

    if len(res['sentences']) != 1:
        tokens = []
        for sentence in res['sentences']:
            tokens.extend(sentence['tokens'])
    else:
        sentence = res['sentences'][0]
        tokens = sentence['tokens']

    words = []
    _e1 = []
    _e2 = []
    for i, token in enumerate(tokens):
        words.append(token['word'])
        b = token['characterOffsetBegin']
        e = token['characterOffsetEnd']

        if (e1[0] <= b and e <= e1[1]+1) or (b <= e1[0] and e1[1]+1 <= e):
            _e1.append(i)
        if (e2[0] <= b and e <= e2[1]+1) or (b <= e2[0] and e2[1]+1 <= e):
            _e2.append(i)
    e1, e2 = _e1, _e2


    if len(res['sentences']) != 1:
        sdp = None
    else:
        dependencies = sentence['basicDependencies']
        sdp = search_sdp(e1, e2, words, dependencies)

    return (e1, e2, words, sdp)


def split_relation_and_direction(r):
    O = 'Other'
    if r == O:
        return r, None
    else:
        return  r[:-7], r[-7:]


def load_dataset(rows, split=True):
    data = []
    target = []
    rows = iter(rows)
    while True:
        try:
            sentnce_ann = next(rows).rstrip('\n').split('\t')[-1].strip('"')
        except StopIteration:
            break
        e1, e2, sentence =  parse_sentence(sentnce_ann)
        relation = next(rows).rstrip('\n')
        next(rows) # Comment
        next(rows) # Blank line

        data.append((e1, e2, sentence))
        if split:
            relation, direction = split_relation_and_direction(relation)
            target.append((relation, direction))
        else:
            target.append(relation)
    return data, target


if __name__ == '__main__':
    TRAIN_FILE = 'data/TRAIN_FILE.TXT'
    TEST_FILE = 'data/TEST_FILE_FULL.TXT'

    with open(TRAIN_FILE, 'r', encoding='utf-8') as fd:
        train_data, train_target = load_dataset(fd)
    train_data = [annotate(e1, e2, s) for e1, e2, s in train_data]
    with open('result/task8_train.pickle', 'wb') as fd:
        pickle.dump((train_data, train_target), fd)

    with open(TEST_FILE, 'r', encoding='utf-8') as fd:
        test_data, test_target = load_dataset(fd)
    test_data = [annotate(e1, e2, s) for e1, e2, s in test_data]
    with open('result/task8_test.pickle', 'wb') as fd:
        pickle.dump((test_data, test_target), fd)

