import sys
from html.parser import HTMLParser


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


def split_relation_and_direction(r):
    O = 'Other'
    if r == O:
        return r, None
    else:
        return  r[:-7], r[-7:]


def load_dataset(rows, split=False):
    rows = iter(rows)
    while True:
        try:
            sentnce_ann = next(rows).rstrip('\n').split('\t')[-1].strip('"')
        except StopIteration:
            return
        e1, e2, sentence =  parse_sentence(sentnce_ann)
        relation = next(rows).rstrip('\n')
        next(rows) # Comment
        next(rows) # Blank line
        print(e1, e2, sentence, split_relation_and_direction(relation))


if __name__ == '__main__':
    TRAIN_FILE = 'data/TRAIN_FILE.TXT'
    TEST_FILE = 'data/TEST_FILE_FULL.TXT'

    with open(TRAIN_FILE, 'r', encoding='utf-8') as fd:
        load_dataset(fd)

    with open(TEST_FILE, 'r', encoding='utf-8') as fd:
        load_dataset(fd)
