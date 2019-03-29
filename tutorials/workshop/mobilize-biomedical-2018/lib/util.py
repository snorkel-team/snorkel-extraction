from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import pandas as pd
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels

import re
from snorkel.lf_helpers import (
    get_left_tokens, get_right_tokens, get_between_tokens,
    get_text_between, get_tagged_text,
)


FPATH = 'data/gold_labels.tsv'

def number_of_people(sentence):
    active_sequence = False
    count = 0
    for tag in sentence.ner_tags:
        if tag == 'PERSON' and not active_sequence:
            active_sequence = True
            count += 1
        elif tag != 'PERSON' and active_sequence:
            active_sequence = False
    return count


def load_external_labels(session, candidate_class, annotator_name='gold'):
    gold_labels = pd.read_csv(FPATH, sep="\t")
    for index, row in gold_labels.iterrows():    

        # We check if the label already exists, in case this cell was already executed
        context_stable_ids = "~~".join([row['person1'], row['person2']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))
                    
        # Because it's a symmetric relation, load both directions...
        context_stable_ids = "~~".join([row['person2'], row['person1']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))

    # Commit session
    session.commit()

    # Reload annotator labels
    reload_annotator_labels(session, candidate_class, annotator_name, split=1, filter_label_split=False)
    reload_annotator_labels(session, candidate_class, annotator_name, split=2, filter_label_split=False)


# def check_exercise_1(subclass):
#     """
#     Check if type is Person
#     :param subclass:
#     :return:
#     """
#     v = subclass.__mapper_args__['polymorphic_identity'] == "person"
#     v &= len(subclass.__argnames__) == 1 and 'person' in subclass.__argnames__
#     print('Correct!' if v else 'Sorry, try again!')


# def check_exercise_2(c):
#     s1 = c[0].get_span()
#     s2 = c[1].get_span()
#     print('Correct!' if "{} {}".format(s1, s2) == "Katrina Dawson Paul Smith" else 'Sorry, try again!')
    
    
#
# Answer LFs
#
def married_between(c):
    return 'married' in get_text_between(c).lower()
        
def same_last_name(c):
    p1 = c[0].get_attrib_tokens('words')
    p2 = c[1].get_attrib_tokens('words')
    # ignore single word names or candidates where p1 == p2
    if len(p1) == 1 or len(p2) == 1 or p1 == p2:
        return False
    return p1[-1].lower() == p2[-1].lower()

def count_tokens_between(c):
    return len(get_text_between(c))

#
# Check Exercises
#

def check_exercise(num, lf, candidates):
    v = True
    if num == 1:
        tlf = count_tokens_between
    elif num == 2:
        tlf = married_between
    elif num == 3:
        tlf = same_last_name
    else:
        print("Invalid Exercise ID")
        return 
    
    # check answer
    n,N = 0,0
    for c in candidates:
        lbl1 = lf(c)
        lbl2 = tlf(c)
        v &= (lbl1 == lbl2)
        n += 0 if not (lbl1 == lbl2) else 1
        N += 1
    print(v, "{:2.1f}% agreement".format(n/N*100.0))


#
# Print Answers
#
def show_exercise1_answer():
    answer = '''
    def len_between_tokens(c):
        return len(get_text_between(c)) 
    '''
    print(answer)
    
def show_exercise2_answer():
    answer = '''
    def married_in_between_tokens(c):
        return 'married' in get_text_between(c).lower()'''
    print(answer)
    
def show_exercise3_answer():    
    answer = '''
    def same_last_name(c):
        # fetch a list of all words in a person name (for both spans)
        p1 = c[0].get_attrib_tokens('words')
        p2 = c[1].get_attrib_tokens('words')
        # ignore single word names or candidates where p1 == p2
        if len(p1) == 1 or len(p2) == 1 or p1 == p2:
            return False
        return p1[-1].lower() == p2[-1].lower()'''
    print(answer)
