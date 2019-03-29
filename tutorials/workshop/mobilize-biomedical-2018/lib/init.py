"""
Configure database connection for all workshop notebooks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import os

DBNAME = 'MIMIC'

if DBNAME == 'spouse':
    os.environ['SNORKELDB'] = "postgresql://ubuntu:snorkel@localhost/spouse"
elif DBNAME == 'CDR':
    os.environ['SNORKELDB'] = "sqlite:///data/db/cdr.db"
elif DBNAME == 'MIMIC':
    os.environ['SNORKELDB'] = "postgresql:///clef_2014_task2_py3" # ubuntu:snorkel@localhost

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.models import Candidate, Sentence, Span, Document

session = SnorkelSession()
