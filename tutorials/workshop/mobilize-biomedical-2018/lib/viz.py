from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from IPython.core.display import display, HTML
from snorkel.lf_helpers import *


def view_labeled_candidates(X, y, label=-1, n_max=10):
    i = 0/Users/fries/Library/Mobile Documents/com~apple~Grab/Documents/Untitled 8.png
    for x,y in zip(X, y):
        if y[0] != label:
            continue
        render(x, y[0])
        if i > n_max:
            break
        i+=1
        
def render(c, label=0):
    style = {0:'#0079AA', 1:'#FF7C00'}
    display(candidate_html(c, label=label, style=style))

def candidate_html(c, label=0, style={}, full_sent=True, use_colors=True, pretty_print=True):
    colors = {1: u"#00e600", 0: u"#CCCCCC", -1: u'#ff4000'}
    div_tmpl = u'''<div style="border: 1px dotted #858585; border-radius:8px;
    background-color:#FDFDFD; padding:5pt 10pt 5pt 10pt">{}</div>'''
    
    sent_tmpl = u'<p style="font-size:12pt;">{}</p>'
    arg_tmpl = u'<b style="color:white; text-shadow: 1px 1px 2px #000000; background-color:{};padding:2pt 5pt 2pt 5pt; border-radius:8px">{}</b>'
    chunks = get_text_splits(c)

    text = []
    
    # if spans are nested, then build a single relation span
    a_i, a_j = c[0].char_start, c[0].char_end
    b_i, b_j = c[1].char_start, c[1].char_end
    if (a_i >= b_i and a_i <= b_j) or (a_j >= b_i and a_j <= b_j) or (b_i >= a_i and b_i <= a_j) or (b_j >= a_i and b_j <= a_j):
        parent = c[0] if (a_j - a_i) > (b_j - b_i) else c[1]
        child = c[0] if parent == c[1] else c[1]
        
        span = parent.get_span()
        span = span.replace("padding:2pt 5pt 2pt 5pt", "padding:5pt 5pt 5pt 5pt")
        repl = arg_tmpl.format(style[0] if child == c[0] else style[1], child.get_span())
        span = span.replace(child.get_span(), repl) 
        span = arg_tmpl.format(style[0] if parent == c[0] else style[1], span)
        
        matched = False
        for s in chunks[0:]:
            if s in [u"{{A}}", u"{{B}}"]:
                if not matched:
                    text += [span]
                    matched = True
            elif not pretty_print:
                text += [s.replace(u"\n", u"<BR/>")]
            else:
                text += [s]
    
    else:
        for s in chunks[0:]:
            if s in [u"{{A}}", u"{{B}}"]:
                span = c[0].get_span() if s == u"{{A}}" else c[1].get_span()
                text += [arg_tmpl.format( style[0] if s == u"{{A}}" else style[1], span)]
            elif not pretty_print:
                text += [s.replace(u"\n", u"<BR/>")]
            else:
                text += [s]
            
    text = "".join(text)
       
    if pretty_print:
        text = re.sub(r'''\s{2,}|\n+''', ' ', text)
    
    text = text.replace("||","")
    html = div_tmpl.format(sent_tmpl.format(text.strip()))
    return HTML(html)

def display_candidate(c, label=0):
    display(candidate_html(c,label))
