import string
import difflib

PUNCTUATION = string.punctuation.replace('-', '')


def simplify_text(text: str):
    return text.strip().lower().translate(str.maketrans('', '', PUNCTUATION))


def get_diff(seq1, seq2):
    seq1 = simplify_text(seq1).split(' ')
    seq2 = simplify_text(seq2).split(' ')
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    modified1 = []
    modified2 = []
    for op in matcher.get_opcodes():
        if not op[0] == 'equal':
            mod1 = ""
            mod2 = ""
            for x in range(op[1], op[2]):
                mod1 += ' ' + seq1[x]
            for x in range(op[3], op[4]):
                mod2 += ' ' + seq2[x]
            modified1.append(simplify_text(mod1))
            modified2.append(simplify_text(mod2))

    return modified1, modified2