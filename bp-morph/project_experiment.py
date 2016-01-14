from corpora import Lexicon, LexiconCollection, Dictionary
import cPickle as pickle
import argparse
import numpy as np
from model import RRBM

# for debugging
np.random.seed(0)


parser = argparse.ArgumentParser(description='Map some tags.')
parser.add_argument('--target')
parser.add_argument('--source')
parser.add_argument('--align')
args = parser.parse_args()

# languages
lang1, lang2 = "lang1", "lang2"

# add the lexicons
lex_col = LexiconCollection()
lex_col.add(lang1, args.source)
lex_col.add(lang2, args.target)
lex_col.create()

# adds the dictionary
d = Dictionary(args.align, lang1, lang2)

# # pickle the stuff
# pickle.dump(lex_col, open("tmp/lex_col.pickle", 'wb'))
# pickle.dump(d, open("tmp/d.pickle", 'wb'))

#lex_col = pickle.load(open("tmp/lex_col.pickle", 'rb'))
#d = pickle.load(open("tmp/d.pickle", 'rb'))

def sample_from_d(d, N, lex1, lex2, filt=None):
    """ get N samples from the dictionary N """
    # filter according to the lexicon
    lst = []
 
    for (l1, l2) in d.store.keys():
        if l1 in lex1.words and l2 in lex2.words:
            if filt is not None:
                if (l1, l2) not in filt:
                    lst.append(str((l1, l2)))
            else:                
                lst.append(str((l1, l2)))

    return map(lambda x: eval(x), np.random.choice(lst, size=N, replace=False))
    

# tag to tag projection experiments

# number of tag sizes to get for the learning curve
test_size = 1000
sizes = map(lambda x : 10*x, xrange(1, 101))
# number of experiments to run with each size
experiments = 100

# model
model = RRBM(lex_col.N, [], C=0.1)

# main loop over different experiments (parallelize)
for experiment in xrange(experiments):
    test = []
    test_keys = sample_from_d(d, test_size, lex_col[lang1], lex_col[lang2])
    test_keys_set = set(test)
    for (l1, l2) in test_keys:
        test.append((lex_col[lang1][l1], lex_col[lang2][l2]))
            
        entry1 = lex_col[lang1].pp(l1)
        entry2 = lex_col[lang2].pp(l2)

        #if "num=PL" in entry2[1]:
        #    print l2, entry2[1]

        
    for size in [110]: #sizes:
        train_keys = sample_from_d(d, size, lex_col[lang1], lex_col[lang2], test_keys_set)
        train = []
        for (l1, l2) in train_keys:
            train.append((lex_col[lang1][l1], lex_col[lang2][l2]))

        model.reset()
        model.train = train

        before = model.eval(test)[0:2]
        #print size, experiment

        C_best = 0
        err, hmm, mistakes = float("inf"), float("inf"), None
        # grid search
        for C in [0.1]: #[0.1, 0.5, 1.0, 2.0, 3.0, 4.0]:
            model.C = C
            model.learn(disp=0)
            err_tmp, hmm_tmp, mistakes_tmp = model.eval(test)

            if hmm_tmp < hmm:
                err, hmm, mistakes = err_tmp, hmm_tmp, mistakes_tmp
                C_best = C
        after = (err, hmm)
        
        print size, experiment, before, after, C_best

        for k, v in sorted(mistakes.items(), key=lambda x: -x[1]):
            print lex_col.avs.lookup(k), v

        for l1, l2 in test_keys:
            i1 = lex_col[lang1][l1]
            i2 = lex_col[lang2][l2]
            print l1, lex_col[lang1].pp(l1)
            print l2, lex_col[lang2].pp(l2)
            err, hmm, mistakes = model.eval([(i1, i2)])
            for k, v in mistakes.items():
                print lex_col.avs.lookup(k)
            raw_input()
        raw_input()
        #print len(test), len(train), len(test.intersection(train))
        #exit(0)
