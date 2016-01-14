from collections import defaultdict as dd
from arsenal.alphabet import Alphabet

class Dictionary(object):
    """ Reads in the dictionary with confidence scores """

    def __init__(self, fin, source_lang, target_lang):
        # variables
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # intern the variables
        self.source = Alphabet()
        self.target = Alphabet()
        
        self.store = {}
        with codecs.open(fin, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                source, target, score = line.split(" ")
                #if "Buch" in source or "Stuhl" in source:
                score = float(score)
                self.store[(source, target)] = score
                self.source.add(source)
                self.target.add(target)


class LexiconCollection(object):
    """
    Contains all the lexicon
    objects
    """
    def __init__(self):
        self.atts = Alphabet()
        self.vals = Alphabet()
        self.av = dd(set)
        self.avs = Alphabet()

        self.lexicons = {}

    def add(self, lang, fin):
        self.lexicons[lang] = Lexicon(lang, self.atts, self.vals, self.av, self.avs)
                
                
class Lexicon(object):
    """ Reads in the universal morpholigcal lexicon """

    def __init__(self, fin, atts, vals, av, avs):
        # probably redundant...
        # but not optimizing for space so who cares
        self.atts, self.vals, self.av, self.avs = atts, vals, av, avs

        self.lexicon = dd(list)
        self.words = Alphabet()
        
        with codecs.open(fin, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                word, lemma, tags = line.split(" ")
                self.words.add(word)
                tags = tags.split(",")

                for tag in tags:
                    a, v = tag.split("=")
                    self.av[a].add(v)

                    self.atts.add(a)
                    self.vals.add(v)

                self.lexicon[word].append((lemma, tags))

        # get rid of default dict wrapper
        self.lexicon = dict(self.lexicon)
        self.av = dict(self.av)
        
        for a, s in self.av.items():
            for v in s:
                self.avs.add((a, v))
        self.N = len(self.avs)
        self.W = zeros((len(self.lexicon), self.N))
        
        # use Manaal's encoding (http://arxiv.org/abs/1512.05030)
        for w, lst in self.lexicon.items():
            vec = zeros((self.N))
            for l, ts in lst:
                for tag in ts:
                    a, v = tag.split("=")
                    j = self.avs[(a, v)]
                    vec[j] = 1.0
            i = self.words[w]
            self.W[i] = vec

        print self.pp(5)

        
    def pp(self, i):
        """ pretty print the morphological tag of a word """
        word = self.words.lookup(i)
        lst = []
        for n in xrange(self.N):
            if self.W[i, n] > 0:
                lst.append("=".join(self.avs.lookup(n)))
        return word, ",".join(lst)
        
                    
class EntailmentReader(object):
    """ Reads in John Sylak-Glassman's entailments for the features """
    pass
