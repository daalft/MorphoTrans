import codecs
import sys

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
threshold = float(sys.argv[2])
with codecs.open(sys.argv[1], 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        splits = line.split("|||")
        source = splits[0].strip()
        target = splits[1].strip()
        score = float(splits[2].strip().split(" ")[0])
        if len(source.split(" ")) == 1 and len(target.split(" ")) == 1:
            if score > threshold:
                print source, target, score
