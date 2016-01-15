from model import StringFeatures

feats = StringFeatures()

assert len(feats) == 0
assert len(feats.get_attributes("running")) == 8

feats.get_attributes("running", True)
assert len(feats.get_attributes("cattable", True)) == 8
assert len(feats.attributes) == 16
feats.store("running")
print feats["cattable"]
#print len(feats)
