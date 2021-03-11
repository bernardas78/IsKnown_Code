import PairSequence

ps = PairSequence.PairSequence()

for x,y in ps:
    print ("len(x): {}, x[0].shape: {}".format(len(x), x[0].shape))
    print ("y: {}".format(y))