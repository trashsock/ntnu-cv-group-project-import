from v2 import functions

def v2main(amount = 500) : 
    Pos = []
    Neg = []
    Pos, Neg = functions.getRandomImages(amount, Pos, Neg)
    resultList = [functions.results(Pos, "positive"), functions.results(Neg, "negative")]
    return resultList;