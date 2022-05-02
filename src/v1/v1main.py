from v1 import functions

def v1main(amount = 500) : 
    Pos = []
    Neg = []
    Pos, Neg = functions.getRandomImages(amount, Pos, Neg)
    resultList = [functions.results(Pos, "positive", 60), functions.results(Neg, "negative", 60)]
    return resultList;