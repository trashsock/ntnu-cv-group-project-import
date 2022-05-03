from v1 import functions
def v1main(amount = 500) : 
    Pos = []
    Neg = []
    Pos, Neg = functions.getRandomImage(int(amount), Pos, Neg)
    resultList = [functions.results(Pos, "positive"), functions.results(Neg, "negative")]
    return resultList;
