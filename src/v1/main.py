import functions

def main() : 
    Pos = []
    Neg = []
    Pos, Neg = functions.getRandomImages(500, Pos, Neg)
    functions.results(Pos, "positive", 60)
    functions.results(Neg, "negative", 60)
    return 0;

main()