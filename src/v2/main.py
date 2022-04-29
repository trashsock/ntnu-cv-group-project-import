from turtle import clear
import functions
def main() : 
    Pos = []
    Neg = []
    Pos, Neg = functions.getRandom225x225(500, Pos, Neg)
    functions.results(Pos, "positive")
    functions.results(Neg, "negative")
    return 0;

main()