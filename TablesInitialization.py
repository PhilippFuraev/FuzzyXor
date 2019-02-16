import random

FUZZY_XOR_TABLE_NAME = "fuzzy_table.txt"
WEIGHTS_YABLE_NAME = "weigths.txt"
FUZZY_XOR_TABLE_HEIGTH = 100000;


def create_fuzzy_xor_table():
    f = open("fuzzy.txt", 'w')
    fuzzy_xor_table = [[0 for x in range(3)] for y in range(FUZZY_XOR_TABLE_HEIGTH)]
    for i in range(len(fuzzy_xor_table)):
        for j in range(len(fuzzy_xor_table[i]) - 1):
            fuzzy_xor_table[i][j] = random.randint(0, 1)
        fuzzy_xor_table[i][2] = fuzzy_xor_table[i][0] ^ fuzzy_xor_table[i][1]
        fuzzy_xor_table[i][0] = fuzzy_xor_table[i][0] + random.normalvariate(0, 0.1)
        fuzzy_xor_table[i][1] = fuzzy_xor_table[i][1] + random.normalvariate(0, 0.1)
        string = ""
        for j in range(3):
            string += str(fuzzy_xor_table[i][j]) + ' '
        f.write(string + "\n")
    f.close()


def create_weigths_table():
    f = open("weigths1.txt", 'w')
    weigths_table = [[0 for x in range(3)] for y in range(2)]
    for i in range(len(weigths_table)):
        for j in range(len(weigths_table[i])):
            weigths_table[i][j] = random.random()
        string = ""
        for j in range(len(weigths_table[0])):
            string += str(weigths_table[i][j]) + ' '
        f.write(string + "\n")
    f.close()
    f = open("weigths2.txt", 'w')
    weigths_table = [[0 for x in range(3)] for y in range(1)]
    for i in range(len(weigths_table)):
        for j in range(len(weigths_table[i])):
            weigths_table[i][j] = random.random()
        string = ""
        for j in range(len(weigths_table[0])):
            string += str(weigths_table[i][j]) + ' '
        f.write(string + "\n")
    f.close()


create_fuzzy_xor_table()
create_weigths_table()
