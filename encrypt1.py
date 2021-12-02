import numpy as np
import string
import random
from time import time
from math import sqrt
import math

rounds_no = random.randint(2,4)

dict1 = {"rounds_no": rounds_no}
two_bit_list = ['00', '01', '10', '11']
dna_bases = ['A', 'C', 'G', 'T']

four_bit_list = ['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100',
                 '1101', '1110', '1111']
two_dna_bases = ['TA', 'TC', 'TG', 'TT', 'GA', 'GC', 'GG', 'GT', 'CA', 'CC', 'CG', 'CT', 'AA', 'AC', 'AG', 'AT']

two_bits_to_dna_base_table = None
dna_base_to_two_bits_table = None
four_bits_to_two_dna_base_table = None
two_dna_base_to_four_bits_table = None
chromosome_length = None
chromosomes_no=[]
chromosome_length1=[]
rf=[]
crossover_p=[]
bases1=[]
bases2=[]
bases1a=[]
bases2a=[]
point1=[]
point2=[]
point11=[]
point22=[]
point1a=[]
point2a=[]
point11a=[]
point22a=[]
test=0
test1=0

def str2bin(sstring):
   
    bs = ''
    for c in sstring:
        bs = bs + bin(ord(c))[2:].zfill(8)
    return bs


def byte2bin(byte_val):
    
    return bin(byte_val)[2:].zfill(8)


def bitxor(a, b):
    
    return "".join([str(int(x) ^ int(y)) for (x, y) in zip(a, b)])


def generate_pre_processing_tables():
    
    global two_bits_to_dna_base_table
    global dna_base_to_two_bits_table

    
    two_bits_to_dna_base_table = dict(zip(two_bit_list, dna_bases))
    dna_base_to_two_bits_table = dict(zip(two_bits_to_dna_base_table.values(), two_bits_to_dna_base_table.keys()))


def generate_mutation_tables():
    
    global four_bits_to_two_dna_base_table
    global two_dna_base_to_four_bits_table

    
    four_bits_to_two_dna_base_table = dict(zip(four_bit_list, two_dna_bases))
    two_dna_base_to_four_bits_table = dict(
        zip(four_bits_to_two_dna_base_table.values(), four_bits_to_two_dna_base_table.keys()))


def group_bits(byte, step=2):
    
    bits = []
    for i in range(0, len(byte), step):
        bits.append(byte[i:i + step])
    return bits


def generate_bits(byte_data):
    
    grouped_bits_data = []

    for byte in byte_data:
        print("byte in byte_data",byte)
        grouped_bits_data.extend(group_bits(byte))

    print("Grouped bits in binarized data:",grouped_bits_data)

    return grouped_bits_data


def binarized_data(data):

    for c in data:
        print(byte2bin(ord(c)))
    
    byte_data = [byte2bin(ord(c)) for c in data]

    return generate_bits(byte_data)


def bits_to_dna(data, conversion_table):
    
    return "".join([conversion_table[bits] for bits in data])


def dna_to_bits(data):
    
    return "".join([dna_base_to_two_bits_table[dna_base] for dna_base in data])


def encrypt_key(data, key):
    
    if len(data) > len(key):
        factor = int(len(data) / len(key))
        key += key * factor

        return bitxor(data, key)

    return bitxor(data, key)


def reshape(dna_sequence):
    
    global chromosomes_no
    global chromosome_length
    global chromosome_length1

    
    chromosome_no = random.randint(2, int(len(dna_sequence) / 2))
    print("chromosome_no in reshape:",chromosome_no)
    chromosomes_no.insert(0,chromosome_no)
   
    chromosome_length = int(len(dna_sequence) / chromosome_no)
    chromosome_length1.insert(0,chromosome_length)
    chromosomes = []

   
    for i in range(0, len(dna_sequence), chromosome_length):
        chromosomes.append(dna_sequence[i:i + chromosome_length])
    print("Reshaped chromosomes:",chromosomes)

    return chromosomes


def reverse_reshape(population):
   
    return "".join(population)


def rotate_crossover(population):
    global rf
    global chromosome_length
    flag = 1
    new_population = []

    
    rotation_offset = random.randint(0, chromosome_length)
    rf.insert(0,rotation_offset)

    

    for chromosome in population:
    
        
        flag =  flag * 4 
        print("flag:",flag)
        if flag < 9:
            print("right")
            right_first = chromosome[0: len(chromosome) - rotation_offset]
            right_second = chromosome[len(chromosome) - rotation_offset:]
            new_population.append(right_second + right_first)
        else:
            print("left")
            left_first = chromosome[0: rotation_offset]
            left_second = chromosome[rotation_offset:]
            new_population.append(left_second + left_first)
        flag = flag + 1
    return new_population


def single_point_crossover(population):
    
    flag1 = 3
    new_population = []
    for i in range(0, len(population) - 1, 2):
        candidate1 = population[i]
        candidate2 = population[i + 1]

        
        length = min(len(candidate1), len(candidate2))
        
        crossover_point =int ( ( ( flag1 ** 2.2 ) + 5 * ( flag1 ** 3.3 ) ) / ( 4 * ( flag1 **4.4 ) ) )
        if crossover_point > length: 
            crossover_point = crossover_point % length
        offspring1 = candidate2[0: crossover_point] + candidate1[crossover_point:]
        offspring2 = candidate1[0: crossover_point] + candidate2[crossover_point:]
        new_population.append(offspring1)
        new_population.append(offspring2)
    
    if len(population) % 2 == 1:
        new_population.append(population[len(population) - 1])
    flag1 = flag1 + 1
    return new_population


def crossover(population):
    global crossover_p
    
    cp= random.randint(1,10)
    crossover_p.insert(0,cp)
    
    if cp < 3:
        return rotate_crossover(population)
    elif cp >= 3 and cp <= 6:
        return single_point_crossover(population)
    else:
        population = rotate_crossover(population)
        return single_point_crossover(population)


def complement(chromosome, point1, point2):
    
    new_chromosome = ""

    for i in range(len(chromosome)):
        if i >= point1 and i <= point2:
            if chromosome[i] == '0':
                new_chromosome += '1'
            else:
                new_chromosome += '0'
        else:
            new_chromosome += chromosome[i]

    return new_chromosome


def alter_dna_bases(bases):
    global bases1
    global bases2
    global test1
    global bases1a
    global bases2a
    
    alter_dna_table = {}

    for _ in range(2):
        
        b1=random.randint(0, len(bases) - 1)
        base1 = bases[b1]
        print("b1:",b1)
        if ( test1 == 0):
            bases1.append(b1)
        if ( test1 > 0 ):
            bases1a.append(b1)
            print("bases1a:",bases1a)
        bases.remove(base1)
        
        b2=random.randint(0, len(bases) - 1)
        base2 = bases[b2]
        print("b2:",b2)
        if ( test1 == 0 ):
            bases2.append(b2)
        if ( test1 > 0 ):
            bases2a.append(b2)
            print("bases2a:",bases2a)
        bases.remove(base2)
        
        alter_dna_table[base1] = base2
        alter_dna_table[base2] = base1
        print("alter_dna_table:",alter_dna_table)
    if ( test1 > 0 ):
        bases1=bases1a+bases1
        bases2=bases2a+bases2
        bases1a=[]
        bases2a=[]
    test1 = test1 + 1

    return alter_dna_table


def mutation(population):
    global point1
    global point2
    global point11
    global point22
    global point1a
    global point2a
    global point11a
    global point22a
    global test
   
    global two_bits_to_dna_base_table
    global four_bits_to_two_dna_base_table

    bases = ['A', 'C', 'G', 'T']
    alter_dna_table = alter_dna_bases(bases)
    print("altered dna table:",alter_dna_table)

    new_population = []
    for chromosome in population:
        
        b_chromosome = dna_to_bits(chromosome)
        print("dna_to_bits in mutation:",b_chromosome)
        print("len of b_chromosome:",len(b_chromosome))
        p1 = random.randint(0, len(b_chromosome) - 1)
        p2 = random.randint(p1, len(b_chromosome) - 1)
        print("p1:",p1)
        print("p2:",p2)
        if ( test == 0 ):
            point1.append(p1)
            point2.append(p2)
        if ( test > 0 ):
            point1a.append(p1)
            point2a.append(p2)
        
        b_chromosome = complement(b_chromosome, p1, p2)
        print("b_chromosome:",b_chromosome)

        
        four_bits_vector = group_bits(b_chromosome, 4)
        print("four_bits_vector:",four_bits_vector)

        last_dna_base = None
        if len(four_bits_vector[len(four_bits_vector) - 1]) == 2:
            last_dna_base = two_bits_to_dna_base_table[four_bits_vector[len(four_bits_vector) - 1]]
            four_bits_vector = four_bits_vector[:-1]
        dna_seq = bits_to_dna(four_bits_vector, four_bits_to_two_dna_base_table)
        if last_dna_base is not None:
            dna_seq += last_dna_base
        p11 = random.randint(0, len(dna_seq) - 1)
        p22 = random.randint(p11, len(dna_seq) - 1)
        if ( test == 0 ):
            point11.append(p11)
            point22.append(p22) 
        if ( test > 0 ):
            point11a.append(p11)
            point22a.append(p22)
        new_chromosome = ""
        print("dna_seq before alter dna:",dna_seq)
        print("len of dna_seq before alter dna:",len(dna_seq))
        for i in range(len(dna_seq)):
            if i >= p11 and i <= p22:
                
                new_chromosome += alter_dna_table[dna_seq[i]]
            else:
                new_chromosome += dna_seq[i]

        new_population.append(new_chromosome)
    if ( test1 > 0 ):
        point1=point1a+point1
        point2=point2a+point2
        point1a=[]
        point2a=[]
        point11=point11a+point11
        point22=point22a+point22
        point11a=[]
        point22a=[]
    test = test + 1
    return new_population


def dna_get(text, key):
    global rounds_no
    global two_bits_to_dna_base_table

    print("\nDNA-GET is running...\n")

    
    b_data1 = binarized_data(text)
    dna_seq = bits_to_dna(b_data1, two_bits_to_dna_base_table)
    print("dna_seq after binarized data:",dna_seq)
    

    b_data2 = dna_seq
    print("len of b_data2",len(b_data2))
    print("Initial DNA sequence:", dna_seq)

    
    while rounds_no > 0:
        print("____________________________________________")
        print("round no:",rounds_no)

        print("b_data2 before round 1:",b_data2)

    
        b_data2 = bits_to_dna(
            group_bits(encrypt_key(dna_to_bits(reverse_reshape(b_data2)), key)), two_bits_to_dna_base_table)

        print("b_data2 after round1:",b_data2)
    
    
        b_data2 = reshape(b_data2)

        print("b_data2 after reshape:",b_data2)
    

    
        b_data2 = crossover(b_data2)

        print("b_data2 after crossover:",b_data2)

    
        b_data2 = mutation(b_data2)
    
        print("b_data2 after mutation:",b_data2)
        rounds_no -= 1

    return reverse_reshape(b_data2)


def main():
    global chromosomes_no
    global chromosome_length1
    global rf
    global crossover_p
    global bases1
    global bases2
    global point1
    global point2
    global point11
    global point22
    text = "reshma1234567890"
    

    print("Text:", text)


    keyen = str2bin(''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16)))
    

    print("Key:", len(keyen), keyen)
    decrypt = {'keyen' : keyen }
    
    generate_pre_processing_tables()
    generate_mutation_tables()

    start = time()
    seq = dna_get(text, keyen)
    print("Final dna sequence:",seq)
    decrypt['seq']=seq
    
    end = time()
    dict1['chromosome_no']=chromosomes_no
    dict1['chromosome_length1']=chromosome_length1
    dict1['rotation_offset']=rf
    dict1['crossover_p']=crossover_p
    dict1['base1']=bases1
    dict1['base2']=bases2
    dict1['point1']=point1
    dict1['point2']=point2
    dict1['point11']=point11
    dict1['point22']=point22

    print(end - start)
    

    with open("dec.txt", 'w') as fi:
        for key, value in decrypt.items():
            fi.write('%s:%s\n' % (key, value))
    np.save('test1.npy',dict1)
    read_d  = np.load('test1.npy').item()
    with open("test1.txt", 'w') as fi:
        for key, value in read_d.items():
            fi.write('%s:%s\n' % (key, value))

if __name__ == '__main__':
    main()