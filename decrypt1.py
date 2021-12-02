import numpy as np
import string
import random
from time import time
from math import sqrt
import math

my_dict = {}
c_no=0
c_no1=0
cp=0
rf=0
bases1=0
bases2=0
o=0
p=0
m=0
n=0
flag1=0
chromosome_length = None

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

def bits_to_dna(data, conversion_table):
    
    return "".join([conversion_table[bits] for bits in data])


def groupdna2(new_population,step):
    
    dna = []
    for i in range(0, len(new_population), step):
        dna.append(new_population[i:i + step])
    return dna

def groupdna(new_population,step):
    
    dna = []
    for i in range(0, len(new_population)-1, step):
        dna.append(new_population[i:i + step])
    return dna
def alter_dna_bases(bases):
    global bases1
    global bases2
    global read_d
    print("bases:",bases)
    
    alter_dna_table = {}

    for _ in range(2):
        
        b1=read_d['base1'][bases1]
        print("b1:",b1)
        bases1=bases1+1
        base1 = bases[b1]
        bases.remove(base1)
        
        b2=read_d['base2'][bases2]
        print("b2:",b2)
        bases2=bases2+1
        base2 = bases[b2]
        bases.remove(base2)
        
        alter_dna_table[base1] = base2
        alter_dna_table[base2] = base1
        print("alter_dna_table:",alter_dna_table)

    return alter_dna_table

def group_bits(byte, step):
    
    bits = []
    for i in range(0, len(byte), step):
        bits.append(byte[i:i + step])
    return bits

def dna_to_bits(data):
    
    return "".join([dna_base_to_two_bits_table[dna_base] for dna_base in data])

def byte_to_text(data):
    
    d1=[]
    for c in data:
        d=int(c, 2)
        d1.append(d)
    decimal="".join(map(chr,d1))
    return decimal


def rotate_crossover(population):
    global read_d
    global rf
    flag = 1
    new_population = []

    
    rotation_offset = read_d['rotation_offset'][rf]
    print("rotation_offset:",rotation_offset)
    rf=rf+1

    for chromosome in population:
    
        
        flag =   flag * 4  
        print("flag:",flag)
        if flag < 9:
            print("left")
            left_first = chromosome[0: rotation_offset]
            left_second = chromosome[rotation_offset:]
            new_population.append(left_second + left_first)
            
        else:
            print("right")
            right_first = chromosome[0: len(chromosome) - rotation_offset]
            right_second = chromosome[len(chromosome) - rotation_offset:]
            new_population.append(right_second + right_first)
        flag = flag + 1
    print("crossover:",new_population)
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
    global read_d
    global cp
    crossover_p=read_d['crossover_p'][cp]
    print("crossover_p:",crossover_p)
    cp=cp+1
    if crossover_p < 3:
        return rotate_crossover(population)
    elif crossover_p >= 3 and crossover_p <= 6:
        return single_point_crossover(population)
    else:
        population = single_point_crossover(population)
        print("after single_point_crossover:",population)
        return rotate_crossover(population)
    
        
def reshape(dna_sequence):
    
    
    global chromosome_lengths
    global read_d
    global c_no
    
    chromosome_no = read_d['chromosome_no'][c_no]
    c_no=c_no+1
   
    chromosome_length = int(len(dna_sequence) / chromosome_no)
    chromosomes = []

   
    for i in range(0, len(dna_sequence), chromosome_length):
        chromosomes.append(dna_sequence[i:i + chromosome_length])
    print("Reshaped chromosomes:",chromosomes)

    return chromosomes

def dna_to_bits_mutation(data):
    
    return "".join([two_dna_base_to_four_bits_table[dna_base] for dna_base in data])
    
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

def mutation(seq):
    global two_bits_to_dna_base_table
    global four_bits_to_two_dna_base_table
    global read_d
    global o
    global p,m,n
    new_population=[]
    new_population3=[]
    bases = ['A', 'C', 'G', 'T']
    alter_dna_table = alter_dna_bases(bases)
    print("altered dna table:",alter_dna_table)

    for chromosomes in seq:
        new_chromosome = ""
        p11 = read_d['point11'][o]
        p22 = read_d['point22'][p]
        o = o + 1
        p = p + 1
        for i in range(len(chromosomes)):
            if i >= p11 and i <= p22:
                
                new_chromosome += alter_dna_table[chromosomes[i]]
            else:
                new_chromosome += chromosomes[i]
        print("new_chromosome:",new_chromosome)
        two_dna_vector = groupdna2(new_chromosome, 2)
        print("two_dna_vector:",two_dna_vector)
        last_dna_base = None
        if len(two_dna_vector[len(two_dna_vector) - 1]) == 1:
            print("len of two dna vector last:",len(two_dna_vector[len(two_dna_vector) - 1]))
            last_dna_base = dna_base_to_two_bits_table[two_dna_vector[len(two_dna_vector) - 1]]
            two_dna_vector = two_dna_vector[:-1]
        dna_seq = dna_to_bits_mutation(two_dna_vector)
        if last_dna_base is not None:
            dna_seq += last_dna_base
        print("dna_seq:",dna_seq)
        p1 = read_d['point1'][m]
        p2 = read_d['point2'][n]
        m = m + 1
        n = n + 1
        chromosomes = complement(dna_seq, p1, p2)
        print("complented chromosomes:",chromosomes)
        new_population=group_bits(chromosomes,2)
        print("new_population:",new_population)
        new_population3.append(bits_to_dna(new_population,two_bits_to_dna_base_table ))
        print("new_population3:",new_population3)
    return new_population3
    
   
def reverse_reshape(population):
   
    return "".join(population)

def bitxor(a, b):
        return "".join([str(int(x.strip('')) ^ int(y.strip(''))) for (x, y) in zip(a, b)])
        

def decrypt_key(data, key):
    
    if len(data) > len(key):
        factor = int(len(data) / len(key))
        key += key * factor

        return bitxor(data, key)

    return bitxor(data, key)
    
def text_get(fseq, key):

    global r_no
    global c_no1
    global flag1

    while r_no > 0:
        print("flag:",flag1)
        print("____________________________________________")
        print("round no:",r_no)
        chromosome_length = read_d['chromosome_length1'][c_no1]
        print("chromosome_length:",chromosome_length)
        c_no1=c_no1+1
        if ( flag1 ) == 0 :
            seq=groupdna(fseq,chromosome_length)
            seq[-1] = seq[-1].strip()
            print("flag1 is 0")
        if ( flag1 ) > 0 :
            seq=groupdna2(dna_seq,chromosome_length)
            print("flag1 is not 0")
        print("after groupdna:",seq)
        dna_data2 = mutation(seq)
        dna_data2 = crossover(dna_data2)
        print("dna_data2 after crossover:",dna_data2)
        dna_data2=reverse_reshape(dna_data2)
        dna_data2 = reshape(dna_data2)
        dna_data2=bits_to_dna(group_bits(decrypt_key(dna_to_bits(reverse_reshape(dna_data2)),key),2),two_bits_to_dna_base_table)
        print(dna_data2)

        dna_seq=dna_data2

        print("Initial DNA sequence:", dna_seq)

        r_no = r_no - 1

        flag1 = flag1 + 1

    b_data2=dna_to_bits(dna_data2)

    b_data2=group_bits(b_data2,8)

    text=byte_to_text(b_data2)

    return(text)


    
        



def main():
    global my_dict

with open('dec.txt') as fileobj:
  for line in fileobj:
      key, value = line.split(":")
      my_dict[key] = value

key = my_dict['keyen']
fseq = my_dict['seq']

generate_pre_processing_tables()
generate_mutation_tables()

read_d  = np.load('test1.npy').item()
r_no = read_d['rounds_no']

text = text_get(fseq, key)
print("Original content:",text)

if __name__ == '__main__':
    main()






