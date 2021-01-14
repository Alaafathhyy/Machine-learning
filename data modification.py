import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

data = pd.read_csv("gene_table.txt")
chromosome_table = data.groupby('chromosome')['gene_name'].count()


def part1():
    # the total nmber of genes is equal to the data size
    print("TotalNumberOfGenes =", data["gene_name"].size)
    # taking tha genes in a list to remove the duplicates and print the size
    genes_types =set(data["gene_biotype"])
    print("TotalNumberOfBioTypesOfGenes =", len(genes_types))

def part2():
    print("MaxValue =", data["transcript_count"].max())
    print("MinValue =", data["transcript_count"].min())
    print("MeanValue =", data["transcript_count"].mean())
    print("MedianValue =", data["transcript_count"].median())


def part3():
    # count for each chrosome how many genes for it
    print("NumberOfGenesAtEachChromsome =", chromosome_table.sort_values())
    # the barchart
    # take the values and the index as a list and send them to barfunction
    y = chromosome_table.values.tolist()
    x = chromosome_table.index.tolist()
    plt.bar(x, y)
    plt.plot(x, y)
    plt.show()

def part4():
    # get all the data which the strand =='+' in positive _strands
    positive_Strands = data[data.get('strand') == '+']
    # group the positive strands by the chromosome type and count of each type
    positive_Strands = positive_Strands.groupby("chromosome")['strand'].count()
    y = (chromosome_table.index).tolist()
    lst = list()
    for i in y:
        if (i in positive_Strands):  # maybe type of chromosome doesn't exist in positive_Strands
            lst.append([i, (positive_Strands[i] / chromosome_table[i]) * 100])
    df = DataFrame(lst, columns=['Type', 'percentage'])
    print("ThePercentageOf+InEachChromosome", df)


def part5():
    print("theAverageOfEachBiotype =", data.groupby('gene_biotype')['transcript_count'].mean())


part1()
part2()
part3()
part4()
part5()
