import math
import numpy as np

class HomogeneityIndependence:
    def __init__(self, calculation_data, choppedARRAYS):
        self.calculation_data = calculation_data
        self.choppedARRAYS = choppedARRAYS

    def coincidenceMeanDep(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1] 

        zL = [a - b for a, b in zip(xL, yL)]

        zLMean = sum(zL)/len(zL)

        zLVariance = sum(math.pow(z - zLMean, 2) for z in zL)/(len(zL)-1)

        t = (zLMean*math.sqrt(len(zL)))/math.sqrt(zLVariance)

        return t

    def coincidenceMeanIndep(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1] 

        xLMean = sum(xL)/len(xL)
        yLMean = sum(yL)/len(yL)
        zLMean = xLMean - yLMean

        xLVariance = sum(math.pow(x - xLMean, 2) for x in xL)/(len(xL)-1)
        yLVariance = sum(math.pow(y - yLMean, 2) for y in yL)/(len(yL)-1)
        zLVariance = math.sqrt(xLVariance/len(xL) + yLVariance/len(yL))

        t = zLMean/zLVariance

        return t 
    
    def coincidenceVarianceIndep(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1] 

        xLMean = sum(xL)/len(xL)
        yLMean = sum(yL)/len(yL)

        xLVariance = sum(math.pow(x - xLMean, 2) for x in xL)/(len(xL)-1)
        yLVariance = sum(math.pow(y - yLMean, 2) for y in yL)/(len(yL)-1)

        f = 0
        if xLVariance >= yLVariance:
            f = xLVariance/yLVariance
        else:
            f = yLVariance/xLVariance

        return f

    def wilcoxon(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1]

        sorted_values, source_list, ranks = getRanks(xL, yL)

        x_ranks = [ranks[i] for i, source in enumerate(source_list) if source == 'x']

        y_ranks = [ranks[i] for i, source in enumerate(source_list) if source == 'y']

        n1 = len(x_ranks)
        n2 = len(y_ranks)
        n = len(sorted_values)

        wilW = sum(x_ranks)
        eW = (n1*(n+1))/2
        dW = (n1*n2*(n+1))/12

        w = (wilW-eW)/math.sqrt(dW)

        return w

    def mannaWhitney(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1]

        n1 = len(xL)
        n2 = len(yL)

        mySum = 0
        for j in range(n2):
            for i in range(n1):
                if xL[i]>yL[j]:
                    mySum = mySum + 1
                if xL[i]<=yL[j]:
                    mySum = mySum + 0

        eU = (n1*n2)/2
        dU = (n1*n2*(n1+n2+1))/12

        u = (mySum-eU)/math.sqrt(dU)

        return u

    def diffMeanRanks(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1]

        sorted_values, source_list, ranks = getRanks(xL, yL)

        x_ranks = [ranks[i] for i, source in enumerate(source_list) if source == 'x']

        y_ranks = [ranks[i] for i, source in enumerate(source_list) if source == 'y']

        n1 = len(x_ranks)
        n2 = len(y_ranks)
        n = n1+n2

        rXMean = sum(x_ranks)/n1
        rYMean = sum(y_ranks)/n2

        v = (rXMean - rYMean)/(n*math.sqrt((n+1)/(12*n1*n2)))

        return v

    def kolmogorovSmirnov(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1] 

        n1, n2 = len(xL), len(yL)
        
        samples = xL + yL

        sorted_samples = sorted(samples)
        z = 0
        for value in sorted_samples:
            first_emp = sum(x <= value for x in xL) / n1
            second_emp = sum(x <= value for x in yL) / n2

            current_difference = abs(first_emp - second_emp)
            z = max(z, current_difference)

        n = min(len(xL),len(yL))

        partOneFormula = 1-(2*z)/(3*math.sqrt(n))+((2*z**2)/(3*n))*(1-(2*z**2)/3) 
        partTwoFormula = (4*z)/(9*math.sqrt(n**3))*(1/5-(19*z**2)/(15)+(2*z**4)/(3)) 
        lZ = 1 - math.exp(-2*z**2)*(partOneFormula+partTwoFormula)

        alfa = 0.05

        return 1 - lZ

    def singsSureDepButHomogeneity(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1] 

        zL = [a - b for a, b in zip(xL, yL)]

        uL = []
        for i in range(len(zL)):
            if zL[i] > 0:
                uL.append(1)
            if zL[i] < 0:
                uL.append(0)

        sumOfuL = sum(uL)

        sStar = (2*sumOfuL-1-len(zL))/(math.sqrt(len(zL)))

        return sStar

    def AbbeIndep(self):
        xL = self.calculation_data[0]
        yL = self.calculation_data[1]

        zL = [a - b for a, b in zip(xL, yL)]

        zLMean = sum(zL)/len(zL)

        dSquare = (sum((zL[l+1] - zL[l])**2 for l in range(len(zL)-1)))/(len(zL)-1)

        zLVariance = (sum(math.pow(z - zLMean, 2) for z in zL))/(len(zL)-1)

        q = dSquare/(2*zLVariance)

        u = (q-1)*math.sqrt(((len(zL))**2-1)/(len(zL)-2))

        return u

    def Bartlett(self):
        myMatrix = self.choppedARRAYS
        m = len(myMatrix)
        n = len(myMatrix[0])

        xMeanI = [sum(myMatrix[i]) / n for i in range(m)]
        xVarianceI = [sum((x - xMeanI[i]) ** 2 for x in row)/(len(row)-1) for i, row in enumerate(myMatrix)]

        sSqueare = sum((len(row) - 1)*xVarianceI[i] for i, row in enumerate(myMatrix))/sum(len(row) - 1 for row in myMatrix)

        b = -sum((len(row)-1)*math.log(xVarianceI[i]/sSqueare) for i, row in enumerate(myMatrix))

        c = 1 + 1/(3*(m-1))*(sum(1/(len(row)-1) for row in myMatrix)-1/sum(len(row)-1 for row in myMatrix))

        hiSquare = b/c

        return hiSquare

    def UnivariateVarianceAnalysis(self):
        myMatrix = self.choppedARRAYS

        xMeanRow = []

        for row in myMatrix:
            xMeanRow.append(sum(row)/len(row))

        variances = []

        for row in myMatrix:
            mean = sum(row) / len(row)
            squared_diff = sum((x - mean) ** 2 for x in row)
            variance = squared_diff / len(row)
            variances.append(variance)

        n = sum(len(myMatrix[i]) for i in range(len(myMatrix)))

        xMeanMatrix = sum(len(myMatrix[i])*xMeanRow[i] for i in range(len(myMatrix)))/n 

        sMSquare = sum(len(myMatrix[i])*(xMeanRow[i]-xMeanMatrix)**2 for i in range(len(myMatrix)))/(len(myMatrix)-1)

        sBSquare = sum((len(myMatrix[i])-1)*variances[i] for i in range(len(myMatrix)))/(n-len(myMatrix))

        f = sMSquare/sBSquare
        return f


    def Hcriterion(self):
        myMatrix = self.choppedARRAYS

        rankMatrix = getRanksForMatrix(myMatrix) 

        wIStrkoke = []

        for row in rankMatrix:
            row_average = sum(row) / len(row)
            wIStrkoke.append(row_average)

        n = sum(len(myMatrix[i]) for i in range(len(myMatrix)))

        h = sum((wIStrkoke[i]-(n+2)/2)**2/((n+1)*(n-len(myMatrix[i]))/(12*len(myMatrix[i])))*(1-len(myMatrix[i])/n) for i in range(len(myMatrix)))

        return h

    def Qcriterion(self):
        myMatrix = self.choppedARRAYS

        xMean = [sum(row) / len(row) for row in myMatrix]

        binarizedMyMatrix = [[1 if element > avg else 0 for element in row] for row, avg in zip(myMatrix, xMean)]

        tSum = [sum(row) for row in binarizedMyMatrix]

        uSum = [sum(column) for column in zip(*binarizedMyMatrix)]

        k = len(myMatrix)

        tMean = sum(tSum)/k

        firstSum = sum(math.pow(tJ - tMean, 2) for tJ in tSum)
        secondSum = k*sum(uSum)
        thirdSum = sum(math.pow(u, 2) for u in uSum)

        qIndex = (k*(k-1)*firstSum)/(secondSum - thirdSum)

        return qIndex
        
def getRanks(x, y):
    combined_list = [(value, 'x') for value in x] + [(value, 'y') for value in y]

    combined_list.sort()

    sorted_values = [item[0] for item in combined_list] 
    source_list = [item[1] for item in combined_list]  

    ranks = list(range(1, len(combined_list) + 1))

    unique_sorted_values = sorted(set(sorted_values))
    for value in unique_sorted_values:
        indices = [i for i, x in enumerate(sorted_values) if x == value]
        if len(indices) > 1:
            average_rank = sum(ranks[i] for i in indices) / len(indices)
            for i in indices:
                ranks[i] = average_rank 

    return sorted_values, source_list, ranks
def getRanksForMatrix(list_of_lists):
    all_values = []
    list_indices = []

    for i, sublist in enumerate(list_of_lists):
        for value in sublist:
            all_values.append(value)
            list_indices.append(i)

    sorted_data = sorted(zip(all_values, list_indices))
    sorted_values, sorted_indices = zip(*sorted_data)

    ranks = []
    rank_counts = get_rank_counts(sorted_data)

    for start_rank, count in rank_counts:
        avg_rank = sum(range(start_rank, start_rank + count)) / count
        ranks.extend([avg_rank] * count)

    ranks = [x + 1 for x in ranks]
    result_lists = create_lists_by_indices_and_ranks(list(sorted_indices), ranks)

    return result_lists

def get_rank_counts(sorted_data):
    rank_counts = []
    start_rank = 0
    count = 1

    for i in range(1, len(sorted_data)):
        if sorted_data[i][0] == sorted_data[i - 1][0]:
            count += 1
        else:
            rank_counts.append((start_rank, count))
            start_rank = i
            count = 1

    rank_counts.append((start_rank, count))
    
    return rank_counts

def create_lists_by_indices_and_ranks(indices, ranks):
    result_dict = {}

    for index, rank in zip(indices, ranks):
        if index not in result_dict:
            result_dict[index] = [rank]
        else:
            result_dict[index].append(rank)

    result_lists = [result_dict[key] for key in sorted(result_dict.keys())]
    return result_lists