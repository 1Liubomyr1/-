import math
import numpy as np
class CorrelationalRegression:
    def __init__(self, calculation_data, choppedARRAYS):
        self.calculation_data = calculation_data
        self.choppedARRAYS = choppedARRAYS

    def AverageSquareForXY(self):
        x_list = self.calculation_data[0]
        y_list = self.calculation_data[1]

        corr_x_average = sum(x_list)/len(x_list)
        corr_x_sq = math.sqrt(sum([math.pow(x - corr_x_average, 2) for x in x_list]) / len(x_list))

        corr_y_average = sum(y_list)/len(y_list)
        corr_y_sq = math.sqrt(sum([math.pow(y - corr_y_average, 2) for y in y_list]) / len(y_list))

        return corr_x_average,corr_x_sq,corr_y_average,corr_y_sq
    
    def z_for_graph_norm(self):
        x_list = np.array(self.calculation_data[0])
        y_list = np.array(self.calculation_data[1])

        corr_x_average = np.mean(x_list)
        corr_x_sq = np.sqrt(np.sum(np.power(x_list - corr_x_average, 2)) / len(x_list))
        corr_y_average = np.mean(y_list)
        corr_y_sq = np.sqrt(np.sum(np.power(y_list - corr_y_average, 2)) / len(y_list))

        z_list = 1 / (2 * math.pi * corr_x_sq * corr_y_sq) * np.exp(
            -0.5 * ((np.power((x_list - corr_x_average) / corr_x_sq, 2) +
                    np.power((y_list - corr_y_average) / corr_y_sq, 2)))
        )

        return z_list
    
    def create_table_combinations_foo(self, n, m):
        data_y = self.calculation_data[1]
        data_x = self.calculation_data[0]

        bins_x = np.linspace(min(data_x), max(data_x), m+1)
        bins_y = np.linspace(min(data_y), max(data_y), n+1)

        hist, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=[bins_x, bins_y])
        hist_calc = np.flipud(hist.T)

        return hist_calc


    def Hi_square_2d(self):
        data_y = self.calculation_data[1]
        data_x = self.calculation_data[0]
        M_x = findM(len(data_x))
        M_y = findM(len(data_y))
        bins_x = np.linspace(min(data_x), max(data_x), M_x+1)
        bins_y = np.linspace(min(data_y), max(data_y), M_y+1)

        hist, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=[bins_x, bins_y])
        hist_calc = np.flipud(hist.T)
        hist_graph = hist
        rel_freq = hist_graph / len(data_x)

        rel_freq_calc = np.flipud(rel_freq.T)
        rel_freq_calc_2 = np.flipud(hist / np.sum(hist))

        non_zero_mask = rel_freq_calc_2 != 0
        chi_squared = np.sum((rel_freq_calc[non_zero_mask] - rel_freq_calc_2[non_zero_mask])**2 / rel_freq_calc_2[non_zero_mask])
        return chi_squared
    
    def correlation_coeff(self):
        y_list = self.calculation_data[1]
        x_list = self.calculation_data[0]
        n = len(x_list)
        corr_x_average = np.mean(x_list)
        corr_x_sq = np.sqrt(np.sum(np.power(x_list - corr_x_average, 2)) / len(x_list))
        corr_y_average = np.mean(y_list)
        corr_y_sq = np.sqrt(np.sum(np.power(y_list - corr_y_average, 2)) / len(y_list))
        corr_xy_mean = xy_mean(x_list,y_list)

        r_x_y = n/(n-1) * (corr_xy_mean-corr_x_average*corr_y_average)/(corr_x_sq*corr_y_sq)

        return r_x_y, corr_x_sq, corr_y_sq

    def corr_coeff_check(r_x_y,n):
        t = (r_x_y*math.sqrt(n-2))/(math.sqrt(1-r_x_y**2))
        return t
    
    def SKV_corr_coeff(r_x_y, n, u):
        r_up = r_x_y + (r_x_y*(1-r_x_y**2))/(2*n) + u*(1-r_x_y**2)/(math.sqrt(n-1))
        r_down = r_x_y + (r_x_y*(1-r_x_y**2))/(2*n) - u*(1-r_x_y**2)/(math.sqrt(n-1))

        return r_up, r_down

    def relation_correlation(self):
        data_x = self.calculation_data[0]
        data_y = self.calculation_data[1]
        number_of_groups = findM(len(data_x))
        min_x = min(data_x)
        max_x = max(data_x)
        range_x = max_x - min_x
        bin_size = range_x / number_of_groups

        bins = [min_x + i * bin_size for i in range(number_of_groups)]

        grouped_data = {bin_key: [] for bin_key in bins}

        for i in range(len(data_x)):
            x = data_x[i]
            bin_index = int((x - min_x) / bin_size)
            if bin_index == number_of_groups:
                bin_index -= 1
            bin_key = bins[bin_index]

            grouped_data[bin_key].append(data_y[i])

        overall_mean = np.mean(data_y)

        total_variance = np.sum((y - overall_mean) ** 2 for y in data_y)

        between_group_variance = sum(len(group) * (np.mean(group) - overall_mean) ** 2
                                    for group in grouped_data.values() if group)
        
        rel_corr = between_group_variance / total_variance if total_variance != 0 else 0

        t = (math.sqrt(rel_corr*(len(data_x)-2)))/math.sqrt(1-math.sqrt(rel_corr))
        return  rel_corr, t
    
    def coefficient_Spearman(self):
        data_x = self.calculation_data[0]
        data_y = self.calculation_data[1]
        N = len(data_x)
        rank_x, rank_y = get_ranks_for_one_list(np.sort(data_x)), get_ranks_for_one_list(np.sort(data_y))

        sorted_indices_x = np.argsort(data_x)
        sorted_indices_y = np.argsort(data_y)

        rank_x_sorted = [0] * len(rank_x)
        rank_y_sorted = [0] * len(rank_y)

        for j, index in enumerate(sorted_indices_x):
            rank_x_sorted[index] = rank_x[j]

        for j, index in enumerate(sorted_indices_y):
            rank_y_sorted[index] = rank_y[j]

        sort_indices = np.argsort(rank_x_sorted)
        rank_y_sorted = [rank_y_sorted[i] for i in sort_indices]
        rank_x_sorted = sorted(rank_x_sorted)

        d, d_2 = [], []
        for i in range(N):
            d_i = rank_x_sorted[i] - rank_y_sorted[i]
            d.append(d_i)
            d_2.append(d_i**2)

        tau_c = 1 - ((6*sum(d_2))/(N*(N**2-1)))

        t = (tau_c*math.sqrt(N-2))/(math.sqrt(1-tau_c**2))

        return tau_c, t
    
    def skv_spirmen(tau_c, t,n):
        sigma_t = math.sqrt((1-tau_c**2)/(n-2))
        spirmen_up = tau_c + t*sigma_t
        spirmen_down = tau_c - t*sigma_t
        return spirmen_up, spirmen_down

    def u_for_kendel_foo(tau_k,n):
        u = 3*tau_k/math.sqrt(2*(2*n+5))*math.sqrt(n*(n-1))
        return u
    
    def count_points_in_quadrants_matrix(self):
        x = self.calculation_data[0]
        y = self.calculation_data[1]
        min_x, max_x = min(x), max(x)
        min_y, max_y = min(y), max(y)

        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        quadrant_counts = [0, 0, 0, 0]

        for i in range(len(x)):
            if x[i] >= mid_x and y[i] >= mid_y:
                quadrant_counts[1] += 1  
            elif x[i] < mid_x and y[i] >= mid_y:
                quadrant_counts[0] += 1  
            elif x[i] < mid_x and y[i] < mid_y:
                quadrant_counts[2] += 1  
            elif x[i] >= mid_x and y[i] < mid_y:
                quadrant_counts[3] += 1 

        matrix = [
            [quadrant_counts[0], quadrant_counts[1]],
            [quadrant_counts[2], quadrant_counts[3]]
        ]

        return matrix, quadrant_counts[0], quadrant_counts[1], quadrant_counts[2], quadrant_counts[3]
    
    def coeff_spoluch_pirson_foo(matrix, num):
        n, m = matrix.shape

        row_sums = np.transpose(np.sum(matrix, axis=1))

        column_sums = np.sum(matrix, axis=0)

        product_matrix = np.outer(row_sums, column_sums)/num

        error_sum = 0.0

        for i in range(n):
            for j in range(m):
                error_sum += ((matrix[i, j] - product_matrix[i, j]) ** 2) / product_matrix[i, j]

        c = math.sqrt(error_sum/(num+error_sum))

        return c, error_sum
    
    def coeff_spoluch_kendall_foo(matrix, num):
        P = 0
        Q = 0
        m, n = matrix.shape
        for i in range(n):
            for j in range(m):
                for k in range(i + 1, n):
                    for l in range(j + 1, m):
                        P += matrix[i][j] * matrix[k][l]
                        Q += matrix[i][l] * matrix[k][j] 

        row_sums = np.transpose(np.sum(matrix, axis=1))
        column_sums = np.sum(matrix, axis=0)

        t1 = 0.5 * np.sum(n_i * (n_i - 1) for n_i in row_sums)
        t2 = 0.5 * np.sum(m_j * (m_j - 1) for m_j in column_sums)  

        tau_b = (P-Q)/math.sqrt((0.5*num*(num-1)-t1)*(0.5*num*(num-1)-t2))

        return tau_b
    
    def coeff_spoluch_steward_foo(matrix, num):
        n, m = matrix.shape
        P = 0
        Q = 0
        for i in range(n):
            for j in range(m):
                for k in range(i + 1, n):
                    for l in range(j + 1, m):
                        P += matrix[i][j] * matrix[k][l]
                        Q += matrix[i][l] * matrix[k][j] 
        tau_st = (2*(P-Q)*min(m,n))/(num**2*min(m,n)-1)
        return tau_st

    def find_a_b_for_lin_reg(self, r):
        x_list = self.calculation_data[0]
        y_list = self.calculation_data[1]

        corr_x_average = sum(x_list)/len(x_list)
        corr_x_sq = math.sqrt(sum([math.pow(x - corr_x_average, 2) for x in x_list]) / len(x_list))

        corr_y_average = sum(y_list)/len(y_list)
        corr_y_sq = math.sqrt(sum([math.pow(y - corr_y_average, 2) for y in y_list]) / len(y_list))

        b = r*(corr_y_sq/corr_x_sq)
        a = corr_y_average - b * corr_x_average
        return a, b, corr_x_sq, corr_y_sq
    
    def find_a_b_for_lin_reg_teyla(self):
        x_list = np.array(self.calculation_data[0])
        y_list = np.array(self.calculation_data[1])
        N = len(x_list)

        b_list = []
        for i in range(N):
            for j in range(i + 1, N):
                b_list.append((y_list[j]-y_list[i])/(x_list[j]-x_list[i]))

        b = np.median(np.array(b_list))

        a = np.median(y_list - b* x_list)

        return a, b
    
    def find_a_b_c_for_hyper(self):
        x_list = np.array(self.calculation_data[0])
        y_list = np.array(self.calculation_data[1])
        n = len(x_list)

        a = sum(y_list)/n

        x_av_loc_hyper = sum(x_list)/n
        b = sum((x_list[i]-x_av_loc_hyper)*y_list[i] for i in range(n))/sum((x_list[i]-x_av_loc_hyper)**2 for i in range(n))

        c = sum(fi_2(x_list)*y_list)/sum((fi_2(x_list))**2)

        return a, b, c

def fi_1(x):
    av = np.mean(x)
    fi_1 = x - av
    return fi_1


def fi_2(x):
    av = np.mean(x)
    fi_1 = x - av
    fi_2 = x**2 - ((np.mean(x**3) - av * np.mean(x**2)) / (np.var(x))) * fi_1 - np.mean(x ** 2)
    return fi_2
        
        











def findM(myN):
    if myN < 100:
        m1 = math.sqrt(myN)
        is_int1 = m1.is_integer()
        if is_int1 == True:
            if getParity(m1) == 0:
                myM = math.floor((m1-1))
                return myM
            else:
                myM = math.floor(m1)
                return myM
        else:
            myM = math.floor(m1)
            return myM
    else:
        m2 = myN ** (1/3)
        is_int2 = m2.is_integer()
        if is_int2 == True:
            if getParity(m2) == 0:
                myM = math.floor((m2-1))
                return myM
            else:
                myM = math.floor(m2)
                return myM
        else:
            myM = math.floor(m2)
            return myM

def getParity(n):
    if type(n) == int:
        return (bin(n).count("1")) % 2
    else:
        return 1
    
def xy_mean(x, y):
    n = len(x)
    xy_mean_num = sum(x[i] * y[i] for i in range(n)) / n
    return xy_mean_num

def get_ranks_for_one_list(input_list):
    input_list = sorted(input_list)
    sorted_indices = sorted(range(len(input_list)), key=lambda k: input_list[k])
    ranks = [sorted_indices.index(i) + 1 for i in range(len(input_list))]

    unique_values = set(input_list)
    for value in unique_values:
        indices = [i for i, x in enumerate(input_list) if x == value]
        if len(indices) > 1:
            average_rank = sum(ranks[i] for i in indices) / len(indices)
            for i in indices:
                ranks[i] = average_rank

    return ranks

def find_MED(number_list):
    N = len(number_list)
    if N % 2 == 0:
        k = int(N/2)
        return (number_list[k]+number_list[k+1])/2
    else:
        k = int((N-1)/2)
        return number_list[k]