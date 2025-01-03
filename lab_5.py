import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import t
from scipy.stats import chi2
import scipy.stats as st
from scipy.stats import kurtosis

class lab_5:
    def __init__(self, choppedARRAYS):
        self.choppedARRAYS = choppedARRAYS

    def calculate_averages(self):
        myMatrix = self.choppedARRAYS
        averages = []
        for array in myMatrix:
            avg = sum(array) / len(array)
            averages.append(avg)
        return averages
    
    def calculate_variances(self):
        variances = []
        for array in self.choppedARRAYS:
            mean = sum(array) / len(array)
            variance = math.sqrt(sum((x - mean) ** 2 for x in array) / (len(array)-1))
            variances.append(variance)
        return variances
    
    def calculate_covariance_matrix(self):
        data_matrix = np.array(self.choppedARRAYS)
        
        cov_matrix = np.cov(data_matrix, rowvar=False)
        
        return cov_matrix
    
    def calculate_std_matrix(self):
        std_devs = self.calculate_variances()
        std_devs = [x ** 2 for x in std_devs]
        size = len(std_devs)
        std_matrix = np.zeros((size, size)) 

        for i in range(size):
            std_matrix[i][i] = std_devs[i]  
            
        for i in range(size):
            for j in range(size):
                if i != j:
                    var_i = calculate_variance_2(self.choppedARRAYS[i])
                    var_j = calculate_variance_2(self.choppedARRAYS[j])
                    corr_ij = correlation_coeff(self.choppedARRAYS[i], self.choppedARRAYS[j])
                    std_matrix[i][j] = (var_i * var_j) * corr_ij

        return std_matrix
    
    def correlation_coeff_for_3(self):
        matrix = self.choppedARRAYS
        i_list = matrix[0]
        j_list = matrix[1]
        c_list = matrix[2]
        
        r_ijc = (correlation_coeff(i_list, j_list)-correlation_coeff(i_list,c_list)*correlation_coeff(j_list,c_list)) / math.sqrt((1-(correlation_coeff(i_list,c_list))**2)*(1-(correlation_coeff(j_list,c_list))**2))
    
        if r_ijc >= 1:
            r_ijc = 0.99
        if r_ijc <= -1:
            r_ijc = -0.99
    
        return r_ijc
    
    def znach_corr_coeff_for_3(self):
        r_ijc = self.correlation_coeff_for_3()
        N = len(self.choppedARRAYS[0])
        w = len(self.choppedARRAYS)
        t = (r_ijc*math.sqrt(N-w-2))/math.sqrt(1-r_ijc**2)
        return t
    
    def dovirchi_intervalu_coeff_cor_for_3(self):
        r_ijc = self.correlation_coeff_for_3()
        N = len(self.choppedARRAYS[0])
        w = len(self.choppedARRAYS)
        u_quantile = stats.t.ppf(0.95, N-w-3)
        
        v1 = 0.5 * np.log((1 + r_ijc) / (1 - r_ijc)) - u_quantile / np.sqrt(N - w - 3)
        v2 = 0.5 * np.log((1 + r_ijc) / (1 - r_ijc)) + u_quantile / np.sqrt(N - w - 3)
        
        down = (np.exp(2*v1)-1)/(np.exp(2*v1)+1)
        up = (np.exp(2*v2)-1)/(np.exp(2*v2)+1)
        
        return down,up
        
    def mult_corr_coef_for_n(self):
        data = np.array(self.choppedARRAYS)
        data_transposed = data.T
        correlation_matrix = np.corrcoef(data_transposed, rowvar=False)
        k = 2
        mask = np.ones(correlation_matrix.shape[0], dtype=bool)
        mask[k] = False
        reduced_matrix = correlation_matrix[mask][:, mask]
        determinant_reduced = np.linalg.det(reduced_matrix)
        determinant = np.linalg.det(correlation_matrix)
        
        r_xn = math.sqrt(1- determinant/determinant_reduced)
        
        if r_xn >= 1:
            r_xn = 0.99
        if r_xn <= -1:
            r_xn = -0.99
        
        return r_xn
    
    def znach_mult_corr_coef_for_n(self):
        N = len(self.choppedARRAYS[0])
        w = len(self.choppedARRAYS)
        r_xn = self.mult_corr_coef_for_n()
        f = (N-w-1)/w *(r_xn**2)/(1-r_xn**2)
        
        return f
    
    def find_A_vector_without_a_0(self):
        matrix = self.choppedARRAYS
        x_matrix = matrix[:-1]
        y_list = matrix[-1]
        X = np.array(x_matrix)
        Y = np.array(y_list)
        Y_T = Y.reshape(-1, 1)
        X_X_T = np.dot(X, X.T)
        X_X_T_inv = np.linalg.inv(X_X_T)
        X_Y_T = np.dot(X, Y_T)
        A = np.dot(X_X_T_inv, X_Y_T)
        A_list = A.flatten().tolist()

        return A_list

    def find_A_vector_with_a_0(self):
        matrix = self.choppedARRAYS
        x_matrix = matrix[:-1]
        y_list = matrix[-1]
        X = np.array(x_matrix)
        Y = np.array(y_list)

        means_X = np.mean(X, axis=1)

        mean_Y = np.mean(Y)
        means_X_list = means_X.tolist()
        X = X - means_X[:, np.newaxis]
        Y = Y - mean_Y
        
        Y_T = Y.reshape(-1, 1)
        X_X_T = np.dot(X, X.T)
        X_X_T_inv = np.linalg.inv(X_X_T)
        X_Y_T = np.dot(X, Y_T)        
        A = np.dot(X_X_T_inv, X_Y_T)
        A_list = A.flatten().tolist()
        
        mean_X_list = np.array(means_X_list)
    
        sum_A_mean_X = np.sum(A_list * mean_X_list)
        result = mean_Y - sum_A_mean_X
        
        A_list_with_result = np.insert(A_list, 0, result)
        
        return A_list_with_result
    
    def znach_regression(self):
        N = len(self.choppedARRAYS[0])
        w = len(self.choppedARRAYS)
        R = self.mult_corr_coef_for_n()
        
        f = R**2/(1-R**2) * (N-w-1-1)/(w-1)
        
        return f
    

    def znach_a(self):
        A_vectors_without_a0 = self.find_A_vector_without_a_0()
        A = np.array(A_vectors_without_a0)
        matrix = self.choppedARRAYS
        x_matrix = matrix[:-1]
        y_list = matrix[-1]
        X = np.array(x_matrix)
        Y = np.array(y_list)
        AT_X = np.dot(A.T, X)
        Y_minus_AT_X = Y - AT_X
        sigma_sq = np.dot(Y_minus_AT_X, Y_minus_AT_X.T)
        sigma = math.sqrt(sigma_sq)
        
        X_XT = np.dot(X, X.T)
        X_XT_inv = np.linalg.inv(X_XT)
        
        diagonal_elements = np.diag(X_XT_inv).tolist()
        
        t_znach = A / (sigma * np.sqrt(diagonal_elements))
        
        t_znach = t_znach.flatten().tolist()
        
        N = len(self.choppedARRAYS[0])
        w = len(self.choppedARRAYS)
        
        df = N - w - 1
    
        t_quantile = t.ppf(1 - 0.05, df)
        
        down_list = A - t_quantile * sigma * np.sqrt(diagonal_elements)
        
        up_list = A + t_quantile * sigma * np.sqrt(diagonal_elements)
        
        return t_znach, down_list, up_list
        
    def standart_param(self):
        A_vectors_without_a0 = self.find_A_vector_without_a_0()
        A = np.array(A_vectors_without_a0)
        matrix = self.choppedARRAYS
        x_matrix = matrix[:-1]
        y_list = matrix[-1]
        
        sigma_x_list = []
        
        for sample in x_matrix:
            sigma_x = math.sqrt(variance(sample))
            sigma_x_list.append(sigma_x)
        
        y_sigma = math.sqrt(variance(y_list))
        
        sig_x = np.array(sigma_x_list)
        
        a_standart = (A*sig_x)/y_sigma
        
        return a_standart
    
    def deter_coeff(self):
        R = (self.mult_corr_coef_for_n())**2
        return R
    
    def dov_interv_sigma(self):
        A_vectors_without_a0 = self.find_A_vector_without_a_0()
        A = np.array(A_vectors_without_a0)
        matrix = self.choppedARRAYS
        x_matrix = matrix[:-1]
        y_list = matrix[-1]
        X = np.array(x_matrix)
        Y = np.array(y_list)
        AT_X = np.dot(A.T, X)
        Y_minus_AT_X = Y - AT_X
        sigma_sq = np.dot(Y_minus_AT_X, Y_minus_AT_X.T)
        N = len(self.choppedARRAYS[0])
        w = len(self.choppedARRAYS)
        
        alf1 = (1+(1-0.05))/2
        alf2 = (1-(1-0.05))/2
        
        down = (sigma_sq*(N-w-1))/(chi2.ppf(1 - alf2, N-w-1))
        up = (sigma_sq*(N-w-1))/(chi2.ppf(1 - alf1, N-w-1))
        
        return sigma_sq, down, up
    
    def dov_intrv_regression(self,string_nums_inp):
        string_list = string_nums_inp.split()
        x_list_input = [int(num) for num in string_list]
        
        A_vectors_without_a0 = self.find_A_vector_without_a_0()
        A = np.array(A_vectors_without_a0)
        matrix = self.choppedARRAYS
        x_matrix = matrix[:-1]
        y_list = matrix[-1]
        X = np.array(x_matrix)
        Y = np.array(y_list)
        AT_X = np.dot(A.T, X)

        Y_minus_AT_X = Y - AT_X
        sigma_sq = np.dot(Y_minus_AT_X, Y_minus_AT_X.T)
        sigma = math.sqrt(sigma_sq)
        C = np.dot(X, X.T)
        C_ready = np.linalg.inv(C)
        
        N = len(self.choppedARRAYS[0])
        w = len(self.choppedARRAYS)
        
        df = N - w - 1
    
        t_quantile = t.ppf(1 - 0.05, df)
        
        X_NEW_list = np.array(x_list_input)
                
        y_down = np.dot(A.T, X_NEW_list) - t_quantile*(sigma)*math.sqrt(1+np.dot(np.dot(X_NEW_list.T, C_ready), X_NEW_list))
        y_up = np.dot(A.T, X_NEW_list) + t_quantile*(sigma)*math.sqrt(1+np.dot(np.dot(X_NEW_list.T, C_ready), X_NEW_list))
        
        y_inside = np.dot(A.T, X_NEW_list)
        
        str_message_box = f'Довірчий інтервал для точки {round(y_inside,3)}\n'
        str_message_box += f'[{round(y_down,3)}] ; [{round(y_up)}]\n'
        
        return str_message_box
        
    def dots_for_diagnostic_diagram(self):
        a_vector = self.find_A_vector_without_a_0()
        matrix = self.choppedARRAYS
        x_matrix = matrix[:-1]
        y_list = matrix[-1]
        
        a_vector = np.array(a_vector)
        x_matrix = np.array(x_matrix)
        y_list = np.array(y_list)

        eps_l = []
        y_av_list = []
        
        for i in range(len(y_list)):
            y_av =  np.dot(a_vector, (x_matrix[:, i]))
            y_av_list.append(y_av)

        for i in range(len(y_list)):
            epsilon =  y_list[i] - np.dot(a_vector, (x_matrix[:, i]))
            eps_l.append(epsilon)


        return y_list, eps_l
        
            
    def yakobi(self):
        eps = 0.01
        max_it = 50
        #A = correlation_matrix(self.choppedARRAYS)
        A = self.calculate_std_matrix()
        n = len(A)
        H = [[0] * n for _ in range(n)]
        D = [[0] * n for _ in range(n)]
        H1 = [[0] * n for _ in range(n)]
        V = [[0] * n for _ in range(n)]
        temp = eps
        E = 0
        while temp % 10 != 0:
            temp *= 10
            E += 1

        k1 = 0
        text = ""
        max = 1
        while (abs(max) > eps and k1 < max_it):
            i1 = 0
            j1 = 0
            j = 1
            max = 0
            for i in range(n):
                while (j < n):
                    if (abs(A[i][j]) > abs(max)):
                        max = A[i][j]
                        i1 = i
                        j1 = j
                    j += 1
                j = i + 1
                j += 1
            try:
                angle = ((2 * A[i1][j1]) / (A[i1][i1] - A[j1][j1]))
            except:
                angle = math.inf
            arctan = (math.atan(angle)) / 2
            arctan_angle = round((180 * arctan) / math.pi)
            text += f"k = {k1}\n"
            text += f"a_i_j максимальне = {max:.{len(str(eps)) - 2}f}\n"
            text += f"Кут повороту: {arctan_angle}\n"
            for i in range(n):
                for j11 in range(n):
                    if (i == j11):
                        H[i][j11] = 1
                    else:
                        H[i][j11] = 0
            H[i1][i1] = math.cos(arctan)
            H[j1][j1] = math.cos(arctan)
            H[i1][j1] = (-1) * math.sin(arctan) 
            H[j1][i1] = math.sin(arctan)  
            for i in range(n):
                for j11 in range(n):
                    H1[i][j11] = H[i][j11]
            text += "Mатриця H:\n"
            for i in range(n):
                for j11 in range(n):
                    formatted_value = f"{H[i][j11]     :.{len(str(eps)) - 2}f}  "
                    text += f"{formatted_value}"
                text += "\n"
            Obernena(H, n) 
            text += f"Обернена матриця H:\n"
            for i in range(n):
                for j2 in range(n):
                    formatted_value = f"{H[i][j2]     :.{len(str(eps)) - 2}f}   "
                    text += f"{formatted_value}"
                text += "\n"
            X = Multiplying(H, A)
            A = Multiplying(X, H1)
            text += "Нова матриця A:\n"
            for i in range(n):
                for j2 in range(n):
                    formatted_value = f"{A[i][j2]:.{len(str(eps)) - 2}f}    "
                    text += f"{formatted_value}"
                text += "\n"
            if (k1 < 1):
                for i in range(n):
                    for j11 in range(n):
                        V[i][j11] = H1[i][j11]

            if (k1 > 0):
                V = Multiplying(V, H1)
            max = 0
            j = 1
            for i in range(n):
                for j in range(j, n):
                    if (abs(A[i][j]) > abs(max)):
                        max = A[i][j]
                j = i + 1
                j += 1
            k1 += 1
            text += "\n"
        
        eigenvalues = []
        text += f"Власні значення:\n"
        for i in range(n):
            formatted_value = f"{A[i][i]:.{len(str(eps))+4}f}"
            eigenvalues.append(float(formatted_value))
            text += f"labm{i+1} = {formatted_value}\n"
        text += f"\nВласні вектори V = P:\n"
        for i in range(n):
            for j in range(n):
                text += f"{V[i][j]:.{len(str(eps))+4}f}     "
            text += "\n"
        text += f"Матриця D:\n"
        for i in range(n):
            D[i][i] = A[i][i]

        for i in range(n):
            for j in range(n):
                text += f"{D[i][j]:.{len(str(eps))+4}f}     "
            text += "\n"
            
        sorted_indices = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)

        V_sorted = [[V[row][i] for i in sorted_indices] for row in range(len(V))]

        sorted_lambda = [eigenvalues[i] for i in sorted_indices]
                    
        fractions = [value*100 / n for value in sorted_lambda]
        cumulative_sums = [sum(fractions[:i+1]) for i in range(len(fractions))]
                
        return sorted_lambda, V_sorted, fractions, cumulative_sums
    
    def find_new_x(self):
        matrix = correlation_matrix(self.choppedARRAYS)
        eigenvalues, V, fractions, cumulative_sums = self.yakobi()
        
        #stand_chopped = standardize(self.choppedARRAYS)
        
        X = np.array(self.choppedARRAYS)
        Alf = (np.array(V))
        
        new_args = []
        
        for i in range(len(X)):
            new_x = np.zeros_like(X[i])
            for j in range(len(X)):
                new_x += Alf[j][i]*X[j]
            new_args.append(new_x)
        
        return new_args
        
        
    def find_recon_x(self):
        matrix = self.choppedARRAYS
        eigenvalues, V, fractions, cumulative_sums = self.yakobi()
        
        X = np.array(self.find_new_x())
        Alf = (np.array(V))
        
        new_args = []
        
        for i in range(len(X)):
            new_x = np.zeros_like(X[i])
            for j in range(len(X)):
                new_x += Alf[i][j]*X[j]    
            new_args.append(new_x)
        
        return new_args
    
    def find_recon_w_x(self,w):
        matrix = self.choppedARRAYS
        eigenvalues, V, fractions, cumulative_sums = self.yakobi()
        
        X = np.array(self.find_new_x())
        Alf = (np.array(V))
        
        new_args = []
        
        for i in range(len(X)):
            new_x = np.zeros_like(X[i])
            for j in range(w):
                new_x += Alf[i][j]*X[j]     
            new_args.append(new_x)
        
        return new_args
        
    def hipotesa_2_means(args):
        x, y = args[0], args[1]
        n = len(x)
        n1, n2 = len(x[0]), len(y[0])
        s_0 = np.zeros((n, n), dtype=float)
        s_1 = np.zeros((n, n), dtype=float)

        n_with_2 = 1 / (n1 + n2 - 2)
        n_none_2 = 1 / (n1 + n2)

        for i in range(n):
            for j in range(n):
                s_0[i, j] = n_with_2 * (sum(x[i] * x[j]) + sum(y[i] * y[j]) -
                                        n_none_2 * (sum(x[i]) + sum(y[i])) * (sum(x[j]) + sum(y[j])))

        for i in range(n):
            for j in range(n):
                s_1[i, j] = n_with_2 * (sum(x[i] * x[j]) + sum(y[i] * y[j]) -
                                        1 / n1 * sum(x[i]) * sum(x[j]) - 1 / n2 * sum(y[i]) * sum(y[j]))

        s0_det, s1_det = np.linalg.det(s_0), np.linalg.det(s_1)
        
        v = -(n1 + n2 - 2 - n / 2) * np.log(s1_det / s0_det)

        return v, n
    

    def hipotesa_k_means(args):
        k = len(args)
        n = len(args[0])

        means = [[] for _ in range(k)]
        for i in range(k):
            for j in range(len(args[i])):
                means[i].append([np.mean(args[i][j])])

        nd = [len(args[i][0]) for i in range(k)]
        sd = [[] for _ in range(k)]
        for i in range(k):
            si = 1 / (len(args[i][0]) - 1) * (args[i] - means[i]) @ (args[i] - means[i]).T
            sd[i].append(si)

        x_mean = np.linalg.inv(np.sum(np.array([nd[i] * np.linalg.inv(sd[i]) for i in range(k)]), axis=(0, 1))) @ np.sum(np.array([nd[i] * np.linalg.inv(sd[i]) @ means[i] for i in range(k)]), axis=(0, 1))
        v = np.sum(np.array(
            [nd[i] * (means[i] - x_mean).T @ np.linalg.inv(sd[i]) @ (means[i] - x_mean) for i in range(k)]),
            axis=(0, 1))

        v_hi_sq = n * (k - 1)

        v_num = v[0][0]
        
        return v_num, v_hi_sq


    def hipotesa_k_dk(args):
        k = len(args)
        n = len(args[0])

        means = [[] for _ in range(k)]
        for i in range(k):
            for j in range(len(args[i])):
                means[i].append([np.mean(args[i][j])])

        s_d = [[] for _ in range(k)]
        for i in range(k):
            si = 1 / (len(args[i][0]) - 1) * (args[i] - means[i]) @ (args[i] - means[i]).T
            s_d[i].append(si)
        
        nd = [len(args[i][0]) for i in range(k)]
        total_n = 0
        for i in range(k):
            total_n += nd[i]

        s = (1 / (total_n - k)) * np.sum(np.array([(nd[i] - 1) * s_d[i] for i in range(k)]), axis=(0, 1))
        v = np.sum(np.array([(nd[i] - 1) / 2 * np.log(np.linalg.det(s) / np.linalg.det(s_d[i])) for i in range(k)]),
                axis=(0, 1))

        v_num = n * (n + 1) * (k - 1) / 2

        return v, v_num

    def del_anomaly(my_lists):
        indices_to_remove = set()
        
        for i, my_list in enumerate(my_lists):
            s = np.sqrt(np.var(my_list, ddof=1))
            mean = np.mean(my_list)
            e = kurtosis(my_list)
            my_line_kurt = 1 / (math.sqrt(abs(e)))

            t = 1.2 + 3.6 * math.sqrt(abs(1 - my_line_kurt)) * math.log10(len(my_list) / 10)

            a = mean - t * s
            b = mean + t * s

            for index, value in enumerate(my_list):
                if not (a <= value <= b):
                    indices_to_remove.add(index)
        
        cleaned_lists = []
        for my_list in my_lists:
            new_list = [value for index, value in enumerate(my_list) if index not in indices_to_remove]
            cleaned_lists.append(new_list)
        
        return cleaned_lists

    def del_anomaly_2(my_lists):
        def remove_anomalies(my_list):
            s = np.sqrt(np.var(my_list, ddof=1))
            mean = np.mean(my_list)
            e = kurtosis(my_list)
            my_line_kurt = 1 / (math.sqrt(abs(e)))

            t = 1.2 + 3.6 * math.sqrt(abs(1 - my_line_kurt)) * math.log10(len(my_list) / 10)

            a = mean - t * s
            b = mean + t * s

            clear_values = [value for value in my_list if a <= value <= b]
            return clear_values

        cleaned_lists = [remove_anomalies(my_list) for my_list in my_lists]
                
        return cleaned_lists


def Obernena(arr: list[list[float]], size: int) -> None:
    for i in range(size):
        for j in range(i):
            tmp = arr[i][j]
            arr[i][j] = arr[j][i]
            arr[j][i] = tmp

def Multiplying(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if (len(a[0]) != len(b)):
        raise Exception("Матриці не можна перемножити")
    r = [[0] * len(b) for _ in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                r[i][j] += a[i][k] * b[k][j]
    return r
    
def mean(lst):
    return sum(lst) / len(lst)

def variance(lst):
    m = mean(lst)
    return sum((x - m) ** 2 for x in lst) / len(lst)

def mean(data):
    return sum(data) / len(data)

def covariance(x, y):
    mean_x = mean(x)
    mean_y = mean(y)
    cov = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y)) / (len(x) - 1)
    return cov

def std_dev(data):
    mean_data = mean(data)
    variance = sum((x - mean_data) ** 2 for x in data) / (len(data) - 1)
    return variance ** 0.5

def standardize(samples):
    standardized_samples = []
    for sample in samples:
        mean = np.mean(sample)
        variance = np.var(sample)
        std_dev = np.sqrt(variance)
        if std_dev == 0:
            standardized_samples.append([0] * len(sample))
        else:
            standardized_samples.append([(x - mean) / std_dev for x in sample])
    return standardized_samples

def correlation_matrix(samples):
    n = len(samples)
    corr_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i][j] = 1  
            else:
                cov = covariance(samples[i], samples[j])
                std_i = std_dev(samples[i])
                std_j = std_dev(samples[j])
                corr_matrix[i][j] = cov / (std_i * std_j)
    
    return corr_matrix


def correlation_coeff(x_list, y_list):
    n = len(x_list)
    corr_x_average = mean(x_list)
    corr_x_sq = math.sqrt(variance(x_list))
    corr_y_average = mean(y_list)
    corr_y_sq = math.sqrt(variance(y_list))
    corr_xy_mean = xy_mean(x_list, y_list)
    
    r_x_y = n / (n - 1) * (corr_xy_mean - corr_x_average * corr_y_average) / (corr_x_sq * corr_y_sq)
    
    return r_x_y

def xy_mean(x, y):
    n = len(x)
    return sum(x[i] * y[i] for i in range(n)) / n    
    
def correlation_coeff(x_list, y_list):
    n = len(x_list)
    corr_x_average = np.mean(x_list)
    corr_x_sq = np.sqrt(np.sum(np.power(x_list - corr_x_average, 2)) / len(x_list))
    corr_y_average = np.mean(y_list)
    corr_y_sq = np.sqrt(np.sum(np.power(y_list - corr_y_average, 2)) / len(y_list))
    corr_xy_mean = xy_mean(x_list,y_list)

    r_x_y = n/(n-1) * (corr_xy_mean-corr_x_average*corr_y_average)/(corr_x_sq*corr_y_sq)

    return r_x_y
    
def xy_mean(x, y):
    n = len(x)
    xy_mean_num = sum(x[i] * y[i] for i in range(n)) / n
    return xy_mean_num

def calculate_variance_2(array):
    N = len(array)
    mean = sum(array) / N
    variance = math.sqrt(sum((xi - mean) ** 2 for xi in array) / (N - 1))
    return variance
