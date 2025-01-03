from tkinter import messagebox
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.ttk import *
import math
import os
from sklearn.svm import OneClassSVM
from tkinter import BooleanVar, Listbox, StringVar, Variable, filedialog as fd
from scipy.interpolate import interp1d
import scipy.special
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from math import fabs, sqrt
import copy
from scipy.optimize import fsolve
from CalculationData import CalculationData
from matplotlib.patches import Rectangle
from HomogeneityIndependence import HomogeneityIndependence
from CorrelationalRegression import CorrelationalRegression
from CorrelationalRegression import fi_1, fi_2
from lab_5 import lab_5
from FindSquare import FindSquare
from tabulate import tabulate

calculation_data = CalculationData([])
calculation_data.array = [[], []]

from scipy.stats import f, norm, chi2, t
import scipy.stats as stats


ARRAYS = []

start_column = -1
end_column = -1

choppedARRAYS = []

rel_freq_calc,rel_freq_calc_2 = [],[]
M = 0
corr_x_sq, corr_y_sq = 0,0
alfa_for_qwant = 0.05
z_list_for_graph = []
str_for_new_x = ""
listBoxCoef = []
anwer_for_correlational_regression = ""
text_for_box_m_n_table = []
lines = []
lines_y_emp = []
height = []
height_emp = []
krok = []
left = []
lines_min_max = []
temp_0001 = []
temp_0002 = []
temp_0003 = []
temp_0004 = []
filename = ''
file_name = ''
x_average = 0
x_sq_avg = 0
coun_kurtosis = 0
asym_coef = 0
kurtosis = 0
pirson = 0
x_2_average = 0
list_class_borders = []
ni_original = []
param_expon_lambda = 0 
param_lognorm_m = 0
param_lognorm_sigma = 0 
param_normal_m = 0
param_normal_sigma = 0 
param_veib_alfa = 0
param_veib_beta = 0
param_rivnom_a = 0
param_rivnom_b = 0

answer_for_lab_5_1 = ""

parametric_values = []
parametric_values_hist = []
confidence_intervals_1 = []
confidence_intervals_2 = []

myArrays_lab_5 = lab_5(choppedARRAYS)

cuass_val = False
linear_val = False
hyperbolic = False

root = tk.Tk()
w_window= root.winfo_screenwidth() 
h_window= root.winfo_screenheight()
root.geometry("%dx%d" % (w_window, h_window))
tabControl = Notebook(root,width=1500)


def my_w_child_open():
    global variant_class, variant_graph, variant_param, variant_confidence_intervals, c_v1, c_v2, c_v3, c_v4

    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("300x200")
    my_w_child.title("Варіанти")

    def swith_class():
        global variant_class, c_v1
        variant_class = not variant_class
        c_v1.set(variant_class)

    def swith_graph():
        global variant_graph, c_v2
        variant_graph = not variant_graph
        c_v2.set(variant_graph)

    def swith_param():
        global variant_param, c_v3
        variant_param = not variant_param
        c_v3.set(variant_param)

    def swith_confidence_intervals():
        global variant_confidence_intervals, c_v4
        variant_confidence_intervals = not variant_confidence_intervals
        c_v4.set(variant_confidence_intervals)

    def my_w_child_on_closing():
        my_w_child.destroy()
        click_graph_emp()

    my_w_child.protocol("WM_DELETE_WINDOW", my_w_child_on_closing)
    ch_class = tk.Checkbutton(
        my_w_child, variable=c_v1, text="Класи", command=swith_class)
    ch_class.grid(row=1, column=2)
    ch_graph = tk.Checkbutton(
        my_w_child, variable=c_v2, text="Графік емпіричний", command=swith_graph)
    ch_graph.grid(row=2, column=2)
    ch_param = tk.Checkbutton(
        my_w_child, variable=c_v3, text="Графік відтворений", command=swith_param)
    ch_param.grid(row=3, column=2)
    ch_confidence_intervals = tk.Checkbutton(
        my_w_child, variable=c_v4, text="Довірчі інтервали", command=swith_confidence_intervals)
    ch_confidence_intervals.grid(row=4, column=2)


def my_w_child_4_open():
    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("300x200")
    my_w_child.title("Ввести назву файла")
 
    entry_name_save_file = tk.Entry(my_w_child)
    entry_name_save_file.place(x=10,y=10)
    entry_name_save_file.insert(0, file_name.split('/')[-1])

    def save_current_file():
        with open(entry_name_save_file.get(), 'w') as file:
            for item in lines:
                file.write(f"{item}\n")

    button_for_save = tk.Button(my_w_child,text = "Зберегати", command=save_current_file).place(x=140,y=10)


def my_w_child_3_open():
    global entry_name_save_file

    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("300x200")
    my_w_child.title("Варіанти гістограми")

    def swith_hist():
        global hist_variant_hist, cc_v1
        hist_variant_hist = not hist_variant_hist
        cc_v1.set(hist_variant_hist)

    def swith_density():
        global hist_variant_density, cc_v2
        hist_variant_density = not hist_variant_density
        cc_v2.set(hist_variant_density)

    def swith_param():
        global hist_variant_param, cc_v3
        hist_variant_param = not hist_variant_param
        cc_v3.set(hist_variant_param)

    def my_w_child_on_closing():
        my_w_child.destroy()
        click_graph()

    my_w_child.protocol("WM_DELETE_WINDOW", my_w_child_on_closing)
    ch_density = tk.Checkbutton(
        my_w_child, variable=cc_v2, text="Функція щільності", command=swith_density)
    ch_density.grid(row=2, column=2)
    ch_param = tk.Checkbutton(
        my_w_child, variable=cc_v3, text="Функція щільності відтворена", command=swith_param)
    ch_param.grid(row=3, column=2)

def second_coord(N, number_list, lamb):
    list_of_coord = [0.0] * N
    for i in range(N):
        list_of_coord[i] = (0.95/N)*i

    phi_x = [0.0] * N
    for i in range(N):
        phi_x[i] = math.log(1/(1-list_of_coord[i]))

    return phi_x

def find_horizontal_coordinates(num_points, start=0.05, stop=0.95):
    num_points = int(num_points/30)
    my_list = []
    factor = (stop / start) ** (1 / (num_points - 1))
    
    current_value = start
    for _ in range(num_points):
        my_list.append(current_value)
        current_value *= factor
    return my_list

def my_w_child_2_open():
    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("600x600")
    my_w_child.title("Імовірнісна сітка")

    figure11 = plt.Figure(figsize=(6, 5), dpi=100)
    ax11 = figure11.add_subplot(111)
    ax11.set_title('Імовірнісна сітка')
    bar11 = FigureCanvasTkAgg(figure11, master=my_w_child)
    bar11.get_tk_widget().grid(row=0, column=0)
    toolbar1 = NavigationToolbar2Tk(bar11, my_w_child, pack_toolbar=False)
    toolbar1.update()
    toolbar1.grid(row=1, column=0, columnspan=2)

    N = len(lines)
    horiz_nums = lines
    vertic_nums = second_coord(N, lines, lamb=7.75795)
    max_vert_num = max(vertic_nums)
    vertic_nums = [x / max_vert_num for x in vertic_nums]
    y_coords = find_horizontal_coordinates(N)

    plt.figure(figsize=(8, 6))

    for y in y_coords:
        ax11.axhline(y=y, color='black', linestyle='-')
    ax11.plot(horiz_nums, vertic_nums, 'o', color="red")
    ax11.set_xlabel('x')
    ax11.set_ylabel('φ(x)')
    ax11.set_title('Імовірнісна сітка')
    ax11.legend().remove()
    ax11.grid(True)
    bar11.draw()


def create_file():
    global filename, distribution_options_var, x_average

    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("500x800")
    my_w_child.title("Створити файл")

    method_choice = tk.IntVar()
    def select_action():
        num_choice = method_choice.get()
        return num_choice

    def create():
        xx = []
        size = int(N_ent.get())
        y = np.random.uniform(0, 1, size)
        if select_action() == 3:
            a = float(rivnom_a.get())
            b = float(rivnom_b.get())
            xx = [a + un_num*b - un_num*a for un_num in y]
        elif select_action() == 4:
            sig = float(norm_sig.get())
            m = float(norm_m.get())
            ro = 0.2316419
            b_1 = 0.31938153
            b_2 = -0.356563782
            b_3 = 1.781477937
            b_4 = -1.821255978
            b_5 = 1.330274429
            e_u = 7.8*(10)**(-8)
            def equation_to_solve(x, y_value):
                return y_value - (1 / (np.sqrt(2 * np.pi))) * np.exp(-(x**2 - 2 * m * x + m**2) / (2 * sig**2)) * (
                        b_1 * (sig / (sig + x * ro - m * ro)) +
                        b_2 * (sig / (sig + x * ro - m * ro))**2 +
                        b_3 * (sig / (sig + x * ro - m * ro))**3 +
                        b_4 * (sig / (sig + x * ro - m * ro))**4 +
                        b_5 * (sig / (sig + x * ro - m * ro))**5) + e_u
            for y_value in y:
                x_solution = fsolve(equation_to_solve, x0=0.0, args=(y_value,))
                xx.append(x_solution[0])
        elif select_action() == 1:
            lam = float(expon_lamb.get())
            xx = [-np.log(1-un_num)/lam for un_num in y]
        elif select_action() == 2:
            sig = float(lognorm_sigma.get())
            m = float(lognorm_m.get())
            xx = [np.exp(sig*np.sqrt(2)*scipy.special.erfinv(2*un_num-1)+m) for un_num in y]
        elif select_action() == 5:
            alf = float(veib_alf.get())
            bet = float(veib_bet.get())
            xx = [(-alf*np.log(1-un_num))**(1/bet) for un_num in y]
        elif select_action() == 6:
            t_n = int(T_N_ent.get())
            input_lambd = float(expon_lamb.get())
            xx.append(size)
            count = 0
            while count <= t_n-1:
                t_test_un_ar = np.random.uniform(0, 1, size)
                x_exp = [-np.log(1-un_num)/input_lambd for un_num in t_test_un_ar]
                x_exp = sorted(x_exp)
                x_av = (sum(x_exp)/size)
                lamd_calc = 1/x_av
                disp_lamba = lamd_calc**2/size
                sigma_lamb = np.sqrt(disp_lamba)
                t_test = (input_lambd - lamd_calc)/sigma_lamb
                xx.append(t_test)
                count = count + 1
        if len(xx) > 0:
            with open(filename_ent.get(), 'w') as file:
                for number in xx:
                    file.write(str(number) + '\n')

    Label(my_w_child, text='Кількість елементів: ').place(x = 10, y = 1)
    N_ent = Entry(my_w_child)
    N_ent.place(x = 10, y = 20)

    Label(my_w_child, text='Кількість т-тесті: ').place(x = 200, y = 185)
    T_N_ent = Entry(my_w_child)
    T_N_ent.place(x = 200, y = 210)

    Label(my_w_child, text='Ім\'я файлу: ').place(x = 10, y = 50)
    filename_ent = Entry(my_w_child)
    filename_ent.place(x = 10, y = 70)
    filename_ent.insert(0, filename.split('/')[-1])

    Button(my_w_child, text='Створити', command=create).place(x = 10, y = 100)

    Label(my_w_child, text='Експоненціальний').place(x = 10, y = 140)
    expon_lamb = Entry(my_w_child)
    expon_lamb.place(x = 32, y = 170)
    Label(my_w_child, text= "λ =").place(x=10,y=170)

    Label(my_w_child, text='Логнормальний').place(x = 10, y = 200)
    lognorm_m = Entry(my_w_child)
    lognorm_m.place(x = 33, y = 230)
    Label(my_w_child, text= "m =").place(x=10,y=230)
    lognorm_sigma = Entry(my_w_child)
    lognorm_sigma.place(x = 33, y = 260)
    Label(my_w_child, text= "σ =").place(x=10,y=260)

    Label(my_w_child, text='Рівномірний').place(x = 10, y = 290)
    rivnom_a = Entry(my_w_child)
    rivnom_a.place(x = 33, y = 310)
    Label(my_w_child, text= "a =").place(x=10,y=310)
    rivnom_b = Entry(my_w_child)
    rivnom_b.place(x = 33, y = 340)
    Label(my_w_child, text= "b =").place(x=10,y=340)

    Label(my_w_child, text='Нормальний').place(x = 10, y = 370)
    norm_sig = Entry(my_w_child)
    norm_sig.place(x = 33, y = 400)
    Label(my_w_child, text= "σ =").place(x=10,y=400)
    norm_m = Entry(my_w_child)
    norm_m.place(x = 33, y = 430)
    Label(my_w_child, text= "m =").place(x=10,y=430)

    Label(my_w_child, text='Вейбул').place(x = 10, y = 470)
    veib_alf = Entry(my_w_child)
    veib_alf.place(x = 33, y = 500)
    Label(my_w_child, text= "α =").place(x=10,y=500)
    veib_bet = Entry(my_w_child)
    veib_bet.place(x = 33, y = 530)
    Label(my_w_child, text= "β =").place(x=10,y=530)

    rad_but_exp = Radiobutton(my_w_child,variable=method_choice,value=1, text="Експоненціальний")
    rad_but_lognorm = Radiobutton(my_w_child,variable=method_choice,value=2,text="Логнормальний")
    rad_but_rinvom = Radiobutton(my_w_child,variable=method_choice,value=3,text="Рівномірний")
    rad_but_norm = Radiobutton(my_w_child,variable=method_choice,value=4,text="Нормальниий")
    rad_but_veib = Radiobutton(my_w_child,variable=method_choice,value=5,text="Вейбула")
    rad_but_t_test = Radiobutton(my_w_child,variable=method_choice,value=6,text="T-тести для експоненціального")
    rad_but_exp.place(x=200,y=5)
    rad_but_lognorm.place(x=200,y=35)
    rad_but_rinvom.place(x=200,y=65)
    rad_but_norm.place(x=200, y=95)
    rad_but_veib.place(x=200, y=125)
    rad_but_t_test.place(x=200,y=155)


def change_column():
    global lines, listBoxCoef, filename, list_class_borders, ni_original, temp_0001, temp_0002,temp_0003,temp_0004
    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("780x500")
    my_w_child.title("Змінити стовпець")

    ent_data = StringVar(my_w_child, value=0)

    main_frame = tk.Frame(my_w_child)
    main_frame.grid(row=0, column=0)

    Entry(main_frame, textvariable=ent_data).pack()

    def on_confirm():
        global lines, listBoxCoef, filename, list_class_borders, ni_original, temp_0001, temp_0002,temp_0003,temp_0004
        try:
            pos = int(ent_data.get())
            lines = ARRAYS[pos][:]
            my_w_child.destroy()
            temp_0001 = []
            temp_0002 = []
            temp_0003 = []
            temp_0004 = []
            ni_original = []
            list_class_borders = []
        except:
            messagebox.showerror("Помика", "Помилка")
        main()
        win(lines, listBoxCoef)

    Button(main_frame, text='Підтвердити', command=on_confirm).pack()

    listvars = []
    canvas = tk.Canvas(my_w_child)
    child_frame = tk.Frame(canvas)
    canvas.create_window(0, 0, window=child_frame, anchor="nw")
    def update_size(e=None):
        canvas["scrollregion"] = canvas.bbox("all")
    canvas.bind('<Configure>', update_size)
    canvas.after_idle(update_size)
    canvas.grid(row=1, column=0)
    for i in range(len(ARRAYS)):
        listvars.append(Variable(my_w_child, value=ARRAYS[i]))
        Label(child_frame, text=str(i)).grid(row=0, column=i)
        Listbox(child_frame, listvariable=listvars[i]).grid(row=1, column=i)
    scroll_x = tk.Scrollbar(my_w_child, orient="horizontal", command=canvas.xview)
    scroll_x.grid(row=2, column=0, sticky="ew")


def change_column_2_x():
    global calculation_data
    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("780x500")
    my_w_child.title("Змінити стовпець x")

    ent_data = StringVar(my_w_child, value=0)

    main_frame = tk.Frame(my_w_child)
    main_frame.grid(row=0, column=0)

    Entry(main_frame, textvariable=ent_data).pack()

    def on_confirm():
        global calculation_data
        try:
            pos = int(ent_data.get())
            calculation_data.array[0] = ARRAYS[pos][:]
            my_w_child.destroy()
        except:
            messagebox.showerror("Помика", "Помилка")
        main()
        win(lines, listBoxCoef)

    Button(main_frame, text='Підтвердити', command=on_confirm).pack()

    listvars = []
    canvas = tk.Canvas(my_w_child)
    child_frame = tk.Frame(canvas)
    canvas.create_window(0, 0, window=child_frame, anchor="nw")
    def update_size(e=None):
        canvas["scrollregion"] = canvas.bbox("all")
    canvas.bind('<Configure>', update_size)
    canvas.after_idle(update_size)
    canvas.grid(row=1, column=0)
    for i in range(len(ARRAYS)):
        listvars.append(Variable(my_w_child, value=ARRAYS[i]))
        Label(child_frame, text=str(i)).grid(row=0, column=i)
        Listbox(child_frame, listvariable=listvars[i]).grid(row=1, column=i)
    scroll_x = tk.Scrollbar(my_w_child, orient="horizontal", command=canvas.xview)
    scroll_x.grid(row=2, column=0, sticky="ew")


def change_column_2_y():
    global calculation_data
    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("780x500")
    my_w_child.title("Змінити стовпець y")

    ent_data = StringVar(my_w_child, value=0)

    main_frame = tk.Frame(my_w_child)
    main_frame.grid(row=0, column=0)

    Entry(main_frame, textvariable=ent_data).pack()

    def on_confirm():
        global calculation_data
        try:
            pos = int(ent_data.get())
            calculation_data.array[1] = ARRAYS[pos][:]
            my_w_child.destroy()
        except:
            messagebox.showerror("Помика", "Помилка")
        main()
        win(lines, listBoxCoef)

    Button(main_frame, text='Підтвердити', command=on_confirm).pack()

    listvars = []
    canvas = tk.Canvas(my_w_child)
    child_frame = tk.Frame(canvas)
    canvas.create_window(0, 0, window=child_frame, anchor="nw")
    def update_size(e=None):
        canvas["scrollregion"] = canvas.bbox("all")
    canvas.bind('<Configure>', update_size)
    canvas.after_idle(update_size)
    canvas.grid(row=1, column=0)
    for i in range(len(ARRAYS)):
        listvars.append(Variable(my_w_child, value=ARRAYS[i]))
        Label(child_frame, text=str(i)).grid(row=0, column=i)
        Listbox(child_frame, listvariable=listvars[i]).grid(row=1, column=i)
    scroll_x = tk.Scrollbar(my_w_child, orient="horizontal", command=canvas.xview)
    scroll_x.grid(row=2, column=0, sticky="ew")


def change_column_3_x():
    global start_column, choppedArrays
    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("780x500")
    my_w_child.title("Змінити початковий стовпець")

    ent_data = StringVar(my_w_child, value=0)

    main_frame = tk.Frame(my_w_child)
    main_frame.grid(row=0, column=0)

    Entry(main_frame, textvariable=ent_data).pack()

    def on_confirm():
        global start_column
        try:
            pos = int(ent_data.get())
            ARRAYS[pos];
            start_column = pos
            my_w_child.destroy()
        except:
            messagebox.showerror("Помика", "Помилка")
        main()

    Button(main_frame, text='Підтвердити', command=on_confirm).pack()

    listvars = []
    canvas = tk.Canvas(my_w_child)
    child_frame = tk.Frame(canvas)
    canvas.create_window(0, 0, window=child_frame, anchor="nw")
    def update_size(e=None):
        canvas["scrollregion"] = canvas.bbox("all")
    canvas.bind('<Configure>', update_size)
    canvas.after_idle(update_size)
    canvas.grid(row=1, column=0)
    for i in range(len(ARRAYS)):
        listvars.append(Variable(my_w_child, value=ARRAYS[i]))
        Label(child_frame, text=str(i)).grid(row=0, column=i)
        Listbox(child_frame, listvariable=listvars[i]).grid(row=1, column=i)
    scroll_x = tk.Scrollbar(my_w_child, orient="horizontal", command=canvas.xview)
    scroll_x.grid(row=2, column=0, sticky="ew")



def change_column_3_y():
    global end_column
    my_w_child = tk.Toplevel(root)
    my_w_child.geometry("780x500")
    my_w_child.title("Змінити кінцевий стовпець")

    ent_data = StringVar(my_w_child, value=0)

    main_frame = tk.Frame(my_w_child)
    main_frame.grid(row=0, column=0)

    Entry(main_frame, textvariable=ent_data).pack()

    def on_confirm():
        global end_column
        try:
            pos = int(ent_data.get())
            ARRAYS[pos];
            end_column = pos
            my_w_child.destroy()
        except:
            messagebox.showerror("Помика", "Помилка")
        main()

    Button(main_frame, text='Підтвердити', command=on_confirm).pack()

    listvars = []
    canvas = tk.Canvas(my_w_child)
    child_frame = tk.Frame(canvas)
    canvas.create_window(0, 0, window=child_frame, anchor="nw")
    def update_size(e=None):
        canvas["scrollregion"] = canvas.bbox("all")
    canvas.bind('<Configure>', update_size)
    canvas.after_idle(update_size)
    canvas.grid(row=1, column=0)
    for i in range(len(ARRAYS)):
        listvars.append(Variable(my_w_child, value=ARRAYS[i]))
        Label(child_frame, text=str(i)).grid(row=0, column=i)
        Listbox(child_frame, listvariable=listvars[i]).grid(row=1, column=i)
    scroll_x = tk.Scrollbar(my_w_child, orient="horizontal", command=canvas.xview)
    scroll_x.grid(row=2, column=0, sticky="ew")



tab1 = Frame(tabControl, width=100, height=100)
tab2 = Frame(tabControl)
tab3 = Frame(tabControl)
tab4 = Frame(tabControl)

tabControl_for_list = Notebook(tab1)
tab1_list = Frame(tabControl_for_list)
tab2_list = Frame(tabControl_for_list)
tab3_list = Frame(tabControl_for_list)
tab4_list = Frame(tabControl_for_list)
tabControl_for_list.add(tab1_list, text='Коефіцієнти')
tabControl_for_list.add(tab2_list, text='Значення розподілів')
tabControl_for_list.add(tab3_list, text='Т-тести')
tabControl_for_list.add(tab4_list, text='Однорідність')

tabControl_for_lab_5 = Notebook(tab3)
tab1_lab_5 = Frame(tabControl_for_lab_5)
tab2_lab_5 = Frame(tabControl_for_lab_5)
tab3_lab_5 = Frame(tabControl_for_lab_5)
tabControl_for_lab_5.add(tab1_lab_5, text='Перв. стат. аналіз')
tabControl_for_lab_5.add(tab2_lab_5, text='Перевірка стах. зв.')
tabControl_for_lab_5.add(tab3_lab_5, text='Регресія')

tabControl.add(tab1, text='Вкладка 1')
tabControl.add(tab2, text='Вкладка 2')
tabControl.add(tab3, text='Вкладка 3')
tabControl.add(tab4, text='Вкладка 4')
tabControl.grid(row=0, column=0)
for c in range(3):
    tab1.columnconfigure(index=c, weight=1)
for r in range(4):
    tab1.rowconfigure(index=r, weight=1)

def round_message_box_2(lst):
    new_lst = []
    for x in lst:
        new_lst.append(round(x, 4))
    return new_lst

list1 = tk.Variable(value=lines)
list2 = tk.Variable(value=listBoxCoef)
custom_M = tk.StringVar()

probabilistic_grid_window = Button(
    tab1, text="Імовірнісна сітка", command=my_w_child_2_open)

variant_class = True
variant_graph = True
variant_param = False
variant_confidence_intervals = False
c_v1 = tk.BooleanVar(value=variant_class)
c_v2 = tk.BooleanVar(value=variant_graph)
c_v3 = tk.BooleanVar(value=variant_param)
c_v4 = tk.BooleanVar(value=variant_confidence_intervals)

hist_variant_hist = True
hist_variant_density = False
hist_variant_param = False
cc_v1 = tk.BooleanVar(value=hist_variant_hist)
cc_v2 = tk.BooleanVar(value=hist_variant_density)
cc_v3 = tk.BooleanVar(value=hist_variant_param)

log_val = False
anomal_val = False
shift_val = False
standardization_val = False
log_val_var = tk.BooleanVar(value=log_val)
anomal_val_var = tk.BooleanVar(value=anomal_val)
shift_val_var = tk.BooleanVar(value=shift_val)
standardization_val_var = tk.BooleanVar(value=standardization_val)

distribution_options = [
    'Не вибрано',
    'Не вибрано',
    'Рівномірний',
    'Нормальний',
    'Експоненціальний',
    'Логнормальний',
    'Вейбула',
]

distribution_options_var = tk.StringVar()
distribution_options_var.set('Не вибрано')


def on_distribution_options_changed(value):
    global parametric_values, parametric_values_hist, confidence_intervals_1, confidence_intervals_2
    global x_average, x_sq_avg, coun_kurtosis, asym_coef, kurtosis, pirson, x_2_average
    global ni_original, list_class_borders
    global param_expon_lambda, param_lognorm_m, param_lognorm_sigma, param_normal_m, param_normal_sigma, param_veib_alfa, param_veib_beta, param_rivnom_a, param_rivnom_b

    N = len(lines)
    probabilistic_grid_window["state"] = tk.DISABLED
    if value == 'Рівномірний':
        pi_elemets_for_each_lines = list(map(lambda x: x / len(lines), range(1, len(lines) + 1)))

        param_rospod.delete(0, tk.END)

        param_rivnom_a = x_average - math.sqrt(3*(x_2_average-x_average**2))
        param_rivnom_b = x_average + math.sqrt(3*(x_2_average-x_average**2))
        charact_rivnom_E_e = (param_rivnom_a+param_rivnom_b)/2
        charact_rivnom_D_e = (param_rivnom_b-param_rivnom_a)**2/12
        charact_rivnom_A = 0
        charact_rivnom_E = -1.2

        dH1_x = 1 + 3*(param_rivnom_a+param_rivnom_b)/(param_rivnom_b-param_rivnom_a)
        dH1_x_2 = -3/(param_rivnom_b-param_rivnom_a)
        dH2_x = 1 - 3*(param_rivnom_a+param_rivnom_b)/(param_rivnom_b-param_rivnom_a)
        dH2_x_2 = 3/(param_rivnom_b-param_rivnom_a)
        D_x = ((param_rivnom_b-param_rivnom_a)**2)/(12*N)
        cov_x_x_2 = ((param_rivnom_a+param_rivnom_b)*(param_rivnom_b-param_rivnom_a)**2)/(12*N)
        D_x_2 = ((param_rivnom_b-param_rivnom_a)**4+15*(param_rivnom_a+param_rivnom_b)**2*(param_rivnom_b-param_rivnom_a)**2)/(180*N)
        tochn_rivnom_a = (dH1_x**2)*D_x + (dH1_x_2**2)*D_x_2 + 2*dH1_x*dH1_x_2*cov_x_x_2
        tochn_rivnom_b = (dH2_x**2)*D_x + (dH2_x_2**2)*D_x_2 + 2*dH2_x*dH2_x_2*cov_x_x_2
        tochn_rivnom_cov_a_b = dH1_x*dH2_x*D_x + dH1_x_2*dH2_x_2*D_x_2 + (dH1_x*dH2_x_2+dH1_x_2*dH2_x)*cov_x_x_2

        ni_1_F_rivnom = []
        for i in range(1, len(list_class_borders)):
            ni_1_F_rivnom.append((F_rivnom(list_class_borders[i], param_rivnom_a, param_rivnom_b) - F_rivnom(list_class_borders[i-1], param_rivnom_a, param_rivnom_b))*N)
        hi_sq_rivnom = count_hi_sq_pirson(ni_original, ni_1_F_rivnom, N)

        probabilistic_grid_window["state"] = tk.NORMAL
        def F_dov(x, a , b):
            eq_1 = (x-b)**2/(b-a)**4
            eq_2 = (x-b)**2/(b-a)**4
            eq_3 = ((x-a)*(x-b))/(b-a)**4
            return eq_1*tochn_rivnom_a + eq_2*tochn_rivnom_b - 2*eq_3*tochn_rivnom_cov_a_b
        def F(x, a, b):
            if x < a:
                return 0
            elif x >= b:
                return 1
            else:
                return (x-a)/(b-a)
        def F_hist(x, a, b):
            if x < a:
                return 0
            elif x >= b:
                return 1
            else:
                return 1/(b-a)
        parametric_values = list(map(lambda x: F(x, param_rivnom_a, param_rivnom_b), lines))
        parametric_values_hist = list(map(lambda x: F_hist(x, param_rivnom_a, param_rivnom_b), lines))

        confidence_intervals_1 = list(map(lambda x: F(x, param_rivnom_a, param_rivnom_b) - 1.965*np.sqrt(F_dov(x, param_rivnom_a, param_rivnom_b)), lines))
        confidence_intervals_2 = list(map(lambda x: F(x, param_rivnom_a, param_rivnom_b) + 1.965*np.sqrt(F_dov(x, param_rivnom_a, param_rivnom_b)), lines))


        differences = [pi_elemets_for_each_lines[i] - parametric_values[i] for i in range(len(pi_elemets_for_each_lines))]
        rivnom_D_plus = max(abs(difference) for difference in differences)
        differences = [pi_elemets_for_each_lines[i] - parametric_values[i] for i in range(1, len(pi_elemets_for_each_lines)-1)]
        rivnom_D_minus = max(abs(difference) for difference in differences)
        D_list = []
        D_list.append(rivnom_D_plus)
        D_list.append(rivnom_D_minus)
        rivnom_z = np.sqrt(N)*max(D_list)

        P_z = calc_P_z(rivnom_z, N)
        
        param_rospod.insert(tk.END, "Математичне сподівання: " + str(round_message_box(charact_rivnom_E_e)))
        param_rospod.insert(tk.END, "Дисперсія: " + str(round_message_box(charact_rivnom_D_e)))
        param_rospod.insert(tk.END, "Коефіцієнт асиметрії: " + str(round_message_box(charact_rivnom_A)))
        param_rospod.insert(tk.END, "Коефіцієнт ексцесу: " + str(round_message_box(charact_rivnom_E)))
        param_rospod.insert(tk.END, "Точність оцінки a: " + str(round_message_box(tochn_rivnom_a)))
        param_rospod.insert(tk.END, "Точність оцінки b: " + str(round_message_box(tochn_rivnom_b)))
        param_rospod.insert(tk.END, "Коваріація a та b: " + str(round_message_box(tochn_rivnom_cov_a_b)))

        param_rospod.insert(tk.END, "Критерій Колмагорова: " + str(round_message_box(P_z)))
        param_rospod.insert(tk.END, "Критерій хі-квадрат " + str(round_message_box(hi_sq_rivnom)))

        

    elif value == 'Нормальний':
        pi_elemets_for_each_lines = list(map(lambda x: x / len(lines), range(1, len(lines) + 1)))

        param_rospod.delete(0, tk.END)

        param_normal_m = x_average
        param_normal_sigma = (N*sqrt(x_2_average - x_average**2))/(N-1)
        charact_normal_E_e = param_normal_m
        charact_normal_D_e = param_normal_sigma**2
        charact_normal_A = 0
        charact_normal_E = 0
        charact_normal_E1 = 3
        tochn_normal_m = param_normal_sigma**2/N
        tochn_normal_sigma = param_normal_sigma**2/(2*N)
        tochn_normal_cov_m_sigma = 0

        ni_1_F_normal = []
        for i in range(1, len(list_class_borders)):
            ni_1_F_normal.append((F_normal(list_class_borders[i], param_normal_m, param_normal_sigma) - F_normal(list_class_borders[i-1], param_normal_m, param_normal_sigma))*N)
        hi_sq_normal = count_hi_sq_pirson(ni_original, ni_1_F_normal, N)

        probabilistic_grid_window["state"] = tk.NORMAL

        ro = 0.2316419
        b_1 = 0.31938153
        b_2 = -0.356563782
        b_3 = 1.781477937
        b_4 = -1.821255978
        b_5 = 1.330274429
        e_u = 7.8*(10)**(-8)
        def F_dov(x, m, sig):
            df_dm = (-1/(sig*np.sqrt(2*np.pi))) * np.exp(-(x-m)**2/(2*sig**2))
            df_dsig = (-(x-m)/(sig**2*np.sqrt(2*np.pi))) * np.exp(-(x-m)**2/(2*sig**2))
            return  df_dm**2*tochn_normal_m + df_dsig**2*tochn_normal_sigma + 2*df_dm*df_dsig*tochn_normal_cov_m_sigma 
        def F(x, m, sig):
            u = (x-m)/sig
            if u < 0:
                t = 1/(1+ro*abs(u))
                return 1 - (1 - 1/np.sqrt(2*np.pi) * np.exp(-u**2/2) * (b_1*t + b_2*t**2 + b_3*t**3 + b_4*t**4 + b_5*t**5)  + e_u)
            else:
                t = 1/(1+ro*u)
                return 1 - 1/np.sqrt(2*np.pi) * np.exp(-u**2/2) * (b_1*t + b_2*t**2 + b_3*t**3 + b_4*t**4 + b_5*t**5)  + e_u
        def F_hist(x, m, sig):
            return (1/(sig*np.sqrt(2*np.pi)))*np.exp(-(x-m)**2/(2*sig**2)) 
        parametric_values_hist = list(map(lambda x: F_hist(x, param_normal_m, param_normal_sigma), lines))
        parametric_values = list(map(lambda x: F(x, param_normal_m, param_normal_sigma), lines))

        confidence_intervals_1 = list(map(lambda x: F(x, param_normal_m, param_normal_sigma) - 1.965*np.sqrt(F_dov(x, param_normal_m, param_normal_sigma)), lines))
        confidence_intervals_2 = list(map(lambda x: F(x, param_normal_m, param_normal_sigma) + 1.965*np.sqrt(F_dov(x, param_normal_m, param_normal_sigma)), lines))

        differences = [pi_elemets_for_each_lines[i] - parametric_values[i] for i in range(len(pi_elemets_for_each_lines))]
        normal_D_plus = max(abs(difference) for difference in differences)
        differences = [pi_elemets_for_each_lines[i] - parametric_values[i-1] for i in range(1, len(pi_elemets_for_each_lines)-1)]
        normal_D_minus = max(abs(difference) for difference in differences)
        D_list = []
        D_list.append(normal_D_plus)
        D_list.append(normal_D_minus)
        normal_z = np.sqrt(N)*max(D_list)
        P_z = calc_P_z(normal_z, N)

        param_rospod.insert(tk.END, "Математичне сподівання: " + str(round_message_box(charact_normal_E_e)))
        param_rospod.insert(tk.END, "Дисперсія: " + str(round_message_box(charact_normal_D_e)))
        param_rospod.insert(tk.END, "Коефіцієнт асиметрії: " + str(round_message_box(charact_normal_A)))
        param_rospod.insert(tk.END, "Коефіцієнт ексцесу: " + str(round_message_box(charact_normal_E)) + ", E(зсунений): " + str(round_message_box(charact_normal_E1)))
        param_rospod.insert(tk.END, "Точність оцінки m: " + str(round_message_box(tochn_normal_m)))
        param_rospod.insert(tk.END, "Точність оцінки sigma: " + str(round_message_box(tochn_normal_sigma)))
        param_rospod.insert(tk.END, "Коваріація m та sigma: " + str(round_message_box(tochn_normal_cov_m_sigma)))

        param_rospod.insert(tk.END, "Критерій Колмагорова: " + str(round_message_box(P_z)))
        param_rospod.insert(tk.END, "Критерій хі-квадрат " + str(round_message_box(hi_sq_normal)))

    elif value == 'Експоненціальний':
        pi_elemets_for_each_lines = list(map(lambda x: x / len(lines), range(1, len(lines) + 1)))

        param_rospod.delete(0, tk.END)

        param_expon_lambda = 1/x_average 
        charact_expon_E_e = 1/param_expon_lambda
        charact_expon_D_e = 1/param_expon_lambda**2
        charact_expon_A = 2
        charact_expon_E = 6
        tochn_expon_lamba = param_expon_lambda**2/N

        ni_1_F_expon = []
        for i in range(1, len(list_class_borders)):
            ni_1_F_expon.append((F_expon(list_class_borders[i], param_expon_lambda) - F_expon(list_class_borders[i-1], param_expon_lambda))*N)
        hi_sq_expon = count_hi_sq_pirson(ni_original, ni_1_F_expon, N)

        probabilistic_grid_window["state"] = tk.NORMAL
        def F(x, lam):
            if x < 0:
                return 0
            else:
                return 1-math.exp(-lam*x)
        def F_hist(x, lam):
            if x < 0:
                return 0
            else:
                return lam * math.exp(-lam*x)
        parametric_values = list(map(lambda x: F(x, param_expon_lambda), lines))
        parametric_values_hist = list(map(lambda x: F_hist(x, param_expon_lambda), lines))

        confidence_intervals_1 = list(map(lambda x: 1-math.exp(-param_expon_lambda*x)+1.956*math.sqrt(x**2*math.exp(-2*param_expon_lambda*x)*param_expon_lambda**2/N), lines))
        confidence_intervals_2 = list(map(lambda x: 1-math.exp(-param_expon_lambda*x)-1.956*math.sqrt(x**2*math.exp(-2*param_expon_lambda*x)*param_expon_lambda**2/N), lines))
        differences = [pi_elemets_for_each_lines[i] - parametric_values[i] for i in range(len(pi_elemets_for_each_lines))]
        expon_D_plus = max(abs(difference) for difference in differences)
        differences = [pi_elemets_for_each_lines[i] - parametric_values[i-1] for i in range(1, len(pi_elemets_for_each_lines) - 1)]
        expon_D_minus = max(abs(difference) for difference in differences)
        D_list = []
        D_list.append(expon_D_plus)
        D_list.append(expon_D_minus)
        expon_z = np.sqrt(N)*max(D_list)
        P_z = calc_P_z(expon_z, N)

        param_rospod.insert(tk.END, "Математичне сподівання: " + str(round_message_box(charact_expon_E_e)))
        param_rospod.insert(tk.END, "Дисперсія: " + str(round_message_box(charact_expon_D_e)))
        param_rospod.insert(tk.END, "Коефіцієнт асиметрії: " + str(round_message_box(charact_expon_A)))
        param_rospod.insert(tk.END, "Коефіцієнт ексцесу: " + str(round_message_box(charact_expon_E)))
        param_rospod.insert(tk.END, "Точність оцінки lambda: " + str(round_message_box(param_expon_lambda)))
        param_rospod.insert(tk.END, "Точність оцінки lambda: " + str(round_message_box(tochn_expon_lamba)))

        param_rospod.insert(tk.END, "Критерій Колмагорова: " + str(round_message_box(P_z)))
        param_rospod.insert(tk.END, "Критерій хі-квадрат " + str(round_message_box(hi_sq_expon)))

    elif value == 'Логнормальний':
        pi_elemets_for_each_lines = list(map(lambda x: x / len(lines), range(1, len(lines) + 1)))

        param_rospod.delete(0, tk.END)

        param_lognorm_m = 2*math.log(x_average)-0.5*math.log(x_2_average)
        param_lognorm_sigma = math.sqrt(math.log(x_2_average)-2*math.log(x_average))
        charact_lognorm_E_e = math.exp(param_lognorm_m+param_lognorm_sigma**2/2)
        charact_lognorm_D_e = math.exp(2*param_lognorm_m+param_lognorm_sigma**2)*(math.exp(param_lognorm_sigma**2)-1)
        charact_lognorm_A = (2-3*math.exp(param_lognorm_sigma**2)+math.exp(3*param_lognorm_sigma**2))/(math.exp(param_lognorm_sigma**2)-1)**(3/2)
        charact_lognorm_E = (math.exp(6*param_lognorm_sigma**2)-4*math.exp(3*param_lognorm_sigma**2)-3*math.exp(2*param_lognorm_sigma**2)+12*math.exp(param_lognorm_sigma**2)-6)/((math.exp(param_lognorm_sigma**2)-1)**2)
        tochn_lognorm_m = (math.exp(4*param_lognorm_sigma**2)-8*math.exp(2*param_lognorm_sigma**2)+16*math.exp(param_lognorm_sigma**2)-9)/(4*N)
        tochn_lognorm_sigma_2 = (math.exp(4*param_lognorm_sigma**2)-4*math.exp(2*param_lognorm_sigma**2)+4*math.exp(param_lognorm_sigma**2)-1)/(4*N*(param_lognorm_sigma)**2)
        tochn_lognorm_cov = (-math.exp(4*param_lognorm_sigma**2)+6*math.exp(2*param_lognorm_sigma**2)-8*math.exp(param_lognorm_sigma**2)+3)/(4*N*(param_lognorm_sigma)**2)

        ni_1_F_lognorm = []
        for i in range(1, len(list_class_borders)):
            ni_1_F_lognorm.append((F_lognorm(list_class_borders[i], param_lognorm_sigma, param_lognorm_m) - F_lognorm(list_class_borders[i-1], param_lognorm_sigma, param_lognorm_m))*N)

        hi_sq_lognorm = count_hi_sq_pirson(ni_original, ni_1_F_lognorm, N)

        probabilistic_grid_window["state"] = tk.NORMAL

        def F_help(x, m, sig):
            ro = 0.2316419
            b_1 = 0.31938153
            b_2 = -0.356563782
            b_3 = 1.781477937
            b_4 = -1.821255978
            b_5 = 1.330274429
            e_u = 7.8*(10)**(-8)
            u = (x-m)/sig
            if u < 0:
                t = 1/(1+ro*abs(u))
                return 1 - (1 - 1/np.sqrt(2*np.pi) * np.exp(-u**2/2) * (b_1*t + b_2*t**2 + b_3*t**3 + b_4*t**4 + b_5*t**5)  + e_u)
            else:
                t = 1/(1+ro*u)
                return 1 - 1/np.sqrt(2*np.pi) * np.exp(-u**2/2) * (b_1*t + b_2*t**2 + b_3*t**3 + b_4*t**4 + b_5*t**5)  + e_u
        def F_dov(x, sig, m):
            df_dm = 1/(sig*np.sqrt(2*np.pi)) * (1 - np.exp(-(np.log(x)-m)**2/(2*sig**2)))
            df_dsig = F_help(x, m, sig)*(1-1/sig) - 1/(sig*np.sqrt(2*np.pi))*((np.log(x)-m)/sig)*np.exp(-(np.log(x)-m)**2/(2*sig**2))
            return df_dm**2*tochn_lognorm_m + df_dsig**2*tochn_lognorm_sigma_2 +2*df_dm*df_dsig*tochn_lognorm_cov
        def F(x, sig, m):
            if x <= 0:
                return 0
            else:
                return 0.5 + 0.5*scipy.special.erf((np.log(x)-m)/(sig*np.sqrt(2)))
        def F_hist(x, sig, m):
            if x <= 0:
                return 0
            else:
                return (1/(np.sqrt(2*np.pi)*sig*x)) * np.exp(-(np.log(x)-m)**2/(2*sig**2))
        parametric_values_hist = list(map(lambda x: F_hist(x, param_lognorm_sigma, param_lognorm_m), lines))
        parametric_values = list(map(lambda x: F(x, param_lognorm_sigma, param_lognorm_m), lines))

        confidence_intervals_1 = list(map(lambda x: F(x, param_lognorm_sigma, param_lognorm_m) - 1.965*np.sqrt(F_dov(x, param_lognorm_sigma, param_lognorm_m)), lines))
        confidence_intervals_2 = list(map(lambda x: F(x, param_lognorm_sigma, param_lognorm_m) + 1.965*np.sqrt(F_dov(x, param_lognorm_sigma, param_lognorm_m)), lines))

        differences = [pi_elemets_for_each_lines[i] - parametric_values[i] for i in range(len(pi_elemets_for_each_lines))]
        lognorm_D_plus = max(abs(difference) for difference in differences)
        differences = [pi_elemets_for_each_lines[i] - parametric_values[i-1] for i in range(1, len(pi_elemets_for_each_lines)-1)]
        lognorm_D_minus = max(abs(difference) for difference in differences)
        D_list = []
        D_list.append(lognorm_D_plus)
        D_list.append(lognorm_D_minus)
        lognorm_z = np.sqrt(N)*max(D_list)
        P_z = calc_P_z(lognorm_z, N)

        param_rospod.insert(tk.END, "Математичне сподівання: " + str(round_message_box(charact_lognorm_E_e)))
        param_rospod.insert(tk.END, "Дисперсія: " + str(round_message_box(charact_lognorm_D_e)))
        param_rospod.insert(tk.END, "Коефіцієнт асиметрії: " + str(round_message_box(charact_lognorm_A)))
        param_rospod.insert(tk.END, "Коефіцієнт ексцесу: " + str(round_message_box(charact_lognorm_E)))
        param_rospod.insert(tk.END, "Точність оцінки m: " + str(round_message_box(tochn_lognorm_m)))
        param_rospod.insert(tk.END, "Точність оцінки sigam: " + str(round_message_box(tochn_lognorm_sigma_2)))
        param_rospod.insert(tk.END, "Коваріація m та sigma: " + str(round_message_box(tochn_lognorm_cov)))

        param_rospod.insert(tk.END, "Критерій Колмагорова: " + str(round_message_box(P_z)))
        param_rospod.insert(tk.END, "Критерій хі-квадрат " + str(round_message_box(hi_sq_lognorm)))

    elif value == 'Вейбула':
        pi_elemets_for_each_lines = list(map(lambda x: x / len(lines), range(1, len(lines) + 1)))

        param_rospod.delete(0, tk.END)

        a11 = N - 1
        a12 = a21 = sum(list(map(math.log, lines[:-1])))
        a22 = sum(list(map(lambda x: math.log(x) ** 2, lines[:-1])))
        b1 = sum(list(map(lambda fx: math.log(math.log(1/(1-fx))), pi_elemets_for_each_lines[:-1])))
        b2 = sum(list(map(lambda val: math.log(val[0])*math.log(math.log(1/(1-val[1]))), list(zip(lines[:-1], pi_elemets_for_each_lines[:-1])))))
        
        param_veib_A = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
        param_veib_beta = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)
        param_veib_alfa = np.exp(-param_veib_A)
        charact_veib_E_e = param_veib_alfa**(2/param_veib_beta)*math.gamma(1+1/param_veib_beta)
        charact_veib_D_e = param_veib_alfa**(2/param_veib_beta)*(math.gamma(1+2/param_veib_beta)-(math.gamma(1+1/param_veib_beta))**2)
        charact_veib_A = Mk(N, lines, Vk(N, lines, 1), 3)/(Mk(N, lines, Vk(N, lines, 1), 2))**(3/2)
        charact_veib_E = Mk(N, lines, Vk(N, lines, 1), 4)/(Mk(N, lines, Vk(N, lines, 1), 2))**2-3
        S_zal_2 = sum_S_zal_2(N, lines, pi_elemets_for_each_lines, param_veib_A, param_veib_beta)
        tochn_veib_A = (a22*S_zal_2)/(a11*a22-a12*a21)
        tochn_veib_beta = (a11*S_zal_2)/(a11*a22-a12*a21)
        tochn_veib_cov_A_beta = -(a21*S_zal_2)/(a11*a22-a12*a21)
        tochn_veib_alfa = np.exp(-2*param_veib_A)*tochn_veib_A
        tochn_veib_cov_alfa_beta = -np.exp(param_veib_A)*tochn_veib_cov_A_beta

        ni_1_F_veib = []
        for i in range(1, len(list_class_borders)):
            ni_1_F_veib.append((F_veib(list_class_borders[i], param_veib_alfa, param_veib_beta) - F_veib(list_class_borders[i-1], param_veib_alfa, param_veib_beta))*N)
        hi_sq_veib = count_hi_sq_pirson(ni_original, ni_1_F_veib, N)

        def F_dov(x, alf, bet):
            df_da = -x**bet/alf**2*np.exp(-x**bet/alf)
            df_db = x**bet/alf*np.exp(-x**bet/alf)*np.log(x)
            return df_da**2*tochn_veib_alfa + df_db**2*tochn_veib_beta + 2*df_da*df_db*tochn_veib_cov_alfa_beta
        def F(x, alf, bet):
            return 1 - np.exp(-x**bet/alf)
        def F_hist(x, alf, bet):
            return (bet/alf)*(x**(bet-1))*np.exp(-x**bet/alf)
        
        parametric_values = list(map(lambda x: F(x, param_veib_alfa, param_veib_beta), lines))

        differences = [pi_elemets_for_each_lines[i] - parametric_values[i] for i in range(len(pi_elemets_for_each_lines))]
        veib_D_plus = max(abs(difference) for difference in differences)
        differences = [pi_elemets_for_each_lines[i] - parametric_values[i-1] for i in range(1,len(pi_elemets_for_each_lines)-1)]
        veib_D_minus = max(abs(difference) for difference in differences)
        D_list = []
        D_list.append(veib_D_plus)
        D_list.append(veib_D_minus)
        veib_z = np.sqrt(N)*max(D_list)
        P_z = calc_P_z(veib_z, N)

        param_rospod.insert(tk.END, "Математичне сподівання: " + str(round_message_box(charact_veib_E_e)))
        param_rospod.insert(tk.END, "Дисперсія: " + str(round_message_box(charact_veib_D_e)))
        param_rospod.insert(tk.END, "Коефіцієнт асиметрії: " + str(round_message_box(charact_veib_A)))
        param_rospod.insert(tk.END, "Коефіцієнт ексцесу: " + str(round_message_box(charact_veib_E)))
        param_rospod.insert(tk.END, "Точність оцінки A: " + str(round_message_box(tochn_veib_A)))
        param_rospod.insert(tk.END, "Точність оцінки alfa: " + str(round_message_box(tochn_veib_alfa)))
        param_rospod.insert(tk.END, "Точність оцінки beta: " + str(round_message_box(tochn_veib_beta)))
        param_rospod.insert(tk.END, "Коваріація A та beta: " + str(round_message_box(tochn_veib_cov_A_beta)))
        param_rospod.insert(tk.END, "Коваріація alfa та beta: " + str(round_message_box(tochn_veib_cov_alfa_beta)))

        param_rospod.insert(tk.END, "Критерій Колмагорова: " + str(round_message_box(P_z)))
        param_rospod.insert(tk.END, "Критерій хі-квадрат " + str(round_message_box(hi_sq_veib)))
        
        parametric_values = list(map(lambda x: F(x, param_veib_alfa, param_veib_beta), lines))
        parametric_values_hist = list(map(lambda x: F_hist(x, param_veib_alfa, param_veib_beta), lines))
        confidence_intervals_1 = list(map(lambda x: F(x, param_veib_alfa, param_veib_beta) - 1.965*np.sqrt(F_dov(x, param_veib_alfa, param_veib_beta)), lines))
        confidence_intervals_2 = list(map(lambda x: F(x, param_veib_alfa, param_veib_beta) + 1.965*np.sqrt(F_dov(x, param_veib_alfa, param_veib_beta)), lines))

        click_graph_emp()

def calc_P_z(Z, n):
    K_Z = 0
    for K in range(1, n+1):
        f_1 = K**2 - 0.5*(1-((-1)**K))
        f_2 = 5*K**2 + 22 - 7.5*(1-((-1)**K))
        sum_to_N = (-1) ** K * np.exp(-2*K**2 * Z**2) * ((1 - ((2*K**2*Z)/(3*n**(1/2))) - ((1/(18*n))*((f_1-4*(f_1+3)) * K**2 * Z**2 + 8*K**4 * Z**4))) + (((K**2*Z)/(27 * (n**(3/2)))) * ((f_2**2/5) - ((4*(f_2+45) * K**2 * Z**2)/15) + 8*K**4 * Z**4)))
        K_Z = K_Z + sum_to_N
    K_Z = K_Z * 2
    K_Z = K_Z + 1
    return 1 - K_Z

def sum_S_zal_2(n, lines, pi_elements_sum, A, beta):
    total_sum = 0
    for i in range(n-1):
        total_sum += (math.log(math.log(1/(1-pi_elements_sum[i]))) - A - beta*math.log(lines[i]))**2
    return total_sum*(1/(n-3))

def findMid(lines):
    b = lines[0]+krok
    mid_num = (b+lines[0])/2
    return mid_num


def getParity(n):
    if type(n) == int:
        return (bin(n).count("1")) % 2
    else:
        return 1

def sum(arr):
    sum = 0
    for i in arr:
        sum = sum + i
    return(sum)

def find_MED(number_list):
    N = len(number_list)
    if N % 2 == 0:
        k = int(N/2)
        return (number_list[k]+number_list[k+1])/2
    else:
        k = int((N-1)/2)
        return number_list[k]

def find_MAD(number_list, MED):
    sub_mad_list = []
    for i in range(len(number_list)):
        sub_mad_list.append(abs(number_list[i]-MED))
    mad = 1.483 * find_MED(sub_mad_list)
    return mad

def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


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


def plot_sectors_and_points():
    global calculation_data,rel_freq_calc,rel_freq_calc_2
    data_y = calculation_data.array[1]
    data_x = calculation_data.array[0]
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

    fig, ax = plt.subplots(figsize=(5, 4))

    plt.imshow(rel_freq, extent=[bins_x.min(), bins_x.max(), bins_y.min(), bins_y.max()], aspect='auto', origin='lower', cmap='viridis')
    
    plt.title('Гістограма')
    plt.xlabel('X-координата')
    plt.ylabel('Y-координата')

    canvas = FigureCanvasTkAgg(fig, master=tab2)
    canvas.draw()
    canvas.get_tk_widget().place(x=10,y=30)

    toolbar = NavigationToolbar2Tk(canvas, tab2, pack_toolbar=False)
    toolbar.update()
    toolbar.place(x=200,y=440)

def plot_2d_norm_graph():
    global z_list_for_graph
    data_y = calculation_data.array[1]
    data_x = calculation_data.array[0]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data_x, data_y, z_list_for_graph, c=z_list_for_graph, cmap='viridis', marker='o', alpha=0.5)
    ax.set_xlabel('X-координата')
    ax.set_ylabel('Y-координата')
    ax.set_zlabel('Щільність ймовірності')
    plt.title('Двовимірний нормальний розподіл')

    canvas = FigureCanvasTkAgg(fig, master=tab2)
    canvas.draw()
    canvas.get_tk_widget().place(x=1000,y=30)

    toolbar = NavigationToolbar2Tk(canvas, tab2, pack_toolbar=False)
    toolbar.update()
    toolbar.place(x=1100,y=440)

def plot_3d_x_y_z():
    global choppedARRAYS
    data_x = choppedARRAYS[0]
    data_y = choppedARRAYS[1]
    data_z = choppedARRAYS[2]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data_x, data_y, data_z)

    canvas = FigureCanvasTkAgg(fig, master=tab4)
    canvas.draw()
    canvas.get_tk_widget().place(x=1100,y=380)


def plot_points():
    global calculation_data, cuass_val, linear_val, hyperbolic, corr_coef,rell_corr
    data_y = calculation_data.array[1]
    data_x = calculation_data.array[0]
    n = len(data_x)
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(data_x, data_y, color='blue', alpha=0.5, s= 10)
    plt.title('Кореляційне поле')
    plt.xlabel('X-координата')
    plt.ylabel('Y-координата')

    if cuass_val:
        ax.scatter(10, 10, color='red', alpha=0.5)
    if linear_val:
        list_box_for_regression_char.delete(0,tk.END)
        a, b, sigma_x, sigma_y = CorrelationalRegression.find_a_b_for_lin_reg(ForMyCorrelationalString, corr_coef)

        x = np.linspace(min(data_x), max(data_x), 100)
        y=a+b*x

        a_teyla, b_teyla = CorrelationalRegression.find_a_b_for_lin_reg_teyla(ForMyCorrelationalString)

        y_teyla=a_teyla+b_teyla*x

        list_box_for_regression_char.insert(tk.END, "Чорний графік - метод Тейла "+str(round(a_teyla,3)))
        list_box_for_regression_char.insert(tk.END, "Червоний графік - МНК = "+str(round(b_teyla,3)))
        list_box_for_regression_char.insert(tk.END, "\n")
        

        list_box_for_regression_char.insert(tk.END, "Оцінка параметра a = "+str(round(a,3)))
        list_box_for_regression_char.insert(tk.END, "Оцінка параметра b = "+str(round(b,3)))
        list_box_for_regression_char.insert(tk.END, "\n")

        x_av_loc = sum(data_x)/n
        S_zal_sq = 1/(n-2)*sum((data_y[i]-a-b*data_x[i])**2 for i in range(n))

        S_b = math.sqrt(S_zal_sq)/(sigma_x*math.sqrt(n-1))

        list_box_for_regression_char.insert(tk.END, "Значущість лінійної регресії: (t)")
        t_b = b/S_b
        if t_b > t.ppf(1 - 0.05/2, n-2):
            list_box_for_regression_char.insert(tk.END, str(round(t_b,3)) +">"+ str(round(t.ppf(1 - 0.05/2, n-2),3)) + "(Значуща)")
        else:
            list_box_for_regression_char.insert(tk.END, str(round(t_b,3)) +"<="+ str(round(t.ppf(1 - 0.05/2, n-2),3))+ "(Не значуща)")

        coeff_determ_linear = corr_coef**2*100
        list_box_for_regression_char.insert(tk.END, "\n")
        list_box_for_regression_char.insert(tk.END, "Коефіцієнт детермінації = "+str(round(coeff_determ_linear,3))+"%")

        sigma_e = math.sqrt(S_zal_sq)
        f_fisher_loc = sigma_e**2/sigma_y**2
        list_box_for_regression_char.insert(tk.END, "\n")
        list_box_for_regression_char.insert(tk.END, "Перевірка на адекватність відтворення")
        if f_fisher_loc <= f.ppf(1 - 0.005, n-1, n-3):
            list_box_for_regression_char.insert(tk.END, str(round(f_fisher_loc,3)) +"<="+ str(round(f.ppf(1 - 0.005, n-1, n-3),3))+ "(Адекватна)")
        else:
            list_box_for_regression_char.insert(tk.END, str(round(f_fisher_loc,3)) +">"+ str(round(f.ppf(1 - 0.005, n-1, n-3),3))+ "(Не адекватна)")

        y_min_tol = y - sigma_e * t.ppf(1 - 0.05/2, n-2) 
        y_max_tol = y + sigma_e * t.ppf(1 - 0.05/2, n-2)

        S_y_x0 = np.sqrt(sigma_e**2*(1+1/n)+S_b**2*(x-x_av_loc)**2)
        y_min_dov_new = y - S_y_x0 * t.ppf(1 - 0.05/2, n-2)
        y_max_dov_new = y + S_y_x0 * t.ppf(1 - 0.05/2, n-2)

        S_y_x = np.sqrt(sigma_e**2*(1/n)+S_b**2*(x-x_av_loc)**2)
        y_min_dov = y - S_y_x * t.ppf(1 - 0.05/2, n-2)
        y_max_dov = y + S_y_x * t.ppf(1 - 0.05/2, n-2)
        
        ax.plot(x, y_teyla, color='black', alpha=1,linewidth=3)

        ax.plot(x, y_min_dov, color='orange', alpha=1,linewidth=2)
        ax.plot(x, y_max_dov, color='orange', alpha=1,linewidth=2)

        ax.plot(x, y_min_dov_new, color='yellow', alpha=1,linewidth=4)
        ax.plot(x, y_max_dov_new, color='yellow', alpha=1,linewidth=4)

        ax.plot(x, y_min_tol, color='green', alpha=1,linewidth=2, linestyle = "--")
        ax.plot(x, y_max_tol, color='green', alpha=1,linewidth=2, linestyle = "--")

        ax.plot(x, y, color='red', linestyle = "--", alpha=1,linewidth=2)
    if hyperbolic:
        list_box_for_regression_char.delete(0,tk.END)
        a_1, b_1, c_1 = CorrelationalRegression.find_a_b_c_for_hyper(ForMyCorrelationalString)

        list_box_for_regression_char.insert(tk.END, "Оцінки параметрів: ")
        list_box_for_regression_char.insert(tk.END, "a = "+str(round(a_1,3)))
        list_box_for_regression_char.insert(tk.END, "b = "+str(round(b_1,3)))
        list_box_for_regression_char.insert(tk.END, "c = "+str(round(c_1,3)))
        list_box_for_regression_char.insert(tk.END, "\n")

        x = np.linspace(min(data_x), max(data_x), 100)

        np_x_list = np.array(data_x)
        np_y_list = np.array(data_y)

        y = a_1 + b_1 * fi_1(x) + c_1* fi_2(x)

        s_zal_2_2 = 1/(n-3) * sum((np_y_list-a_1-b_1*fi_1(np_x_list)-c_1*fi_2(np_x_list))**2)

        t_a_1 = abs((a_1/np.sqrt(s_zal_2_2)) * np.sqrt(n))

        t_b_1 = abs((b_1*np.sqrt(np.var(np_x_list)))/np.sqrt(s_zal_2_2) * np.sqrt(n))

        t_c_1 = abs(c_1/np.sqrt(s_zal_2_2) * np.sqrt(n * sum(fi_2(np_x_list)**2)))

        list_box_for_regression_char.insert(tk.END, "Перевірка значущості: ")

        if t_a_1 <= t.ppf(1 - 0.05/2, n-3) and t_b_1 <= t.ppf(1 - 0.05/2, n-3) and t_c_1 <= t.ppf(1 - 0.05/2, n-3):
            list_box_for_regression_char.insert(tk.END, "t_a_1 = " + str(round(t_a_1,3)) + "<=" + str(round(t.ppf(1 - 0.05/2, n-2),3)))
            list_box_for_regression_char.insert(tk.END, "t_b_1 = " + str(round(t_b_1,3)) + "<=" + str(round(t.ppf(1 - 0.05/2, n-2),3)))
            list_box_for_regression_char.insert(tk.END, "t_c_1 = " + str(round(t_c_1,3)) + "<=" + str(round(t.ppf(1 - 0.05/2, n-2),3)))
        else:
            list_box_for_regression_char.insert(tk.END, "Один з членів параболи або декілька було")
            list_box_for_regression_char.insert(tk.END, "втрачено")
            list_box_for_regression_char.insert(tk.END, "t_a_1 = " + str(round(t_a_1,3)))
            list_box_for_regression_char.insert(tk.END, "t_b_1 = " + str(round(t_b_1,3)))
            list_box_for_regression_char.insert(tk.END, "t_c_1 = " + str(round(t_c_1,3)))
            list_box_for_regression_char.insert(tk.END, "t критичне значення = " + str(round(t.ppf(1 - 0.05/2, n-2),3)))

        list_box_for_regression_char.insert(tk.END, "\n")

        y_min_tol_hyper = y - t.ppf(1 - 0.05/2, n-3) * np.sqrt(s_zal_2_2)
        y_max_tol_hyper = y + t.ppf(1 - 0.05/2, n-3) * np.sqrt(s_zal_2_2)

        s_y_x_hyper = np.sqrt(s_zal_2_2/n) * np.sqrt(1 + (fi_1(x)**2)/(np.var(x))+(fi_2(x)**2)/(sum(fi_2(x)**2)))

        s_y_x_new_hyper = np.sqrt(s_zal_2_2/n) * np.sqrt(n + 1 + (fi_1(x)**2)/(np.var(x))+(fi_2(x)**2)/(sum(fi_2(x)**2)))

        y_min_dov_hyper = y - t.ppf(1 - 0.05/2, n-3) * s_y_x_hyper
        y_max_dov_hyper = y + t.ppf(1 - 0.05/2, n-3) * s_y_x_hyper

        y_min_dov_new_hyper = y - t.ppf(1 - 0.05/2, n-3) * s_y_x_new_hyper
        y_max_dov_new_hyper = y + t.ppf(1 - 0.05/2, n-3) * s_y_x_new_hyper

        coeff_determ_hyper = rell_corr * 100

        list_box_for_regression_char.insert(tk.END, "Коефіцієнт детермінації = " + str(round(coeff_determ_hyper,3))+ "%")

        sigma_e = s_zal_2_2
        f_fisher_loc = sigma_e/np.var(np_y_list)
        list_box_for_regression_char.insert(tk.END, "\n")
        list_box_for_regression_char.insert(tk.END, "Перевірка на адекватність відтворення")
        if f_fisher_loc <= f.ppf(1 - 0.005, n-1, n-3):
            list_box_for_regression_char.insert(tk.END, str(round(f_fisher_loc,3)) +"<="+ str(round(f.ppf(1 - 0.005, n-1, n-3),3))+ "(Адекватна)")
        else:
            list_box_for_regression_char.insert(tk.END, str(round(f_fisher_loc,3)) +">"+ str(round(f.ppf(1 - 0.005, n-1, n-3),3))+ "(Не адекватна)")

        ax.plot(x, y_min_dov_new_hyper, color='yellow', alpha=1,linewidth=4)
        ax.plot(x, y_max_dov_new_hyper, color='yellow', alpha=1,linewidth=4)

        ax.plot(x, y_min_tol_hyper, color='green', alpha=1,linewidth=2, linestyle = "--")
        ax.plot(x, y_max_tol_hyper, color='green', alpha=1,linewidth=2, linestyle = "--")

        ax.plot(x, y_min_dov_hyper, color='orange', alpha=1,linewidth=2)
        ax.plot(x, y_max_dov_hyper, color='orange', alpha=1,linewidth=2)

        ax.plot(x, y, color='red', alpha=1,linewidth=1)



    canvas = FigureCanvasTkAgg(fig, master=tab2)
    canvas.draw()
    canvas.get_tk_widget().place(x=500,y=30)

    toolbar = NavigationToolbar2Tk(canvas, tab2, pack_toolbar=False)
    toolbar.update()
    toolbar.place(x=700,y=440)

def causs_val_change():
    global cuass_val, linear_val, hyperbolic
    cuass_val = True
    linear_val = False
    hyperbolic = False
    win(lines, listBoxCoef)
    

def linear_val_change():
    global cuass_val, linear_val, hyperbolic
    cuass_val = False
    linear_val = True
    hyperbolic = False
    win(lines, listBoxCoef)

def hyperbolic_val_change():
    global cuass_val, linear_val, hyperbolic
    cuass_val = False
    linear_val = False
    hyperbolic = True
    win(lines, listBoxCoef)

def click_graph():
    value = distribution_options_var.get()
    global hist_variant_hist, hist_variant_density
    figure1 = plt.Figure(figsize=(6, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    bar1 = FigureCanvasTkAgg(figure1, master=tab1)
    bar1.get_tk_widget().grid(row=1, column=2, columnspan=3)
    toolbar = NavigationToolbar2Tk(bar1, tab1, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row=2, column=2, columnspan=3)
    df1 = pd.DataFrame({'x': [round(v, 1) for v in left], 'y': height})
    df2 = pd.DataFrame({'y': [h for h in height]}, index=[v for v in range(len(left))])
    if hist_variant_hist:
        df1.plot.bar(x='x', y='y', width=1, color=['grey', 'black'], ax=ax1, legend=False)
    if hist_variant_density:
        f1 = interp1d(df1.index, df2['y'],kind='cubic',fill_value="extrapolate")
        new_index = np.arange(0,len(left)-0.9, 0.1)
        df3 = pd.DataFrame()
        df3['Функція щільності'] = f1(new_index)
        df3.index = new_index
        df3.plot.line(color="red", ax=ax1, legend=False)
    if hist_variant_param:
        if value == 'Експоненціальний':
            df3 = pd.DataFrame({'Відтворена функція щільності': [x*(max(height)/max(parametric_values_hist)) for x in parametric_values_hist]}, index=[x*(len(left)/max(lines)) for x in lines])
            df3.plot.line(color="green", ax=ax1, legend=False)
        elif value == 'Рівномірний':
            delt = (sum(height)/M) - min(parametric_values_hist)
            df3 = pd.DataFrame({'Відтворена функція щільності': [x + delt for x in parametric_values_hist]}, index=[x - (parametric_values_hist[0] - (lines[0]+(parametric_values_hist[0]-lines[-1]))) for x in lines])
            df3.plot.line(color="green", ax=ax1, legend=False) 
        elif value == 'Логнормальний':
            df3 = pd.DataFrame({'Відтворена функція щільності': [x*(max(height)/max(parametric_values_hist)) for x in parametric_values_hist]}, index=[(x*(len(left)/max(lines))) for x in lines])
            df3.plot.line(color="green", ax=ax1, legend=False) 
        elif value == 'Нормальний':
            df3 = pd.DataFrame({'Відтворена функція щільності': [x*(max(height)/max(parametric_values_hist)) for x in parametric_values_hist]}, index=[x*(len(left)/max(lines)) for x in lines])
            df3.plot.line(color="green", ax=ax1, legend=False) 
        elif value == 'Вейбула':
            df3 = pd.DataFrame({'Відтворена функція щільності': [y*(max(height)/max(parametric_values_hist)) for y in parametric_values_hist]}, index=[x*(len(left)/max(lines)) for x in lines])
            df3.plot.line(color="green", ax=ax1, legend=False) 
    ax1.set_title('Гістограма')

def click_graph_2():
    if calculation_data.array is not None:
        xpos, ypos, zpos, dx, dy, dz = calculation_data.get_heights()
        figure1 = plt.figure(figsize=(6, 5), dpi=100)
        ax1 = figure1.add_subplot(111, projection='3d')
        bar1 = FigureCanvasTkAgg(figure1, master=tab2)
        bar1.get_tk_widget().grid(row=1, column=0, columnspan=5, rowspan=5)
        toolbar = NavigationToolbar2Tk(bar1, tab2, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=6, column=0, columnspan=5)
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
        ax1.set_title('Гістограма')


def click_graph_emp():
    figure1 = plt.Figure(figsize=(6, 5), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.set_xlabel('x')
    ax1.set_ylabel('F(x)')
    bar1 = FigureCanvasTkAgg(figure1, master=tab1)
    bar1.get_tk_widget().grid(row=1, column=0, columnspan=2)
    toolbar = NavigationToolbar2Tk(bar1, tab1, pack_toolbar=False)
    toolbar.update()
    toolbar.grid(row=2, column=0, columnspan=2)
    ax1.set_title('Емпіричний графік функції')
    if variant_class:
        i = 0
        for x, y in height_emp:
            df1 = pd.DataFrame({'Класи': y}, index=x)
            df1.plot.line(color="gray", ax=ax1, legend=False)
            i += 1
    if variant_graph:
        df1 = pd.DataFrame(
            {'Емпірична фукція': list(map(lambda x: x / len(lines), range(1, len(lines) + 1)))}, index=lines)
        df1.plot.line(color="black", ax=ax1, legend=False)
    if variant_param:
        df1 = pd.DataFrame(
            {'Параметрична функція': parametric_values}, index=lines)
        df1.plot.line(color="red", ax=ax1, legend=False)
    if variant_confidence_intervals:
        df1 = pd.DataFrame({'Довірчий інтервал 1': confidence_intervals_1}, index=lines)
        df1.plot.line(color="blue", ax=ax1, legend=False)
        df2 = pd.DataFrame({'Довірчий інтервал 2': confidence_intervals_2}, index=lines)
        df2.plot.line(color="blue", ax=ax1, legend=False)

def log_checkbutton_changed():
    global lines, temp_0001
    if len(temp_0001) == 0:
        min_val = min(lines)
        lines_norm = lines[:]
        if min_val <= 0:
            lines_norm = list(
                map(lambda x: x + abs(min_val) + 0.01, lines_norm))
        temp_0001 = list(map(math.log, lines_norm))
        min_val = min(temp_0001)
        if min_val <= 0:
            temp_0001 = list(
                map(lambda x: x + abs(min_val) + 0.01, temp_0001))
    lines, temp_0001 = temp_0001, lines
    main()
    click_graph()
    click_graph_emp()
    list1.set(lines)
    list2.set(listBoxCoef)

def anomal_checkbutton_changed():
    global lines, temp_0002, x_average, x_sq_avg, coun_kurtosis
    print(x_average)
    print(x_sq_avg)
    print(coun_kurtosis)
    if len(temp_0002) == 0:
        N = len(lines)
        t = 1.2 + 3.6 * (1 - coun_kurtosis) * math.log10(N/10)
        a = x_average - t * x_sq_avg
        b = x_average + t * x_sq_avg
        temp_0002 = [x for x in lines if a < x < b]
    lines, temp_0002 = temp_0002, lines
    main()
    click_graph()
    click_graph_emp()
    list1.set(round_message_box_2(lines))
    list2.set(listBoxCoef)


def shift_checkbutton_changed():
    global lines, temp_0003
    if len(temp_0003) == 0:
        min_val = min(lines)
        if min_val <= 0:
            temp_0003 = list(
                map(lambda x: x + abs(min_val) + 0.01, lines))
        else:
            temp_0003 = lines[:]
    lines, temp_0003 = temp_0003, lines
    main()
    click_graph()
    click_graph_emp()
    list1.set(lines)
    list2.set(listBoxCoef)


def standardization_checkbutton_changed():
    global lines, temp_0004, x_average, x_sq_avg
    if len(temp_0004) == 0:
        temp_0004 = [(x-x_average)/x_sq_avg for x in lines]
    lines, temp_0004 = temp_0004, lines
    main()
    click_graph()
    click_graph_emp()
    list1.set(lines)
    list2.set(listBoxCoef)


def round_array(arr):
    rounded_arr = [round(num, 4) for num in arr]
    return rounded_arr

def round_message_box(x):
    if fabs(x) < 0.0001:
        return format_e(x)
    else:
        return round(x, 4)


def logarithmization(list):
    biggest_negative = max(list, key=lambda x: x if x < 0 else float('-inf'))
    list = [x - biggest_negative for x in list]
    return list


def reset_M():
    main()
    click_graph()
    click_graph_emp()
    list1.set(lines)
    list2.set(listBoxCoef)

def change_emp(event):
    click_graph_emp()

def make_table_connection():
    global n_entry_table, m_entry_table, text_for_box_m_n_table, ForMyCorrelationalString, listbox_m_n_table
    num = len(lines)

    listbox_m_n_table.delete(0, tk.END)
    text_for_box_m_n_table = []
    
    n = int(n_entry_table.get())
    m = int(m_entry_table.get())
    table = CorrelationalRegression.create_table_combinations_foo(ForMyCorrelationalString, n, m)
    for row in table:
        row_text = '     '.join(map(str, row))
        text_for_box_m_n_table.append(row_text)

    for line in text_for_box_m_n_table:
        listbox_m_n_table.insert(tk.END, line)

    listbox_m_n_table.insert(tk.END, "\n")

    coeff_spoluch_pirson, hi_for_pirson = CorrelationalRegression.coeff_spoluch_pirson_foo(table, num)
    
    listbox_m_n_table.insert(tk.END, "Коефіцієнт сполучень Пірсона = "+str(round(coeff_spoluch_pirson,3)))
    if hi_for_pirson > 5.99:
        listbox_m_n_table.insert(tk.END, "Коефіцієнт сполучень Пірсона значущий (hi) = "+ str(round(hi_for_pirson,3)) +" > 5.99")
    else:
        listbox_m_n_table.insert(tk.END, "Коефіцієнт сполучень Пірсона не значущий (hi) = "+ str(round(hi_for_pirson,3))+" <= 5.99")
    if n == m:
        listbox_m_n_table.insert(tk.END, "\n")
        coeff_spoluch_kendall = CorrelationalRegression.coeff_spoluch_kendall_foo(table,num)
        listbox_m_n_table.insert(tk.END, "Коефіцієнт сполучень Кендалла = "+str(round(coeff_spoluch_kendall,3)))
        u_for_kendel = 3*coeff_spoluch_kendall/math.sqrt(2*(2*num+5))*math.sqrt(num*(num-1))

        if abs(u_for_kendel) > 1.96:
            listbox_m_n_table.insert(tk.END, 'Перевірки значущості для коефіцієнта сполучень Кендалла = ' + str(abs(round(u_for_kendel,3)))+ ' > 1.96 (Значуща)\n')
        else:
            listbox_m_n_table.insert(tk.END, 'Перевірки значущості для коефіцієнта сполучень Кендалла = ' + str(abs(round(u_for_kendel,3)))+ ' <= 1.96 (Не значуща)\n')
    if n != m:
        listbox_m_n_table.insert(tk.END, "\n")
        coeff_spoluch_steward = CorrelationalRegression.coeff_spoluch_steward_foo(table, num)
        listbox_m_n_table.insert(tk.END, "Коефіцієнт сполучень Стюарда = "+str(round(coeff_spoluch_steward,3)))
    """
def del_anomal_for_x_y(data_x, data_y, num_bins):
    alpha = 0.05
    data_x, data_y = np.array(data_x), np.array(data_y)
    N = len(data_x)
    bins_x, bins_y = findM(N), findM(N)
    hist, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=[bins_x, bins_y])
    class_probs = hist / N
    for i in range(len(x_edges) - 1):
        for j in range(len(y_edges) - 1):
            class_prob = class_probs[i, j]
            if class_prob <= alpha:
                indices_to_remove = ((data_x >= x_edges[i]) & (data_x <= x_edges[i + 1]) &
                                     (data_y >= y_edges[j]) & (data_y <= y_edges[j + 1]))
                data_x = data_x[~indices_to_remove]
                data_y = data_y[~indices_to_remove]
    data_x = data_x.tolist()
    data_y = data_y.tolist()
    return data_x, data_y
"""



def del_anomal_for_x_y(X):
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    svm.fit(X)
    y_pred = svm.predict(X)
    anomalies = X[y_pred == -1]
    return anomalies.tolist()



def display_values():
    child_win_root_par = tk.Tk()
    child_win_root_par.title("Параметри")

    def linear_regression_modeling():
        n = int(entry_N.get())
        m1 = float(entry_m1.get())
        m2 = float(entry_m2.get())
        q1 = float(entry_q1.get())
        q2 = float(entry_q2.get())
        r = float(entry_r.get())
        filename = str(entry_filename.get())
        rand_value_1 = (np.random.normal(0, 1, n))
        rand_value_2 = (np.random.normal(0, 1, n))
        z1 = rand_value_1
        z2 = r * z1 + np.sqrt(1 - r ** 2) * rand_value_2
        x = m1 + q1 * z1
        y = m2 + q2 * z2
        data_modeling = np.column_stack((x, y))
        np.savetxt(filename, data_modeling, fmt='%0.3f', delimiter='\t')

    labels = ['N', 'm1', 'm2', 'q1', 'q2', 'r', 'Назва файлу']

    for i, label_text in enumerate(labels):
        label = tk.Label(child_win_root_par, text=label_text)
        label.grid(row=i, column=0, sticky='e', padx=5, pady=5)

    entry_N = tk.Entry(child_win_root_par)
    entry_m1 = tk.Entry(child_win_root_par)
    entry_m2 = tk.Entry(child_win_root_par)
    entry_q1 = tk.Entry(child_win_root_par)
    entry_q2 = tk.Entry(child_win_root_par)
    entry_r = tk.Entry(child_win_root_par)
    entry_filename = tk.Entry(child_win_root_par)

    entry_N.grid(row=0, column=1, padx=5, pady=5)
    entry_m1.grid(row=1, column=1, padx=5, pady=5)
    entry_m2.grid(row=2, column=1, padx=5, pady=5)
    entry_q1.grid(row=3, column=1, padx=5, pady=5)
    entry_q2.grid(row=4, column=1, padx=5, pady=5)
    entry_r.grid(row=5, column=1, padx=5, pady=5)
    entry_filename.grid(row=6, column=1, padx=5, pady=5)

    Button(child_win_root_par, text="Зберегти", command=linear_regression_modeling).grid(row=7, column=1, padx=5, pady=5)

    child_win_root_par.mainloop()

def display_values_for_hyper():
    child_win_root_par = tk.Tk()
    child_win_root_par.title("Параметри")

    def linear_regression_modeling():
        a = int(entry_a.get())
        b = int(entry_b.get())
        c = int(entry_c.get())
        shum = int(entry_shum.get())
        num = int(entry_num.get())
        filename = str(entry_filename.get())

        x = np.linspace(-10, 10, num)
        y = a+b*x+c*x**2 + np.random.normal(0, shum, num)

        data_modeling = np.column_stack((x, y))
        np.savetxt(filename, data_modeling, fmt='%0.3f', delimiter='\t')

    labels = ['a', 'b', 'c', 'shum', 'num', 'Назва файлу']

    for i, label_text in enumerate(labels):
        label = tk.Label(child_win_root_par, text=label_text)
        label.grid(row=i, column=0, sticky='e', padx=5, pady=5)

    entry_a = tk.Entry(child_win_root_par)
    entry_b = tk.Entry(child_win_root_par)
    entry_c = tk.Entry(child_win_root_par)
    entry_shum = tk.Entry(child_win_root_par)
    entry_num = tk.Entry(child_win_root_par)
    entry_filename = tk.Entry(child_win_root_par)

    entry_a.grid(row=0, column=1, padx=5, pady=5)
    entry_b.grid(row=1, column=1, padx=5, pady=5)
    entry_c.grid(row=2, column=1, padx=5, pady=5)
    entry_shum.grid(row=3, column=1, padx=5, pady=5)
    entry_num.grid(row=4, column=1, padx=5, pady=5)
    entry_filename.grid(row=5, column=1, padx=5, pady=5)

    Button(child_win_root_par, text="Зберегти", command=linear_regression_modeling).grid(row=6, column=1, padx=5, pady=5)

    child_win_root_par.mainloop()


def del_anomal_for_x_y_for_button():
    global calculation_data, M, choppedARRAYS
    data_x = calculation_data.array[0]
    data_y = calculation_data.array[1]
    
    calculation_data.array[0], calculation_data.array[1] = del_anomal_for_x_y(choppedARRAYS)
    main()
    
def plot_histogram_and_scatter():
    global choppedARRAYS, M
    
    def mean(lst):
        return sum(lst) / len(lst)

    def variance(lst):
        m = mean(lst)
        return sum((x - m) ** 2 for x in lst) / len(lst)

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
    
    data_list = choppedARRAYS
    n = len(data_list)
    
    fig, axes = plt.subplots(n, n, figsize=(8, 4.5), constrained_layout=True)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                data = data_list[i]
                bins = M
                min_val, max_val = min(data), max(data)
                bin_width = (max_val - min_val) / bins
                bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
                counts = [0] * bins
                
                for value in data:
                    for k in range(bins):
                        if bin_edges[k] <= value < bin_edges[k + 1]:
                            counts[k] += 1
                            break
                
                total = sum(counts)
                counts = [count / total for count in counts]
                
                axes[i, j].bar(bin_edges[:-1], counts, width=bin_width, color='skyblue', edgecolor='black')
            elif i > j:
                axes[i, j].scatter(data_list[i], data_list[j], alpha=0.5, s =10)
            else:
                corr = correlation_coeff(data_list[i], data_list[j])
                axes[i, j].text(0.5, 0.5, f'Парн.кор: {corr:.2f}', ha='center', va='center', fontsize=12)
                axes[i, j].axis('off') 

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].label_outer()
    
    canvas = FigureCanvasTkAgg(fig, master=tab3)
    canvas.draw()
    canvas.get_tk_widget().place(x=10, y=30)
    
def plot_samples_on_canvas():
    global choppedARRAYS
    samples = choppedARRAYS
    
    normalized_samples = []
    
    for sample in samples:
        min_val = min(sample)
        max_val = max(sample)
        
        if min_val == max_val:
            normalized_sample = [0.5] * len(sample)  
        else:
            normalized_sample = [(x - min_val) / (max_val - min_val) for x in sample]
        
        normalized_samples.append(normalized_sample)
    
    samples = normalized_samples

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, sample in enumerate(samples):
        x = [i + 1] * len(sample) 
        #ax.scatter(x, sample, color='black') 
        ax.vlines(x[0], ymin=0, ymax=1, color='black')  

    num_samples = len(samples)
    num_points = len(samples[0])
    for j in range(num_points):
        y_values = [samples[i][j] for i in range(num_samples)] 
        x_values = list(range(1, num_samples + 1))  
        ax.plot(x_values, y_values, color='red')  

    ax.set_xlim(1, len(samples))
    ax.set_title('Паралельні координати')
    ax.set_xticks(range(1, len(samples) + 1))

    # Вивід графіка на canvas
    canvas = FigureCanvasTkAgg(fig, master=tab3)  # Використання глобальної змінної tab3
    canvas.draw()
    canvas.get_tk_widget().place(x=820, y=30)
    
def copy_samples():
    global choppedARRAYS
    
    x1 = choppedARRAYS[0]
    x2 = choppedARRAYS[1]

    y_sample = choppedARRAYS[2]
    min_val = min(y_sample)
    max_val = max(y_sample)

    if min_val == max_val:
        normalized_sample = [0.5] * len(y_sample)
    else:
        normalized_sample = [(x - min_val) / (max_val - min_val) for x in y_sample]

    radii = [math.sqrt(value / math.pi) for value in normalized_sample]

    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    ax.scatter(x1, x2, s=[r**3 * 1000 for r in radii], alpha=0.6, edgecolors='w', color='blue')

    ax.set_xlabel(f'x {0}')
    ax.set_ylabel(f'x {1}')
    ax.set_title('Бульбашкова діаграма')
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=tab3)
    canvas.draw()
    canvas.get_tk_widget().place(x=900, y=500)
        
def plot_eps_y_lab_5():
    global choppedARRAYS, M
    myArrays_lab_5_plot = lab_5(choppedARRAYS)
    y_list, eps_l = lab_5.dots_for_diagnostic_diagram(myArrays_lab_5_plot)
    
    fig, ax= plt.subplots(figsize=(4, 2.5))
    
    ax.scatter(y_list, eps_l, color='blue', label='Data points', s=20)
    ax.set_xlabel('y_list')
    ax.set_ylabel('eps_l')
    ax.set_title('Діагностична діаграма')
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=tab3)
    canvas.draw()
    canvas.get_tk_widget().place(x=490, y=500)
    
        
def shift():
    global choppedARRAYS
    samples = choppedARRAYS
    shifted_samples = []
    for sample in samples:
        min_val = min(sample)
        shifted_samples.append([x - min_val for x in sample])
    choppedARRAYS = shifted_samples
    win(lines, listBoxCoef)

def logarithm():
    global choppedARRAYS
    samples = choppedARRAYS
    log_samples = []
    for sample in samples:
        log_samples.append([np.log(x) for x in sample if x > 0])
    choppedARRAYS = log_samples
    win(lines, listBoxCoef)

def standardize():
    global choppedARRAYS
    samples = choppedARRAYS
    standardized_samples = []
    for sample in samples:
        mean = np.mean(sample)
        variance = np.var(sample)
        std_dev = np.sqrt(variance)
        if std_dev == 0:
            standardized_samples.append([0] * len(sample))
        else:
            standardized_samples.append([(x - mean) / std_dev for x in sample])
    choppedARRAYS = standardized_samples
    win(lines, listBoxCoef)

def normalize():
    global choppedARRAYS
    samples = choppedARRAYS
    normalized_samples = []
    for sample in samples:
        min_val = min(sample)
        max_val = max(sample)
        if min_val == max_val:
            normalized_samples.append([0.5] * len(sample))
        else:
            normalized_samples.append([(x - min_val) / (max_val - min_val) for x in sample])
    choppedARRAYS = normalized_samples
    win(lines, listBoxCoef)

def win(arr, coef):
    global lines, listBoxCoef, anwer_for_homogen_indep, alfa_for_qwant
    global t_test_nums, param_rospod, anwer_for_correlational_regression, answer_for_lab_5_1, answer_for_lab_5_2, answer_for_lab_5_3
    global entry_x1, entry_x2, entry_y, list_for_lab_5_1, choppedARRAYS, str_entry_dots, str_entry_eigen, str_for_new_x, entry_w, list_for_lab_5_5
    global list_for_lab_find_sq
    if arr != None:

        list1.set(arr)
        list2.set(coef)

        Button(tab1, text='Відкрити файл', command=open_file_3).place(x = 1, y = 1)
        btn_emp_window = Button(tab1, text="Варіанти графіків з еміричною", command=my_w_child_open)
        btn_emp_window.place(x=210,y=1)

        btn_window = Button(tab1, text="Варіанти графіків з гістограмою", command=my_w_child_3_open)
        btn_window.place(x=810,y=1)

        distribution_dropdown = OptionMenu(
            tab1, distribution_options_var, *distribution_options, command=on_distribution_options_changed)
        distribution_dropdown.grid(row=0, column=5)

        #---------------------------------------------------------------------------------------------

        probabilistic_grid_window.place(x=100,y=1)
        probabilistic_grid_window["state"] = tk.DISABLED

        rounded_list = tk.StringVar(value=[round(x, 4) for x in list1.get()])
        list1.set(rounded_list)

        lst1 = tk.Listbox(tab1, listvariable=rounded_list)
        lst1.grid(row=3, column=1, rowspan=5)

        lst1.insert(0, "Елементів = "+str(len(lines)))

        lst2 = tk.Listbox(tab1_list, width=90, listvariable=list2)
        lst2.grid(row=0, column=0)

        param_rospod = tk.Listbox(tab2_list, width=90)
        param_rospod.grid(row=0, column=0)

        listBoxForHomogeneityIndependence = tk.Listbox(tab4_list, width=90)
        listBoxForHomogeneityIndependence.grid(row=0,column=0)

        listBoxForHomogeneityIndependence.delete(0, tk.END)

        listBoxForCorrelationalRagression = tk.Listbox(tab2, width=120, height=15)
        listBoxForCorrelationalRagression.place(x=10,y=500)

        listBoxForCorrelationalRagression.delete(0, tk.END)

        linesHomogeneityIndep = anwer_for_homogen_indep.strip().split('\n')
        linesCorrlation = anwer_for_correlational_regression.strip().split('\n')           

        for line in linesHomogeneityIndep:
            listBoxForHomogeneityIndependence.insert(tk.END, line)

        for line in linesCorrlation:
            listBoxForCorrelationalRagression.insert(tk.END, line)

        param_rospod.insert(tk.END, "Розподіл не вибраний")

        t_test_nums = tk.Listbox(tab3_list, width=90)
        t_test_nums.grid(row=0, column=0)

        t_test_nums.insert(tk.END, "Кількість т-тестів / Кількість елементів для 1 т-тесту / Седнеє арифметичне / Середнє квадратичне")

        tabControl_for_list.grid(row=3, column=2, rowspan=5)

        log_checkbutton = tk.Checkbutton(
            tab1, text="Логарифмування", variable=log_val_var, command=log_checkbutton_changed)
        log_checkbutton.grid(row=3, column=0)
        anomal_checkbutton = tk.Checkbutton(
            tab1, text="Вилучення аномальних значень", variable=anomal_val_var, command=anomal_checkbutton_changed)
        anomal_checkbutton.grid(row=4, column=0)
        shift_checkbutton = tk.Checkbutton(
            tab1, text="Зсув", variable=shift_val_var, command=shift_checkbutton_changed)
        shift_checkbutton.grid(row=5, column=0)
        standardization_checkbutton = tk.Checkbutton(
            tab1, text="Стандартизація", variable=standardization_val_var, command=standardization_checkbutton_changed)
        standardization_checkbutton.grid(row=6, column=0)

        Entry(tab1, textvariable=custom_M).grid(row=7, column=0)

        Button(tab1, text='Встановити кількість класів', command=reset_M).grid(row=8, column=0)
        Button(tab1, text='Змоделювати нову вибірку', command=create_file).grid(row=8, column=1)
        Button(tab1, text='Відкрити файл т-тесту', command=open_file_t_test).place(x = 530, y = 1)
        Button(tab1, text='Змінити стовпець', command=change_column).place(x = 680, y = 1)
        Button(tab2, text='Змінити стовпець x', command=change_column_2_x).grid(row=0,column=1)
        Button(tab2, text='Змінити стовпець y', command=change_column_2_y).grid(row=0,column=2)

        Button(tab3, text='Змінити початковий стовпець', command=change_column_3_x).grid(row=0,column=1)
        Button(tab3, text='Змінити кінцевий стовпець', command=change_column_3_y).grid(row=0,column=2)

        Button(tab1, text='Зберегти поточну вибірку', command=my_w_child_4_open).grid(row=1,column=6)

        entry_alfa=Entry(tab1)
        entry_alfa.grid(row=2,column=6)
        def get_alfa_for_qwant():
            alfa_for_qwant = float(entry_alfa.get())
            win(lines, listBoxCoef)
            return alfa_for_qwant

        entry_button=Button(tab1,text = "Рівень значущосиі",command=get_alfa_for_qwant).grid(row=3,column=6)

        Button(tab2, text='Побудувати таблицю сполучень',command=make_table_connection).place(x=750,y=500)
        Label(tab2, text="n =").place(x=750,y=530)
        global n_entry_table 
        n_entry_table = Entry(tab2,width=5)
        n_entry_table.place(x=780,y=530)

        global m_entry_table
        Label(tab2, text="m =").place(x=850,y=530)
        m_entry_table = Entry(tab2,width=5)
        m_entry_table.place(x=880,y=530)

        global listbox_m_n_table
        listbox_m_n_table = tk.Listbox(tab2, width=80, height=11)
        listbox_m_n_table.place(x=750,y=560)

        Button(tab2, text='Видалити аномалії', command=del_anomal_for_x_y_for_button).place(x=950,y=500)

        Button(tab2, text='Побудувати лінійну', command=linear_val_change).place(x=1080,y=500)
        Button(tab2, text='Побудувати гіперболічну', command=hyperbolic_val_change).place(x=1200,y=500)
        Button(tab2, text='Побудувати квазилінійну', command=causs_val_change).place(x=1350,y=500)

        Button(tab2, text='Відтворення лінійної', command=display_values).place(x=1080,y=530)
        Button(tab2, text='Відтворення параболічної', command=display_values_for_hyper).place(x=1180,y=530)

        global list_box_for_regression_char 
        list_box_for_regression_char = tk.Listbox(tab2, width=40,height=11)
        list_box_for_regression_char.place(x=1240, y = 560)
                
        tabControl_for_lab_5.place(x=10,y=500)
        
        calc_lab_5_1_list_box_values()
        calc_lab_5_2_list_box_values()
        calc_lab_5_3_list_box_values()
        mgk()
                
        list_for_lab_5_1 = tk.Listbox(tab1_lab_5, width=50, height=10)
        list_for_lab_5_1.grid(row=0, column=0)
        
        list_for_lab_5_2 = tk.Listbox(tab2_lab_5, width=50, height=10)
        list_for_lab_5_2.grid(row=0, column=0)
        
        list_for_lab_5_3 = tk.Listbox(tab3_lab_5, width=50, height=10)
        list_for_lab_5_3.grid(row=0, column=0)
        
        list_for_lab_5_5 = tk.Listbox(tab4, width=130, height=21)
        list_for_lab_5_5.place(x=730,y=40)
        
        list_for_lab_find_sq = tk.Listbox(tab4, width=55, height=24)
        list_for_lab_find_sq.place(x=765,y=380)
        
        text_widget = tk.Text(tab4, wrap='none',width=89,height=21)
        text_widget.place(x=10,y=40)
        
        list_answer_for_lab_5_1 = answer_for_lab_5_1.strip().split('\n')  
        
        list_answer_for_lab_5_2 = answer_for_lab_5_2.strip().split('\n')  
        
        list_answer_for_lab_5_3 = answer_for_lab_5_3.strip().split('\n') 
          
        
        for line in list_answer_for_lab_5_1:
            list_for_lab_5_1.insert(tk.END, line)
            
        for line in list_answer_for_lab_5_2:
            list_for_lab_5_2.insert(tk.END, line)
            
        for line in list_answer_for_lab_5_3:
            list_for_lab_5_3.insert(tk.END, line)

        text_widget.insert(tk.END, str_entry_eigen)
        
        str_entry_dots = Entry(tab3, width=15)
        str_entry_dots.place(x=330,y=550)
        Button(tab3, text='Довірчий інтервал для \n значення регресії',command=mes_box_full).place(x=330,y=500)
            
        Button(tab3, text='Зсув',width=15, command=shift).place(x=330,y=590)
        Button(tab3, text='Логарифмування',width=15, command=logarithm).place(x=330,y=630)
        Button(tab3, text='Стандартизація',width=15, command=standardize).place(x=330,y=670)
        Button(tab3, text='Нормалізація',width=15, command=normalize).place(x=330,y=710)
        
        Button(tab3, text='Збіг середніх для 2',command=parse_indices_2_mean).place(x=10,y=710)
        Button(tab3, text='Збіг середніх для k',command=parse_indices_k_mean).place(x=125,y=710)
        Button(tab3, text='Збіг ДК для k',command=parse_indices_k_DK).place(x=240,y=710)
        Button(tab3, text='Зберегти новий порядок',command=save_new_order).place(x=440,y=0)
        Button(tab3, text='Вилучити аномалії',command=clear_lists).place(x=640,y=0)
        
        Button(tab4, text='Незалежні ознаки', command=new_x_mgk).place(x=10,y=390)
        Button(tab4, text='Повернення по n',command=recon_x_mgk).place(x=120,y=390)
        Button(tab4, text='Повернення по w =',command=recon_x_w_mgk).place(x=225,y=390)
        entry_w = Entry(tab4, width=2)
        entry_w.place(x=350,y=393)
        Button(tab4, text='Рівняння площини',command=output_plane_equation).place(x=370,y=390)
        

        click_graph()
        plot_sectors_and_points()
        plot_points()
        click_graph_emp()
        plot_2d_norm_graph()
        plot_histogram_and_scatter()
        plot_samples_on_canvas()
        copy_samples()
        plot_eps_y_lab_5()
        plot_3d_x_y_z()

    else:
        Button(tab1, text='Відкрити файл', command=open_file_3).place(x=0, y=0)
        Button(tab2, text='Відкрити файл', command=open_file_3).grid(row=0, column=0)
        Button(tab3, text='Відкрити файл', command=open_file_3).grid(row=0, column=0)
        Button(tab4, text='Відкрити файл', command=open_file_3).grid(row=0, column=0)
        
    root.mainloop()
    
def output_plane_equation():
    global list_for_lab_find_sq
    find_square_elements = FindSquare(choppedARRAYS)
    A, B, C, D = FindSquare.plane_equation(find_square_elements)
    text_plane_equation = f"Рівняння площини: \n {round(A,4)}x1 + {round(B,4)}x2 + {round(C,4)}x3 + {round(D,4)} = 0 \n"
    for row in text_plane_equation.split('\n'):
        list_for_lab_find_sq.insert(tk.END, row)
    
    
def clear_lists():
    global choppedARRAYS
    choppedARRAYS = lab_5.del_anomaly(choppedARRAYS) 
    win(lines, listBoxCoef)
    
    
def save_new_order():
    global choppedARRAYS, str_entry_dots
    
    str_input_for_groups = str(str_entry_dots.get())
    
    indices = list(map(int, str_input_for_groups.split()))

    reordered_samples = [choppedARRAYS[i] for i in indices]

    transposed_samples = list(zip(*reordered_samples))

    with open("new_order.txt", 'w') as file:
        for row in transposed_samples:
            row_str = ' '.join(map(str, row))
            file.write(row_str + '\n')
    
    
def parse_indices_k_DK():
    global choppedARRAYS, str_entry_dots
    
    str_input_for_groups = str(str_entry_dots.get())
    args = choppedARRAYS
    
    groups = str_input_for_groups.split(' | ')
    result = []
    for group in groups:
        indices = map(int, group.split())
        
        lists = [args[index] for index in indices]
        
        result.append(lists)
        
    V, v = lab_5.hipotesa_k_dk(np.array(result))
    
    str_answ =""
    
    str_answ += f'V = {V:.4f}\n'
    
    if V <= stats.chi2.ppf(1 - 0.05, v):
        str_answ += 'Дк збігаються.\n'
    else:
        str_answ += 'Дк не збігаються.\n'
        
    messagebox.showinfo("Перевірка збігу дк", str_answ)

def parse_indices_k_mean():
    global choppedARRAYS, str_entry_dots
    
    str_input_for_groups = str(str_entry_dots.get())
    args = choppedARRAYS
    
    groups = str_input_for_groups.split(' | ')
    result = []
    for group in groups:
        indices = map(int, group.split())
        
        lists = [args[index] for index in indices]
        
        result.append(lists)
    
    V, v = lab_5.hipotesa_k_means(np.array(result))
    
    
    str_answ =""
    
    str_answ += f'V = {V:.4f}\n'
    
    if V <= stats.chi2.ppf(1 - 0.05, v):
        str_answ += 'Середні збігаються\n'
    else:
        str_answ += 'Середні не збігаються\n'
        
    messagebox.showinfo("При різних ДК перевірки двох векторів середніх", str_answ)

def parse_indices_2_mean():
    global choppedARRAYS, str_entry_dots
    
    str_input_for_groups = str(str_entry_dots.get())
    args = choppedARRAYS
    
    groups = str_input_for_groups.split(' | ')
    result = []
    for group in groups:
        indices = map(int, group.split())
        
        lists = [args[index] for index in indices]
        
        result.append(lists)
        
    V, n_num_for_v = lab_5.hipotesa_2_means(np.array(result))
    
    str_answ = ""
    
    str_answ += f'V = {V:.4f}\n'

    if V <= stats.chi2.ppf(0.95, n_num_for_v):
        str_answ += 'Середні збігаються.\n'
    else:
        str_answ += 'Середні не збігаються.\n'
        
    messagebox.showinfo("При рівних ДК перевірки двох векторів середніх", str_answ)

def recon_x_mgk():
    global choppedARRAYS, str_for_new_x, list_for_lab_5_5
    myArrays_lab_5_new_x_mgk = lab_5(choppedARRAYS)
    find_back_n_x_nums = (lab_5.find_recon_x(myArrays_lab_5_new_x_mgk))
        
    transposed_matrix = np.array(find_back_n_x_nums).T
    
    str_for_new_x = ""
    str_for_new_x += f'Повернення по n:\n'
    
    for row in find_back_n_x_nums:
        str_for_new_x += '      '.join(f"{float(elem):.4f}" for elem in row) + '\n'
    for line in str_for_new_x.split('\n'):
            list_for_lab_5_5.insert(tk.END, line)
            
    file_path = os.path.join('new_data', 'ortog_back_n.txt')
    os.makedirs('new_data', exist_ok=True)
    
    with open(file_path, 'w') as file:
        for row in transposed_matrix:
            file.write(' '.join(map(str, row)) + '\n')
            
def recon_x_w_mgk():
    global choppedARRAYS, entry_w, str_for_new_x, list_for_lab_5_5
    w = int(entry_w.get())
    myArrays_lab_5_new_w_x_mgk = lab_5(choppedARRAYS)
    find_back_w_x_nums = (lab_5.find_recon_w_x(myArrays_lab_5_new_w_x_mgk,w))
        
    transposed_matrix = np.array(find_back_w_x_nums).T
    
    str_for_new_x = ""
    str_for_new_x += f'Повернення по w:\n'
    
    for row in find_back_w_x_nums:
        str_for_new_x += '      '.join(f"{float(elem):.4f}" for elem in row) + '\n'
        
    for line in str_for_new_x.split('\n'):
            list_for_lab_5_5.insert(tk.END, line)
    
    file_path = os.path.join('new_data', 'ortog_back_w.txt')
    os.makedirs('new_data', exist_ok=True)
    
    with open(file_path, 'w') as file:
        for row in transposed_matrix:
            file.write(' '.join(map(str, row)) + '\n')

    

def new_x_mgk():
    global choppedARRAYS, str_for_new_x, list_for_lab_5_5
    myArrays_lab_5_old_x_mgk = lab_5(choppedARRAYS)
    find_forward_x_nums = (lab_5.find_new_x(myArrays_lab_5_old_x_mgk))
        
    transposed_matrix = np.array(find_forward_x_nums).T
    
    str_for_new_x = ""
    str_for_new_x += f'Незалежні:\n'
    for row in find_forward_x_nums:
        str_for_new_x += '      '.join(f"{float(elem):.4f}" for elem in row) + '\n'

        
    for line in str_for_new_x.split('\n'):
            list_for_lab_5_5.insert(tk.END, line)
            
    file_path = os.path.join('new_data', 'ortog_forward.txt')
    os.makedirs('new_data', exist_ok=True)
    
    with open(file_path, 'w') as file:
        for row in transposed_matrix:
            file.write(' '.join(map(str, row)) + '\n')
    

def mgk():
    global choppedARRAYS, str_entry_eigen
    myArrays_lab_5_mgk = lab_5(choppedARRAYS)
    eigenvalues, V, fractions, cumulative_sums  = lab_5.yakobi(myArrays_lab_5_mgk)

    num_columns = len(V[0])

    headers = [" "] + [f"x'{i+1}" for i in range(num_columns)]
    row_names = [f"x{i+1}" for i in range(num_columns+1)]
    
    A_with_row_names = [[row_name] + row for row_name, row in zip(row_names[:-1], V)]
    
    A_with_row_names.append(["Lambda"] + eigenvalues)
    
    A_with_row_names.append(["Частка"] + fractions)
    
    A_with_row_names.append(["Накопичення"] + cumulative_sums)
    
    str_entry_eigen = tabulate(A_with_row_names, headers, tablefmt="grid", floatfmt=".4f")
    


def mes_box_full():
    global choppedARRAYS, str_entry_dots
    myArrays_lab_5_4 = lab_5(choppedARRAYS)
    str_to_read = str(str_entry_dots.get())
    text_for_mes_box_full = lab_5.dov_intrv_regression(myArrays_lab_5_4,str_to_read)
    messagebox.showinfo("Довірчий інтервал для точки", text_for_mes_box_full)
    

def calc_lab_5_2_list_box_values():
    global answer_for_lab_5_2, choppedARRAYS
    answer_for_lab_5_2 = ""
    myArrays_lab_5_2 = lab_5(choppedARRAYS)
    answer_for_lab_5_2 +=f'Частковий коефіцієнт кореляції: \n'
    r_ijc = lab_5.correlation_coeff_for_3(myArrays_lab_5_2)
    answer_for_lab_5_2 +=f'{round(r_ijc,3)} \n'
    
    znach_coeff_corr_for_3 = lab_5.znach_corr_coeff_for_3(myArrays_lab_5_2)
    t_critical =  stats.t.ppf(1 - 0.05/2, len(choppedARRAYS[0])-len(choppedARRAYS)-2)
    down, up = lab_5.dovirchi_intervalu_coeff_cor_for_3(myArrays_lab_5_2)
    if znach_coeff_corr_for_3 < float(t_critical):
        answer_for_lab_5_2 +=f'Значущий\n'
        answer_for_lab_5_2 +=f'Довірчі:\n'
        answer_for_lab_5_2 +=f'({round(down,3)};{round(up,3)})\n'
    else:
        answer_for_lab_5_2 +=f'Незначущий\n'
    answer_for_lab_5_2 +=f'\n'
    
    mult_corr_coef_for_n = lab_5.mult_corr_coef_for_n(myArrays_lab_5_2)
    answer_for_lab_5_2 +=f'Множиний коефіцієнт кореляції\n'
    answer_for_lab_5_2 +=f'{round(mult_corr_coef_for_n,3)}\n'
    quantile = stats.f.ppf(1 - 0.05, len(choppedARRAYS), len(choppedARRAYS[0])-len(choppedARRAYS)-1)
    znach_mult_corr_coef_for_n = lab_5.znach_mult_corr_coef_for_n(myArrays_lab_5_2)
    if znach_mult_corr_coef_for_n < quantile:
        answer_for_lab_5_2 +=f'Значуща\n'
    else:
        answer_for_lab_5_2 +=f'Незначуща\n'
    
def calc_lab_5_3_list_box_values():
    global answer_for_lab_5_3, choppedARRAYS
    answer_for_lab_5_3 = ""
    myArrays_lab_5_3 = lab_5(choppedARRAYS)
    A_vectors_without_a0 = lab_5.find_A_vector_without_a_0(myArrays_lab_5_3)
    answer_for_lab_5_3 +=f'Оцінки параметрів без а0\n'
    for row in A_vectors_without_a0:
        answer_for_lab_5_3 +=f'{round(row,3)}\n'
    answer_for_lab_5_3 +=f'\n'
    
    answer_for_lab_5_3 +=f'Оцінки параметрів разом із а0\n'
    A_vectors_with_a0 = lab_5.find_A_vector_with_a_0(myArrays_lab_5_3)
    for row in A_vectors_with_a0:
        answer_for_lab_5_3 +=f'{round(row,3)}\n'
    answer_for_lab_5_3 +=f'\n'
    
    answer_for_lab_5_3 +=f'Перевірка значущості відтвореної регресії\n'
    f_regretion = lab_5.znach_regression(myArrays_lab_5_3)
    critical_value = f.ppf(1 - 0.05, len(choppedARRAYS), len(choppedARRAYS[0])-len(choppedARRAYS)-1-1)
    if f_regretion > critical_value:
        answer_for_lab_5_3 +=f'Значуща\n'
    else:
        answer_for_lab_5_3 +=f'Незначуща\n'
    answer_for_lab_5_3 +=f'\n'
    
    t_a_znach, down_list, up_list = lab_5.znach_a(myArrays_lab_5_3)
    answer_for_lab_5_3 +=f'Перевірка значущості оцінок параметрів\n'
    for i in range(len(t_a_znach)):
        if abs(t_a_znach[i]) <= t.ppf(1 - 0.05, len(choppedARRAYS[0])-len(choppedARRAYS)-1):
            answer_for_lab_5_3 +=f'a{i+1} значущий\n'
        else:
            answer_for_lab_5_3 +=f'a{i+1} незначущий\n'
    answer_for_lab_5_3 +=f'\n'
            
    answer_for_lab_5_3 +=f'Довірчі інтервали оцінок параметрів\n'
    for i in range(len(t_a_znach)):
        answer_for_lab_5_3 +=f'[{down_list[i]}] ; [{up_list[i]}]\n'
    answer_for_lab_5_3 +=f'\n'


    a_standart = lab_5.standart_param(myArrays_lab_5_3)
    answer_for_lab_5_3 +=f'Стандартизовані оціноки параметрів\n'
    for i in range(len(t_a_znach)):
        answer_for_lab_5_3 +=f'{round(a_standart[i],3)}\n'
    answer_for_lab_5_3 +=f'\n'
    
    answer_for_lab_5_3 +=f'Коефіцієнт детермінації\n'
    deter_coeff = lab_5.deter_coeff(myArrays_lab_5_3)
    answer_for_lab_5_3 +=f'R^2 = {round(deter_coeff,3)}\n'
    answer_for_lab_5_3 +=f'\n'
    
    sigma_sq, down_dov_sigma, up_dov_sigma = lab_5.dov_interv_sigma(myArrays_lab_5_3)
    answer_for_lab_5_3 +=f'Залишкова дисперсія\n'
    answer_for_lab_5_3 +=f'{round(math.sqrt(sigma_sq),3)}\n'
    answer_for_lab_5_3 +=f'Довірчий інтервал залишкової дисперсії\n'
    answer_for_lab_5_3 +=f'[{round(math.sqrt(down_dov_sigma),3)}] ; [{round(math.sqrt(up_dov_sigma),3)}]\n'
    answer_for_lab_5_3 +=f'\n'
    
def calc_lab_5_1_list_box_values():
    global answer_for_lab_5_1, choppedARRAYS
    answer_for_lab_5_1 = ""
    myArrays_lab_5 = lab_5(choppedARRAYS)
    averages_lab_5 = lab_5.calculate_averages(myArrays_lab_5)
        
    answer_for_lab_5_1 +=f'Вектори середніх \n'
    for number in range(len(averages_lab_5)):
        answer_for_lab_5_1 +=f'x{number+1} => {round(averages_lab_5[number],3)} \n'
    answer_for_lab_5_1 +=f'\n'
            
    variances_lab_5 = lab_5.calculate_variances(myArrays_lab_5)
    answer_for_lab_5_1 +=f'Вектори середньоквадратичних \n'
    for number in range(len(variances_lab_5)):
        answer_for_lab_5_1 +=f'x{number+1} => {round(variances_lab_5[number],3)} \n'
    answer_for_lab_5_1 +=f'\n'
        
    cov_matrix_lab_5 = lab_5.calculate_std_matrix(myArrays_lab_5)
    cov_matrix_lab_5 = np.round(cov_matrix_lab_5, decimals=3)
    answer_for_lab_5_1 +=f'ДК \n'
    for number in range(len(cov_matrix_lab_5)):
        answer_for_lab_5_1 +=f'{cov_matrix_lab_5[number]} \n'
    answer_for_lab_5_1 +=f'\n'

def sort(array, size):
    for s in range(size):
        min_idx = s

        for i in range(s + 1, size):
            if array[i] < array[min_idx]:
                min_idx = i
        (array[s], array[min_idx]) = (array[min_idx], array[s])

    return array


def Vk(N, x, k):
    return 1/N*sum(list(map(lambda x: x**k, x)))


def Mk(N, x, nu, k):
    return 1/N*sum(list(map(lambda x: (x-nu)**k, x)))

def F_lognorm(x, sig, m):
    if x <= 0:
        return 0
    else:
        return 0.5 + 0.5*scipy.special.erf((np.log(x)-m)/(sig*np.sqrt(2)))
    
def F_normal(x, m, sig):
    ro = 0.2316419
    b_1 = 0.31938153
    b_2 = -0.356563782
    b_3 = 1.781477937
    b_4 = -1.821255978
    b_5 = 1.330274429
    e_u = 7.8*(10)**(-8)
    u = (x-m)/sig
    if u < 0:
        t = 1/(1+ro*abs(u))
        return 1 - (1 - 1/np.sqrt(2*np.pi) * np.exp(-u**2/2) * (b_1*t + b_2*t**2 + b_3*t**3 + b_4*t**4 + b_5*t**5)  + e_u)
    else:
        t = 1/(1+ro*u)
        return 1 - 1/np.sqrt(2*np.pi) * np.exp(-u**2/2) * (b_1*t + b_2*t**2 + b_3*t**3 + b_4*t**4 + b_5*t**5)  + e_u

def F_expon(x, lam):
    if x < 0:
        return 0
    else:
        return 1-math.exp(-lam*x)
    
def F_rivnom(x, a, b):
    if x < a:
        return 0
    elif x >= b:
        return 1
    else:
        return (x-a)/(b-a)

def F_veib(x, alf, bet):
    return 1 - np.exp(-x**bet/alf)

def count_hi_sq_pirson(ni_original, ni_distrib, n):
    hi_sq = 0.0
    for i in range(len(ni_original)):
        hi_sq += ((ni_original[i]*n - ni_distrib[i]) ** 2) / ni_distrib[i]
    return hi_sq

def main():
    global height, height_emp, lines, krok, lines_min_max, listBoxCoef, anwer_for_homogen_indep
    global x_average, x_sq_avg, coun_kurtosis, asym_coef, kurtosis, pirson, x_2_average
    global ni_original, list_class_borders, z_list_for_graph
    global M, alfa_for_qwant, anwer_for_correlational_regression, answer_for_lab_5_1, list_for_lab_5_1
    global calculation_data, ARRAYS, lines, choppedARRAYS
    listBoxCoef.clear()

    choppedARRAYS = ARRAYS[start_column:end_column+1].copy()

    lines = sort(lines, len(lines))

    def check_equal_length(list_of_lists):
        return all(len(sublist) == len(list_of_lists[0]) for sublist in list_of_lists)

    isEq = check_equal_length(choppedARRAYS)

    N = len(lines)

    if custom_M.get() == '':
        M = findM(N)
    else:
        M = int(custom_M.get())
    new_lines = lines[:]
    if new_lines[0] > 0:
        new_lines[0] *= 0.99
    else:
        new_lines[0] *= 1.01
    if new_lines[-1] > 0:
        new_lines[-1] *= 1.01
    else:
        new_lines[-1] *= 0.99

    krok = (new_lines[-1] - new_lines[0])/M

    mid_num = findMid(lines)


    left.clear()
    left.append(mid_num)
    while mid_num < lines[-1]:
        mid_num = mid_num + krok
        left.append(mid_num)
    left.pop()

    numb_of_col = len(left)
    height.clear()
    height_emp.clear()
    xi = new_lines[0]

    for i in range(numb_of_col):
        height.append(0)
        height_emp.append(sum(height))
        for j in range(len(lines)):
            if xi < lines[j] and lines[j] <= xi+krok:
                height[i] += 1
                height_emp[i] += 1
        height_emp[-1] = [[xi, xi + krok],
                          [height_emp[-1] / N, height_emp[-1] / N]]
        xi += krok

    for k in range(len(height)):
        height[k] = (height[k]/N)

    ni_original = height.copy()

    xi_for_hi_pirson = lines[0]
    for i in range(M+1):
        list_class_borders.append(xi_for_hi_pirson)
        xi_for_hi_pirson = xi_for_hi_pirson + krok
    
    x_average = (sum(lines)/N)
    x_sq_avg = math.sqrt(sum([math.pow(x - x_average, 2) for x in lines]) / N)
    asym_coef = (math.sqrt(N*(N-1))/(N-2)) * \
        (Mk(N, lines, Vk(N, lines, 1), 3))/(x_sq_avg**3)
    kurtosis = ((N**2-1)/((N-2)*(N-3))) * \
        ((((Mk(N, lines, Vk(N, lines, 1), 4))/(x_sq_avg**4))-3)+(6/(N+1)))
    coun_kurtosis = 1/(math.sqrt(abs(kurtosis)))

    med = find_MED(lines)
    mad = find_MAD(lines, med)
    pirson = mad/med

    x_2_average = (sum(map(lambda x: x**2, lines))/N)

    #inf_sup_skv
    sigma_x_average = x_sq_avg/np.sqrt(N)
    inf_sigma_x_average = x_average - 1.965*sigma_x_average
    sup_sigma_x_average = x_average + 1.965*sigma_x_average

    sigma_x_sq_avg = np.sqrt(2/(N-1))*x_sq_avg**2
    inf_sigma_x_sq_avg = x_sq_avg - 1.965*sigma_x_sq_avg
    sup_sigma_x_sq_avg = x_sq_avg + 1.965*sigma_x_sq_avg

    sigma_asym_coef = np.sqrt((6*(N-2))/((N+1)*(N+3)))
    inf_sigma_asym_coef = asym_coef - 1.965*sigma_asym_coef
    sup_sigma_asym_coef = asym_coef + 1.965*sigma_asym_coef

    sigma_kurtosis = np.sqrt((24*N*(N-1)**2)/((N-3)*(N-2)*(N+3)*(N+5)))
    inf_sigma_kurtosis = kurtosis - 1.965*sigma_kurtosis
    sup_sigma_kurtosis = kurtosis + 1.965*sigma_kurtosis

    kurtosis_1 =(Mk(N, lines, Vk(N, lines, 1), 4))/(x_sq_avg**4)
    sigma_coun_kurtosis = np.sqrt(abs(kurtosis_1)/(29*N))*((abs(kurtosis_1**2-1))**3)**(1/4)
    inf_sigma_coun_kurtosis = coun_kurtosis - 1.965*sigma_coun_kurtosis
    sup_sigma_coun_kurtosis = coun_kurtosis + 1.965*sigma_coun_kurtosis

    sigma_pirson = pirson*np.sqrt((1+2*pirson**2)/(2*N))
    inf_sigma_pirson = pirson - 1.965*sigma_pirson
    sup_sigma_pirson = pirson + 1.965*sigma_pirson

    on_distribution_options_changed(distribution_options_var.get())

    listBoxCoef.append("Оцінка                           /             Нижній довірчий        /          Верхній довірчий       /           СКВ")
    listBoxCoef.append(f"Середнє арифметичне: {(round_message_box(x_average))}    /          {round_message_box(inf_sigma_x_average)}             /                     {round_message_box(sup_sigma_x_average)}               /   {round_message_box(sigma_x_average)}")
    listBoxCoef.append(f"Середнє квадратичне: {(round_message_box(x_sq_avg))}    /          {round_message_box(inf_sigma_x_sq_avg)}             /                     {round_message_box(sup_sigma_x_sq_avg)}               /   {round_message_box(sigma_x_sq_avg)}")
    listBoxCoef.append(f"коефіцієнт асиметрії: {(round_message_box(asym_coef))}    /          {round_message_box(inf_sigma_asym_coef)}             /                     {round_message_box(sup_sigma_asym_coef)}               /   {round_message_box(sigma_asym_coef)}")
    listBoxCoef.append(f"Коефіцієнт екцесу: {(round_message_box(kurtosis))}        /          {round_message_box(inf_sigma_kurtosis)}             /                     {round_message_box(sup_sigma_kurtosis)}               /   {round_message_box(sigma_kurtosis)}")
    listBoxCoef.append(f"Коефіцієнт контрекцесу: {(round_message_box(coun_kurtosis))} /       {round_message_box(inf_sigma_coun_kurtosis)}             /                     {round_message_box(sup_sigma_coun_kurtosis)}               /   {round_message_box(sigma_coun_kurtosis)}")
    listBoxCoef.append(f"Коефіцієнт Пірсона: {(round_message_box(pirson))}    /          {round_message_box(inf_sigma_pirson)}             /                     {round_message_box(sup_sigma_pirson)}               /   {round_message_box(sigma_pirson)}")
    
    myArraysHomogenIndep = HomogeneityIndependence(calculation_data.array, choppedARRAYS)

    anwer_for_homogen_indep = ""

    anwer_for_homogen_indep += "Для незалежних: \n"   

    answer1_coincidenceMeanIndep = abs(HomogeneityIndependence.coincidenceMeanIndep(myArraysHomogenIndep))
    qw_1 =(len(calculation_data.array[0])+len(calculation_data.array[1])-1)
    if answer1_coincidenceMeanIndep > 1.96:
        anwer_for_homogen_indep += f'Збіг середніх (представницькі) = {round(answer1_coincidenceMeanIndep,3)} (t) > {1.96} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'Збіг середніх (представницькі) = {round(answer1_coincidenceMeanIndep,3)} (t) <= {1.96} (Однорідні)\n'

    answer1_coincidenceVarianceIndep = HomogeneityIndependence.coincidenceVarianceIndep(myArraysHomogenIndep)
    qw_2_1 = (len(calculation_data.array[0])-1)
    qw_2_2 = (len(calculation_data.array[1])-1)
    if answer1_coincidenceVarianceIndep > 3.08:
        anwer_for_homogen_indep += f'Збіг дисперсій = {round(answer1_coincidenceVarianceIndep,3)} (f) > {3.08} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'Збіг дисперсій = {round(answer1_coincidenceVarianceIndep,3)} (f) <= {3.08} (Однорідні)\n'

    answer1_wilcoxon = abs(HomogeneityIndependence.wilcoxon(myArraysHomogenIndep))
    qw_3 = 0
    if answer1_wilcoxon > 1.96:
        anwer_for_homogen_indep += f'Вілкоксон = {round(answer1_wilcoxon,3)} (u) > {1.96} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'Вілкоксон = {round(answer1_wilcoxon,3)} (u) <= {1.96} (Однорідні)\n'

    answer1_mannaWhitney = abs(HomogeneityIndependence.mannaWhitney(myArraysHomogenIndep))
    if answer1_mannaWhitney > 1.96:
        anwer_for_homogen_indep += f'Манна–Уїтні = {round(answer1_mannaWhitney,3)} (u) > {1.96} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'Манна–Уїтні = {round(answer1_mannaWhitney,3)} (u) <= {1.96} (Однорідні)\n'
    
    answer1_diffMeanRanks = abs(HomogeneityIndependence.diffMeanRanks(myArraysHomogenIndep))
    if answer1_diffMeanRanks > 1.96:
        anwer_for_homogen_indep += f'Різниці середніх рангів = {round(answer1_diffMeanRanks,3)} (u) > {1.96} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'Різниці середніх рангів = {round(answer1_diffMeanRanks,3)} (u) <= {1.96} (Однорідні)\n'

    answer1_kolmogorovSmirnov = HomogeneityIndependence.kolmogorovSmirnov(myArraysHomogenIndep)
    if answer1_kolmogorovSmirnov > alfa_for_qwant:
        anwer_for_homogen_indep += f'Колмогоров-Смірнов = {round(answer1_kolmogorovSmirnov,3)} (alf) > {alfa_for_qwant} (Однорідні)\n'
    else:
        anwer_for_homogen_indep += f'Колмогоров-Смірнов = {round(answer1_kolmogorovSmirnov,3)} (alf) <= {alfa_for_qwant} (Неоднорідні)\n'
    
    answer1_Bartlett = abs(HomogeneityIndependence.Bartlett(myArraysHomogenIndep))
    qw_6 = (len(choppedARRAYS)-1)
    if answer1_Bartlett > 5.99:
        anwer_for_homogen_indep += f'Бартлетта = {round(answer1_Bartlett,3)} (hi) > {5.99} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'Бартлетта = {round(answer1_Bartlett,3)} (hi) <= {5.99} (Однорідні)\n'

    answer1_UnivariateVarianceAnalysis = HomogeneityIndependence.UnivariateVarianceAnalysis(myArraysHomogenIndep)
    elements_in_each_row = [len(row) for row in choppedARRAYS]
    total_elements = sum(elements_in_each_row)
    qw_4_1 = (len(choppedARRAYS)-1)
    qw_4_2 = (total_elements-len(choppedARRAYS))
    if answer1_UnivariateVarianceAnalysis > 3.08:
        anwer_for_homogen_indep += f'Однофакторний дисперсійний аналіз = {round(answer1_UnivariateVarianceAnalysis,3)} (f) > {3.08} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'Однофакторний дисперсійний аналіз = {round(answer1_UnivariateVarianceAnalysis,3)} (f) <= {3.08} (Однорідні)\n'

    answer1_Hcriterion = abs(HomogeneityIndependence.Hcriterion(myArraysHomogenIndep))
    if answer1_Hcriterion > 5.99:
        anwer_for_homogen_indep += f'H-критерій  = {round(answer1_Hcriterion,3)} (hi) > {5.99} (Неоднорідні)\n'
    else:
        anwer_for_homogen_indep += f'H-критерій  = {round(answer1_Hcriterion,3)} (hi) <= {5.99} (Однорідні)\n'

    if isEq:
        anwer_for_homogen_indep += "Для залежних: \n"

    answer1_coincidenceMeanDep = abs(HomogeneityIndependence.coincidenceMeanDep(myArraysHomogenIndep))
    if answer1_coincidenceMeanDep > 1.96 and isEq:
        anwer_for_homogen_indep += f'Збіг середніх = {round(answer1_coincidenceMeanDep,3)} (t) > {1.96} (Неоднорідні)\n'
    elif answer1_coincidenceMeanDep <= 1.96 and isEq:
        anwer_for_homogen_indep += f'Збіг середніх = {round(answer1_coincidenceMeanDep,3)} (t) <= {1.96} (Однорідні)\n'

    answer1_singsSureDepButHomogeneity = HomogeneityIndependence.singsSureDepButHomogeneity(myArraysHomogenIndep)
    if answer1_singsSureDepButHomogeneity > 1.96 and isEq:
        anwer_for_homogen_indep += f'Знаки = {round(answer1_singsSureDepButHomogeneity,3)} (u) > {1.96} (Неоднорідні)\n'
    elif answer1_singsSureDepButHomogeneity <= 1.96 and isEq:
        anwer_for_homogen_indep += f'Знаки = {round(answer1_singsSureDepButHomogeneity,3)} (u) <= {1.96} (Однорідні)\n'

    answer1_AbbeIndep = abs(HomogeneityIndependence.AbbeIndep(myArraysHomogenIndep))
    if answer1_AbbeIndep > 1.96 and isEq:
        anwer_for_homogen_indep += f'Аббе = {round(answer1_AbbeIndep,3)} (u) > {1.96} (Неоднорідні)\n'
    elif answer1_AbbeIndep <= 1.96 and isEq:
        anwer_for_homogen_indep += f'Аббе = {round(answer1_AbbeIndep,3)} (u) <= {1.96} (Однорідні)\n'

    answer1_Qcriterion = HomogeneityIndependence.Qcriterion(myArraysHomogenIndep)
    if answer1_Qcriterion > chi2.ppf(1 - alfa_for_qwant, qw_6) and isEq:
        anwer_for_homogen_indep += f'Q-критерій  = {round(answer1_Qcriterion,3)} (hi) > {5.99} (Неоднорідні)\n'
    elif answer1_Qcriterion <= 3.84 and isEq:
        anwer_for_homogen_indep += f'Q-критерій  = {round(answer1_Qcriterion,3)} (hi) <= {5.99} (Однорідні)\n'

    #--------------------------------------------------------------------------------------------------------
        
    alpha = 0.05


    global ForMyCorrelationalString    
    ForMyCorrelationalString = CorrelationalRegression(calculation_data.array, choppedARRAYS)

    corr_x_average,corr_x_sq,corr_y_average,corr_y_sq = CorrelationalRegression.AverageSquareForXY(ForMyCorrelationalString)

    anwer_for_correlational_regression = ""

    anwer_for_correlational_regression += f'Середнє арифметичне X = {round(corr_x_average,3)}\n'
    anwer_for_correlational_regression += f'Середнє квадратичне X = {round(corr_x_sq,3)}\n'
    anwer_for_correlational_regression += f'Середнє арифметичне Y = {round(corr_y_average,3)}\n'
    anwer_for_correlational_regression += f'Середнє квадратичне Y = {round(corr_y_sq,3)}\n'

    z_list_for_graph = CorrelationalRegression.z_for_graph_norm(ForMyCorrelationalString)

    anwer_for_correlational_regression +="\n"
    Hi_square_2d_num = CorrelationalRegression.Hi_square_2d(ForMyCorrelationalString)
    if Hi_square_2d_num > chi2.ppf(1 - alfa_for_qwant, qw_6):
        anwer_for_correlational_regression += f'Оцінка адекватності відтворення = {round(Hi_square_2d_num,3)} (hi) > {round(chi2.ppf(1 - alfa_for_qwant, qw_6),3)} (Неадекватна)\n'
    elif Hi_square_2d_num <= chi2.ppf(1 - alfa_for_qwant, qw_6):
        anwer_for_correlational_regression += f'Оцінка адекватності відтворення  = {round(Hi_square_2d_num,3)} (hi) <= {round(chi2.ppf(1 - alfa_for_qwant, qw_6),3)} (Адекватна)\n'


    anwer_for_correlational_regression +="\n"
    global corr_coef
    corr_coef, corr_x_sq, corr_y_sq = CorrelationalRegression.correlation_coeff(ForMyCorrelationalString)
    anwer_for_correlational_regression += f'Коефіцієнт кореляції = {round(corr_coef,3)} \n'

    corr_coeff_check_num = CorrelationalRegression.corr_coeff_check(corr_coef, N)
    r_up, r_down = CorrelationalRegression.SKV_corr_coeff(corr_coef, N, 1.96)

    if corr_coeff_check_num > t.ppf(1 - alpha, qw_2_1):
        anwer_for_correlational_regression += f'Перевірки значущості коефіцієнта кореліяції = {round(corr_coeff_check_num,3)} (t) > {round(t.ppf(1 - alpha, qw_2_1),3)} (Значуща)\n'
        anwer_for_correlational_regression += f'Верхній довірчий = {round(r_up,3)}\n'
        anwer_for_correlational_regression += f'Нижній довірчий = {round(r_down,3)}\n'
    else:
        anwer_for_correlational_regression += f'Перевірки значущості коефіцієнта кореліяції = {round(corr_coeff_check_num,3)} (t) <= {round(t.ppf(1 - alpha, qw_2_1),3)} (Не значуща)\n'

    global rell_corr
    rell_corr, t_rell_corr = CorrelationalRegression.relation_correlation(ForMyCorrelationalString)

    anwer_for_correlational_regression +="\n"

    anwer_for_correlational_regression += f'Кореляційне відношення = {round(rell_corr,3)} \n'
    
    if t_rell_corr > t.ppf(1 - alpha, qw_2_1):
        anwer_for_correlational_regression += f'Перевірки значущості кореляційного відношщення = {round(t_rell_corr,3)} (t) > {round(t.ppf(1 - alpha, qw_2_1),3)} (Значуща)\n'
    else:
        anwer_for_correlational_regression += f'Перевірки значущості кореляційного відношщення = {round(t_rell_corr,3)} (t) <= {round(t.ppf(1 - alpha, qw_2_1),3)} (Не значуща)\n'

    spirmen_corr, spirmen_t = CorrelationalRegression.coefficient_Spearman(ForMyCorrelationalString)

    anwer_for_correlational_regression +="\n"

    anwer_for_correlational_regression += f'Кореляція Спірмена = {round(spirmen_corr,3)} \n'

    if spirmen_t > t.ppf(1 - alpha, qw_2_1):
       anwer_for_correlational_regression += f'Перевірки значущості для кореляції Спірмена = {round(spirmen_t,3)} (t) > {round(t.ppf(1 - alpha, qw_2_1),3)} (Значуща)\n'
    else:
        anwer_for_correlational_regression += f'Перевірки значущості для кореляції Спірмена = {round(spirmen_t,3)} (t) <= {round(t.ppf(1 - alpha, qw_2_1),3)} (Не значуща)\n'

    spirmen_up, spirmen_down = CorrelationalRegression.skv_spirmen(spirmen_corr, 1.96, N)

    anwer_for_correlational_regression += f'Верхній довірчий для Спірмена {round(spirmen_up,3)}\n'
    anwer_for_correlational_regression += f'Нижній довірчий для Спірмена {round(spirmen_down,3)}\n'

    anwer_for_correlational_regression += f'\n'

    kendel_corr = 2/3 * spirmen_corr

    anwer_for_correlational_regression += f'Кореляція Кендалла = {round(kendel_corr,3)}\n'

    u_for_kendel = 3*kendel_corr/math.sqrt(2*(2*N+5))*math.sqrt(N*(N-1))

    if abs(u_for_kendel) > t.ppf(1 - alpha, qw_2_1):
       anwer_for_correlational_regression += f'Перевірки значущості для кореляції Кендалла = {abs(round(u_for_kendel,3))} (t) > {round(t.ppf(1 - alpha, qw_2_1),3)} (Значуща)\n'
    else:
        anwer_for_correlational_regression += f'Перевірки значущості для кореляції Кендалла = {abs(round(u_for_kendel,3))} (t) <= {round(t.ppf(1 - alpha, qw_2_1),3)} (Не значуща)\n'


    sigma_tau_k = math.sqrt((4*N+10)/(9*(N**2-N)))

    kendel_up = kendel_corr + 1.96 * sigma_tau_k
    kendel_down = kendel_corr - 1.96 * sigma_tau_k

    anwer_for_correlational_regression += f'Верхній довірчий для Кендалла {round(kendel_up,3)}\n'
    anwer_for_correlational_regression += f'Нижній довірчий для Кендалла {round(kendel_down,3)}\n'

    anwer_for_correlational_regression += f'\n'

    count_points_in_quadrants, n00, n01, n10, n11 = CorrelationalRegression.count_points_in_quadrants_matrix(ForMyCorrelationalString)

    anwer_for_correlational_regression += f'Таблиця сполучень: \n'
    for row in count_points_in_quadrants:
        anwer_for_correlational_regression += f'{row}\n'

    n0 = n00+n01
    n1 = n11+n10
    m0 = n00+n10
    m1 = n01+n11

    anwer_for_correlational_regression += f'\n'

    fechner_i = (n00+n11-n10-n01)/(n00+n11+n10+n01)
    if fechner_i > 0:
        anwer_for_correlational_regression += f'Індекс фехнера {round(fechner_i,3)} (додатня кореляція)\n'
    elif fechner_i < 0:
        anwer_for_correlational_regression += f'Індекс фехнера {round(fechner_i,3)} (відємна кореляція)\n'
    else:
        anwer_for_correlational_regression += f'Індекс фехнера {round(fechner_i,3)} (звязку немає, отже незалежні)\n'

    anwer_for_correlational_regression += f'\n'

    hi_sq_corr = N * ((n00*n11-n01*n10-0.5)**2)/(n0*n1*m0*m1)

    if hi_sq_corr >= chi2.ppf(1 - alfa_for_qwant, qw_6):
       anwer_for_correlational_regression += f'Коефіцієнт сполучень Фі = {round(hi_sq_corr,3)} (hi) >= {round(chi2.ppf(1 - alfa_for_qwant, qw_6),3)} (Значуща)\n'
    else:
        anwer_for_correlational_regression += f'Коефіцієнт сполучень Фі = {round(hi_sq_corr,3)} (hi) < {round(chi2.ppf(1 - alfa_for_qwant, qw_6),3)} (Не значуща)\n'

    anwer_for_correlational_regression += f'\n'

    q_yula = (n00*n11-n01*n10)/(n00*n11+n01*n10)

    y_yula = (math.sqrt(n00*n11)-math.sqrt(n01*n10))/(math.sqrt(n00*n11)+math.sqrt(n01*n10))

    anwer_for_correlational_regression += f'Коефіцієнт звязку Юла Q = {round(q_yula,3)}\n'
    anwer_for_correlational_regression += f'Коефіцієнт звязку Юла Y = {round(y_yula,3)}\n'

    if n00 != 0 and n10 != 0 and n01 != 0 and n11 != 0:
        s_q_yula = 0.5*(1-q_yula**2)*math.sqrt(1/n00+1/n01+1/n10+1/n11)
        s_y_yula = 0.25*(1-y_yula**2)*math.sqrt(1/n00+1/n01+1/n10+1/n11)

        u_s_q_yula = abs(q_yula/s_q_yula)
        u_s_y_yula = abs(y_yula/s_y_yula)
        
        if u_s_q_yula >= norm.ppf(1 - alpha):
            anwer_for_correlational_regression += f'Коефіцієнт звязку Юла Q, значущість = {round(u_s_q_yula,3)} (u) > {round(norm.ppf(1 - alpha),3)} (Значуща)\n'
        else:
            anwer_for_correlational_regression += f'Коефіцієнт звязку Юла Q, значущість = {round(u_s_q_yula,3)} (u) <= {round(norm.ppf(1 - alpha),3)} (Не значуща)\n'

        if u_s_y_yula >= norm.ppf(1 - alpha):
            anwer_for_correlational_regression += f'Коефіцієнт звязку Юла Y, значущість = {round(u_s_y_yula,3)} (u) > {round(norm.ppf(1 - alpha),3)} (Значуща)\n'
        else:
            anwer_for_correlational_regression += f'Коефіцієнт звязку Юла Y, значущість = {round(u_s_y_yula,3)} (u) <= {round(norm.ppf(1 - alpha),3)} (Не значуща)\n'
    else:
        anwer_for_correlational_regression += f'Присутні аномальні значення, не можна скористатися таблицею 2x2\n'
        
    
    win(lines, listBoxCoef)



def open_file_t_test():
    global lines, listBoxCoef, filename
    filename = fd.askopenfilename()
    with open(filename) as f:
        t_test_list = [line.rstrip() for line in f]
    for i in range(len(t_test_list)):
        t_test_list[i] = float(t_test_list[i])
    
    N_elem = int(t_test_list[0])
    del t_test_list[0]
    N_t_nums = len(t_test_list)

    x_average_t = (sum(t_test_list)/N_t_nums)
    x_sq_avg_t = math.sqrt(sum([math.pow(x - x_average_t, 2) for x in t_test_list]) / N_t_nums)

    t_test_nums.insert(tk.END, f"{N_t_nums}                            /                           {N_elem}                           /                  {round_message_box(x_average_t)}             /               {round_message_box(x_sq_avg_t)}")


def open_file_3():
    global lines, listBoxCoef, filename, list_class_borders, ni_original, temp_0001, temp_0002,temp_0003,temp_0004, start_column, end_column
    temp_0001 = []
    temp_0002 = []
    temp_0003 = []
    temp_0004 = []
    ni_original = []
    list_class_borders = []
    filename = fd.askopenfilename()
    flag = True
    with open(filename) as file_object:
        for line in file_object.readlines():
            data = list(map(float, line.strip().split()))
            if flag:
                for _ in data:
                    ARRAYS.append([])
                flag = False
            for i in range(len(data)):
                ARRAYS[-len(data) + i].append(data[i])
    if len(lines) == 0:
        lines = ARRAYS[0][:]
    if (calculation_data.array is None or len(calculation_data.array[0]) == 0) and len(ARRAYS) > 1:
        if calculation_data.array is None:
            calculation_data.array = [[], []]
        calculation_data.array[0] = ARRAYS[0][:]
        calculation_data.array[1] = ARRAYS[1][:]
    if len(calculation_data.array[0]) == 0:
        calculation_data.array = None
    
    if start_column == -1 and end_column == -1:
        start_column = 0
        end_column = len(ARRAYS) - 1

    #click_graph_2()
    main()
    win(lines, listBoxCoef)


if __name__ == '__main__':
    win(None, None)
