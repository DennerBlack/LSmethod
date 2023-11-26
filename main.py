import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.stats import t

custom_preamble = r"""
\usepackage[utf8]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian,english]{babel}
"""

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = custom_preamble

data_len = 5
data_width = 4
all_data_num = data_len*data_width


data = np.array([np.arange(data_len) for i in range(data_width)]).flatten() * 2 + 1
data.sort()

np.random.seed(109)
synth_data_sin = {'X': data,
                  'Y': np.sin(data*np.random.uniform(0.3, 0.3))*8+np.random.normal(0, np.log(data_len)/2, all_data_num)}
#                                                 ^ частота(ширина) синусоиды    ^ добавление шумов  ^ величина шума
np.random.seed(100)
synth_data_linear = {'X': data,
                     'Y': data+np.random.normal(0, np.log(data_len), all_data_num)}
#                                         ^ добавление шумов  ^ величина шума


np.random.seed(105)
synth_data_expanse = {'X': data,
                      'Y': np.asarray([np.ones(all_data_num)[i]*2+np.abs(np.random.normal(0, (i+2)**2, 1)[0])/55 for i in range(len(data))])}
#                                                   ^ добавление шумов  ^ величина шума

def calculate_slope(x, y):
    mx = x - x.mean()
    my = y - y.mean()
    return sum(mx * my) / sum(mx**2)


def get_params(x, y):
    a = calculate_slope(x, y)
    b = y.mean() - a * x.mean()
    return a, b


def get_correlation_coeff(x, y):
    cov = sum((x-np.mean(x))*(y-np.mean(y)))/(len(x)-1)
    R = cov/np.sqrt(sum(np.power(x-np.mean(x), 2))*sum(np.power(y-np.mean(y), 2)))*(len(x)-1)
    return R


def vis(data, images_number, img_index, name):
    d = data
    x = d['X']
    y = d['Y']
    a, b = get_params(x, y)

    lin_reg = a*x + b

    R = get_correlation_coeff(x, y)
    dof_model = 3
    dof_diff = all_data_num-2-dof_model
    mx = np.mean(x)
    my = np.mean(y)
    ssdx = np.sum((x - mx)**2)
    sdpa = np.sum((x - mx)*(y - my))
    ssdy = np.sum((y - my)**2)
    ssdy_reg = np.sum((lin_reg - mx)**2)
    ssd_lin_real = np.sum((y - lin_reg)**2)
    s2 = ssd_lin_real/(all_data_num-2)
    alpha = 0.05
    s = np.sqrt(s2)
    t_diff = t.interval(1. - alpha, all_data_num-2)
    min_int = [lin_reg[i*data_width] + t_diff[0] * np.sqrt(1 + 1/all_data_num + (((x[i*data_width] - mx)**2)/ssdx))*s for i in range(data_len)]
    max_int = [lin_reg[i*data_width] + t_diff[1] * np.sqrt(1 + 1/all_data_num + (((x[i*data_width] - mx)**2)/ssdx))*s for i in range(data_len)]
    sum_err2 = np.sum((np.asarray([np.mean(lin_reg[i:data_width*(i+1)]) for i in range(data_len)]) - my)**2)
    ss_diff = (ssdy-ssdy_reg)/sum_err2

    dot_element_mx = Line2D([0], [0], marker='o', color='w', label=r'$\bar{X}$ = '+f'{mx:.3f}',markerfacecolor='black', markersize=6)
    dot_element_my = Line2D([0], [0], marker='o', color='w', label=r'$\bar{Y}$ = '+f'{my:.3f}',markerfacecolor='black', markersize=6)
    dot_element_ssdx = Line2D([0], [0], marker='o', color='w', label=r'$\sum\left(x_i-\bar{x}\right)^2$ = '+f'{ssdx:.3f}',markerfacecolor='black', markersize=6)
    dot_element_sdpa = Line2D([0], [0], marker='o', color='w', label=r'$\sum\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)$ = '+f'{sdpa:.3f}',markerfacecolor='black', markersize=6)
    dot_element_ssdy = Line2D([0], [0], marker='o', color='w', label=r'$\sum\left(y_i-\bar{y}\right)^2$ = '+f'{ssdy:.3f}',markerfacecolor='black', markersize=6)
    dot_element_lr_formula = Line2D([0], [0], marker='o', color='w', label=r'Общий вид линейной регрессии: $\hat{y}=b+ax+\varepsilon$',markerfacecolor='black', markersize=6)
    dot_element_coeffs = Line2D([0], [0], marker='o', color='w', label=r'a = '+f'{a:.3f}'+r'; b = '+f'{b:.3f}'+r'; $\varepsilon$ = 0',markerfacecolor='black', markersize=6)
    dot_element_ssdy_reg = Line2D([0], [0], marker='o', color='w', label=r'$\sum\left(\hat{y}-\bar{y}\right)^2$ = '+f'{ssdy_reg:.3f}',markerfacecolor='black', markersize=6)
    dot_element_s2 = Line2D([0], [0], marker='o', color='w', label=r'$s^2=\frac{\sum\left(y_i-\hat{y}\right)^2}{n-2}$ = '+f'{s2:.3f}',markerfacecolor='black', markersize=6)
    dot_element_s = Line2D([0], [0], marker='o', color='w', label=r'S = '+f'{s:.3f}',markerfacecolor='black', markersize=6)
    dot_element_alpha = Line2D([0], [0], marker='o', color='w', label=r'alpha = '+f'{alpha:.3f}',markerfacecolor='black', markersize=6)
    dot_elements_interval = (Line2D([0], [0], marker='o', color='w', label=f'При X = {x[i*data_width]}:\t{min_int[i]:.3f}'+r' $\leq\mathbf{Y}\leq$ '+f'{max_int[i]:.3f}',markerfacecolor='black', markersize=6) for i in range(data_len))
    if np.abs(R) >= 0.7:
        dot_element_r = Line2D([0], [0], marker='o', color='w', label=r'Коэффициент корреляции Пирсона R = '+f'{R:.3f}'+'\nЛинейная корреляция явно выражена',markerfacecolor='black', markersize=6)
    else:
        dot_element_r = Line2D([0], [0], marker='o', color='w', label=r'Коэффициент корреляции Пирсона R = '+f'{R:.3f}'+'\nСлабая или отсутствующая линейная корреляция',markerfacecolor='black', markersize=6)

    latex_table = r'\begin{tabular}{ c | c | c | c | c } & Таблица дисперсионного анализа \\\hline Источник & Число степеней свободы & Суммы квадратов & Средние квадраты \\\hline Остаток &'+f'{all_data_num-2}'+' & '+f'{ssdy-ssdy_reg:.3f}'+' & '+f'{s2:.3f}'+r' \\\hline Неадекватность &'+f'{dof_model}'+' & '+f'{ss_diff:.3f}'+' & '+f'{(ss_diff/dof_model):.3f}'+r' \\\hline  Чистая ошибка &'+f'{dof_diff}'+' & '+f'{ssdy-ssdy_reg-ss_diff:.3f}'+' & '+f'{(ssdy-ssdy_reg-ss_diff)/dof_diff:.3f}'+r' \end{tabular}'

    dot_element_table = [Line2D([0], [0], marker='', color='w', label=latex_table, markerfacecolor='black', markersize=0)]

    legend_elements = [dot_element_mx,
                       dot_element_my,
                       dot_element_ssdx,
                       dot_element_sdpa,
                       dot_element_ssdy,
                       dot_element_lr_formula,
                       dot_element_coeffs,
                       dot_element_ssdy_reg,
                       dot_element_s2,
                       dot_element_s,
                       dot_element_alpha,
                       *dot_elements_interval,
                       dot_element_r,
                       ]

    plt.rc('text', usetex=True)
    plot = plt.subplot(1, images_number, img_index)
    plt.title(name, fontsize=14)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(x, y)
    plt.plot(x, lin_reg, color='red', label="x")
    leg1 = plot.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.8), fontsize=14)
    leg2 = plot.legend(handles=dot_element_table, loc='center', bbox_to_anchor=(0.5, -1.7), fontsize=14)
    matplotlib.pyplot.gca().add_artist(leg1)
    matplotlib.pyplot.gca().add_artist(leg2)


vis(synth_data_linear, 3, 1, "Линейновозрастающий набор данных c наложенным Гауссовым шумом")
vis(synth_data_sin, 3, 2, "Синусоидный набор данных c наложенным Гауссовым шумом")
vis(synth_data_expanse, 3, 3, "Плоский набор данных c наложенным мультипликативным Гауссовым шумом")

plt.subplots_adjust(wspace=0.56, top=0.88, bottom=0.6, left=0.015, right=0.89)

plt.show()
