import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import linregress
import os

# Configuration
data_str = """
0.075778267
0.038491042
-0.016870137
-0.021516162
0.006056793
-0.008004569
0.007708111
-0.008533095
-0.008078464
0.028247509
0.022728773
0.040156257
0.069223473
0.061191723
0.033470001
0.022972935
-0.021588949
0.008227189
0.021741382
-0.004906486
0.008689619
-0.013451721
0.014366983
0.025002395
-0.021300645
0.046532799
-0.029657901
-0.033656653
0.001008615
0.015796106
0.014897175
0.009737491
0.034781971
0.002591606
0.037703335
0.06170228
0.056222544
0.021549395
0.034515362
0.020954857
0.000631695
0.032995972
0.009714475
-0.021063525
0.027089454
0.00554503
0.022739933
0.018978573
-0.019207423
-0.045990376
-0.033092835
-0.048776391
-0.029777154
-0.030909482
-0.021241594
-0.035466648
-0.008111129
-0.028467674
-0.016888173
-0.079227015
"""
title = "WorldNews Sentiment Score Trend"


os.makedirs('plots', exist_ok=True)
def process_data_and_plot(data_string):
    weighted_monthly_std_values = list(map(float, data_string.strip().split('\n')))
    X = np.arange(len(weighted_monthly_std_values)).reshape(-1, 1)
    y = np.array(weighted_monthly_std_values)
    model = LinearRegression()
    model.fit(X, y)
    trend_line = model.predict(X)
    slope, intercept, r_value, p_value, std_err = linregress(X.flatten(), y)
    

    plt.figure(figsize=(14, 8))
    plt.scatter(X, y, color='blue', label='Sentiment Score Values')
    plt.plot(X, trend_line, color='red', label='Trend Line')
    plt.title(title)
    plt.xlabel('Time (Months)')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.grid(True)

    stats_text = f'Slope: {slope:.5f}\nIntercept: {intercept:.5f}\nR-squared: {r_value ** 2:.5f}\nP-value: {p_value:.5f}\nStd Error: {std_err:.5f}'
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

    plt.savefig(os.path.join('plots', title + '.png'), bbox_inches='tight')
    plt.show()

    return slope, intercept, r_value ** 2, p_value, std_err

slope, intercept, r_squared, p_value, std_err = process_data_and_plot(data_str)

print(f"Name: ", title )
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")
print(f"P-value: {p_value}")
print(f"Standard Error: {std_err}")