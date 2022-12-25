
import matplotlib.pyplot as plt
import csv
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("datause.csv",sep=',', header=0, parse_dates=True)

print(df['DEXUSEU'])
x = df['DATE'].to_numpy()
y = df['DEXUSEU'].to_numpy()

plt.plot(x, y,label='sber')

plt.legend()
plt.show()

'''
x = []
y = []

with open('datause.csv' ,'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        x.append(row[0])
        y.append(row[1])


plt.plot(x, y)
    #, color = 'g', linestyle = 'dashed',
     #    marker = 'o' ,label = "Weather Data")

#plt.xticks(rotation = 25)
plt.xlabel('Dates')
plt.ylabel('Temperature(°C)')
plt.title('Weather Report', fontsize = 20)
plt.grid()
plt.legend()
plt.show()
'''