from turtle import end_fill
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv", sep=",", index_col=False)
df.columns = ["MinTemp", "MaxTemp"]

def gradient_descent(m_now,b_now,points,L):
    m_gradient = 0
    b_gradient = 0
    n= len(points)

    for i in range(n):
        x= points.iloc[i]['MinTemp']
        y= points.iloc[i]['MaxTemp']

        m_gradient += (-2/n) * x * (y-(m_now * x * b_now))
        b_gradient += (-2/n) * (y-(m_now * x * b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m,b

m = 0
b = 0
L = 0.0001
epochs = 100

for i in range(epochs):
    if i % 10 == 0:
        print(f"Epoch: {i}")
    m,b = gradient_descent(m,b,df,L)

print(m,b)
min = df['MinTemp'].min()
max = df['MaxTemp'].max()
plt.scatter(df['MinTemp'], df['MaxTemp'], color = "black")
plt.plot(list(range(min,max)), [m * x + b for x in range(min,max)], color = "red")
plt.show()