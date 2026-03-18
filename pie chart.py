import pandas as pd
import matplotlib.pyplot as plt
# Creating a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emily'],
'Age': [25, 30, 22, 35, 28],
'Salary': [50000, 60000, 45000, 70000, 55000]}
df = pd.DataFrame(data)
# Displaying the DataFrame
print("DataFrame:")
print(df)
# Plotting a bar chart for Salary
plt.figure(figsize=(8, 6))
plt.bar(df['Name'], df['Salary'], color='skyblue')
plt.title('Salary Distribution')
plt.xlabel('Name')
plt.ylabel('Salary')
plt.show()
# Plotting a pie chart for Age
plt.figure(figsize=(8, 8))
plt.pie(df['Age'], labels=df['Name'], autopct='%1.1f%%', startangle=90)
plt.title('Age Distribution')
plt.show()