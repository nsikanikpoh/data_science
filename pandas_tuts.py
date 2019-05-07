import pandas as pd
# Read data from CSV file 
csv_path = 'TopSellingAlbums.csv' 
df = pd.read_csv(csv_path)
print(df.head())

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

xlsx_path = 'TopSellingAlbums.xlsx'

df = pd.read_excel(xlsx_path)
print(df.head())

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

x = df[['Length']]
print(x)

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
x = df['Length']
print(x)


# Get the column as a dataframe
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
x = type(df[['Artist']])
print(x)



# Access to multiple columns
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
y = df[['Artist','Length','Genre']]
print(y)


# Access the value on the first row and the first column
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
y = df.iloc[0, 0]
print(y)


 # Access the value on the second row and the first column
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
y = df.iloc[1,0]
print(y)



# Access the column using the name
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
y = df.loc[1, 'Artist']

print(y)


# Access the column using the name
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
y = df.loc[1, 'Released']

print(y)



#You can perform slicing using both the index and the name of the column:

# Slicing the dataframe
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
y = df.iloc[0:2, 0:3]

print(y)



# Slicing the dataframe using name
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
y = df.loc[0:2, 'Artist':'Released']

print(y)