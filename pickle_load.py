import pickle

with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

# Access the extracted data
print(data)