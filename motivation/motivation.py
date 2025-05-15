import pickle

with open('results_100.pkl', 'rb') as f:
    results, sigmas = pickle.load(f)

print(results)
print(sigmas)