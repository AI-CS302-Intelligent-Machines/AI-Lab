""" Read through the reference carefully. Implement routines for learning the parameters of HMM given in section 7.
In section 8, “A not-so-simple example”, an interesting exercise is carried out. Perform a similar experiment
on “War and Peace” by Leo Tolstoy. """

# Book Link - https://www.gutenberg.org/ebooks/2600 (War and Peace by Leo Tolstoy)

# ----------------------------------------------------------------------------------------------------------------------
# Importing

import numpy as np
import re

# ----------------------------------------------------------------------------------------------------------------------
# Reading the Book and Preprocessing

# Reading the file
book = 'War_and_Peace.txt'
file = open(book, 'r', encoding='utf-8')
text = file.read()
file.close()

# Removing the punctuations and converting to lower case
text = re.sub(r'[^a-zA-Z]', " ", text)
text = " ".join(text.split()).lower()[:100000]

# Creating a dictionary of all the unique characters
dictionary = {}
for i in range(26):
    dictionary[chr(i + 97)] = i
dictionary[" "] = 26

# ----------------------------------------------------------------------------------------------------------------------
# Initialize the parameters

# Observed sequence
O = np.zeros(len(text), dtype=int)

for i in range(len(text)):
    O[i] = dictionary[text[i]]
# Initial state distribution
pi = np.array(([0.525483, 0.474517]))
# Observable sequence
B = np.array([[0.03735, 0.03408, 0.03455, 0.03828, 0.03782, 0.03922, 0.03688, 0.03408, 0.03875, 0.04062, 0.03735, 0.03968, 0.03548, 0.03735, 0.04062, 0.03595, 0.03641, 0.03408, 0.04062, 0.03548, 0.03922, 0.04062, 0.03455, 0.03595, 0.03408, 0.03408, 0.03688],
              [0.03909, 0.03537,  0.03537, 0.03909, 0.03583,  0.03630, 0.04048, 0.03537, 0.03816, 0.03909, 0.03490, 0.03723, 0.03537, 0.03909, 0.03397, 0.03397, 0.03816, 0.03676, 0.04048, 0.03443, 0.03537, 0.03955, 0.03816,  0.03723,  0.03769, 0.03955, 0.03397]])
# Transition matrix
A = np.array([[0.47468, 0.52532], [0.51656, 0.48344]])
# Set of possible observations
V = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '])
# Set of possible states, Q is hidden
# Number of observation symbols
M = len(V)
# Number of states in the model
N = len(A)
# Length of observation sequence
T = len(O)

# ----------------------------------------------------------------------------------------------------------------------
# Alpha Pass


def alpha_pass(A1, B1, pi1, O1):
    c1 = np.zeros([T, 1])
    alpha1 = np.zeros([T, N])
    c1[0][0] = 0
    for x in range(N):
        alpha1[0][x] = pi1[x] * B1[x][O1[0]]
        c1[0][0] = c1[0][0] + alpha1[0][x]
    c1[0][0] = 1/c1[0][0]
    for x in range(N):
        alpha1[0][x] = c1[0][0] * alpha1[0][x]

    for t in range(1, T):
        c1[t][0] = 0
        for x in range(N):
            alpha1[t][x] = 0
            for y in range(N):
                alpha1[t][x] = alpha1[t][x] + alpha1[t-1][y] * A1[y][x]
            alpha1[t][x] = alpha1[t][x] * B1[x][O1[t]]
            c1[t][0] = c1[t][0] + alpha1[t][x]
        c1[t][0] = 1/c1[t][0]
        for x in range(N):
            alpha1[t][x] = c1[t][0] * alpha1[t][x]
    return alpha1, c1


# ----------------------------------------------------------------------------------------------------------------------
# Beta Pass


def beta_pass(A1, B1, O1, c1):
    beta1 = np.zeros([T, N])
    for x in range(N):
        beta1[T-1][x] = c1[T-1][0]
    for t in range(T-2, -1, -1):
        for x in range(N):
            beta1[t][x] = 0
            for y in range(N):
                beta1[t][x] = beta1[t][x] + A1[x][y] * B1[y][O1[t + 1]] * beta1[t + 1][y]
            beta1[t][x] = c1[t][0] * beta1[t][x]
    return beta1


# ----------------------------------------------------------------------------------------------------------------------
# Compute Gamma(x,t) and Gamma(x,y,t)

def gamma_pass(alpha1, beta1, A1, B1, O1):
    gamma1 = np.zeros([T, N])
    di_gamma1 = np.zeros([T, N, N])
    for t in range(T-1):
        for x in range(N):
            gamma1[t][x] = 0
            for y in range(N):
                di_gamma1[t][x][y] = alpha1[t][x] * A1[x][y] * B1[y][O1[t + 1]] * beta1[t + 1][y]
                gamma1[t][x] = gamma1[t][x] + di_gamma1[t][x][y]
    for x in range(N):
        gamma1[T-1][x] = alpha1[T-1][x]
    return gamma1, di_gamma1


# ----------------------------------------------------------------------------------------------------------------------
# Re-estimate A, B, pi

def re_estimate(gamma1, di_gamma1, A1, B1, pi1):
    for x in range(N):
        pi1[x] = gamma1[0][x]
    for x in range(N):
        denominator = 0
        for t in range(T-1):
            denominator = denominator + gamma1[t][x]
        for y in range(N):
            numerator = 0
            for t in range(T-1):
                numerator = numerator + di_gamma1[t][x][y]
            A1[x][y] = numerator/denominator
    for x in range(N):
        denominator = 0
        for t in range(T):
            denominator = denominator + gamma1[t][x]
        for y in range(M):
            numerator = 0
            for t in range(T):
                if O[t] == y:
                    numerator = numerator + gamma1[t][x]
            B1[x][y] = numerator/denominator
    return A1, B1, pi1


# ----------------------------------------------------------------------------------------------------------------------
# Compute log[P(O|lambda)]

def log_prob(c1):
    logProb1 = 0
    for x in range(T):
        logProb1 = logProb1 + np.log(c1[x][0])
    logProb1 = -logProb1
    return logProb1


# ----------------------------------------------------------------------------------------------------------------------
# Values initially

oldLogProb = -10000000
print("A: \n", A)
print("B: \n", np.concatenate((V.reshape(1, M), B), axis=0).T)
print("pi: ", pi)
print("logProb: ", oldLogProb)


# ----------------------------------------------------------------------------------------------------------------------
# After first iteration

alpha, c = alpha_pass(A, B, pi, O)
beta = beta_pass(A, B, O, c)
gamma, di_gamma = gamma_pass(alpha, beta, A, B, O)
A, B, pi = re_estimate(gamma, di_gamma, A, B, pi)
logProb = log_prob(c)

print("A: \n", A)
print("B: \n", np.concatenate((V.reshape(1, M), np.round_(B, decimals=7)), axis=0).T)
print("pi: ", np.round_(pi, decimals=7))
print("logProb: ", logProb)


# ----------------------------------------------------------------------------------------------------------------------
# Output

maxIter = 100
for ite in range(maxIter):
    alpha, c = alpha_pass(A, B, pi, O)
    beta = beta_pass(A, B, O, c)
    gamma, di_gamma = gamma_pass(alpha, beta, A, B, O)
    A, B, pi = re_estimate(gamma, di_gamma, A, B, pi)
    logProb = log_prob(c)

print("A: \n", A)
print("B: \n", np.concatenate((V.reshape(1, M), np.round_(B, decimals=7)), axis=0).T)
print("pi: ", np.round_(pi, decimals=7))
print("logProb: ", logProb)
