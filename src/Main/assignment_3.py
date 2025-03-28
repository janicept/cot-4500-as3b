#Question 1
import numpy as np

# Augmented matrix [A|b]
# Enter a different matrix here if desired:
A = np.array([[2, -1, 1, 6],
              [1, 3, 1, 0],
              [-1, 5, 4, -3]], dtype=float)

def gaussian_elimination(A):
    n = len(A)
    
    # Perform Gaussian elimination
    for i in range(n):
        if A[i, i] == 0.0:
            for j in range(i + 1, n):
                if A[j, i] != 0.0:
                    A[[i, j]] = A[[j, i]]  # Swap rows
                    break

        # Normalize the pivot row
        A[i] = A[i] / A[i, i]

        # Eliminate the current column below the pivot
        for j in range(i + 1, n):
            A[j] = A[j] - A[j, i] * A[i]

    return A

def backward_substitution(A):
    n = len(A)
    x = np.zeros(n)

    # Start backward substitution
    for i in range(n - 1, -1, -1):
        x[i] = A[i, -1] - np.sum(A[i, i+1:n] * x[i+1:n])
    
    return x

# Perform Gaussian elimination
A = gaussian_elimination(A)

# Perform backward substitution
solution = backward_substitution(A)
solution = np.round(solution).astype(int)

print("Solution:", solution)

#_________________________________
#Question 2
import numpy as np

# Given matrix A
# Enter a different matrix here if desired:
A = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]], dtype=float)

def lu_factorization(A):
    n = len(A)
    L = np.eye(n)  # Start with an identity matrix for L
    U = A.copy()   # Start with a copy of A for U

    # Perform LU decomposition using Gaussian elimination
    for i in range(n):
        for j in range(i + 1, n):
            if U[j, i] != 0:
                # Compute the multiplier
                L[j, i] = U[j, i] / U[i, i]
                # Eliminate the element in the matrix U
                U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]

    return L, U

def compute_determinant(U):
    # The determinant of A is the product of the diagonal elements of U
    return np.prod(np.diagonal(U))

# LU decomposition
L, U = lu_factorization(A)

# Compute determinant of A
determinant = compute_determinant(U)

print()
print(f"{determinant:.15f}")
print()
L = np.round(L).astype(int)
print(L)
print()
U = np.round(U).astype(int)
print(U)


#_______________________________
#Question 3
import numpy as np

# Define the matrix A
# Enter a different matrix here if desired:
A = np.array([[9, 0, 5, 2, 1],
              [3, 9, 1, 2, 1],
              [0, 1, 7, 2, 3],
              [4, 2, 3, 12, 2],
              [3, 2, 4, 0, 8]])

def is_diagonally_dominant(A):
    rows, cols = A.shape
    for i in range(rows):
        diagonal_element = abs(A[i, i])  # Diagonal element
        row_sum = np.sum(np.abs(A[i])) - diagonal_element  # Sum of non-diagonal elements
        if diagonal_element < row_sum:
            return False  # If condition fails, matrix is not diagonally dominant
    return True  # If all rows satisfy the condition

# Check if the matrix is diagonally dominant
result = is_diagonally_dominant(A)

print()
if result:
    print("The matrix is diagonally dominant.")
else:
    print("The matrix is NOT diagonally dominant.")


#___________________________
#Question 4
import numpy as np

# Define the matrix A
# Enter a different matrix here if desired:
A = np.array([[2, 2, 1],
              [2, 3, 0],
              [1, 0, 2]])

# Function to check if the matrix is positive definite
def is_positive_definite(A):
    # Compute the eigenvalues of A
    eigenvalues = np.linalg.eigvals(A)
    
    # Check if all eigenvalues are positive
    return np.all(eigenvalues > 0)

# Check if the matrix is positive definite
print()
if is_positive_definite(A):
    print("The matrix is a positive definite.")  
else:
    print("The matrix is not a positive definite.")  
