import numpy as np
import matplotlib.pyplot as plt
import shutil


def translate_to_origin(matrix):

    ''' Translate a matrix so that its centroid lies on the origin (0,0,0)
    
    :param matrix: Input a matrix of X,Y,Z coordinates
    :type matrix: np.array
    
    :return: Matrix complete with translated coordinates 
    :rtype: np.array
    :return: Means calculated for each of the X,Y,Z columns in matrix
    :rtype: float
    '''
    
    # calculate means for each column in matrix
    matrix_means = np.mean(matrix, axis=1)
    matrix_translated = matrix.copy()

    # subtract mean from each value in column to translate matrix 
    for axis in range(len(matrix_means)):
        matrix_translated[axis] -= matrix_means[axis]

    return matrix_translated, matrix_means


def compute_covariance_matrix(A_translated, B_translated):

    '''Compute the covariance matrix for 2 input matrices
    
    :param A_translated: Translated matrix A (centroid on origin)
    :type A_translated: np.array
    :param B_translated: Translated matrix B (centroid on origin)
    :type B_translated: np.array 
    
    :return: Covariance matrix
    :rtype: np.array
    '''
    
    # covariance matrix = transpose of B multiplied by A
    H = np.matmul(B_translated.T, A_translated)

    return H 


def compute_optimal_rotation_matrix(H):

    '''Compute the optimal rotation matrix from the covariance matrix

    :param H: Covariance matrix
    :type H: np.array
    
    :return: Optimal rotation matrix (R)
    :rtype: np.array
    '''
    
    # rotation matrix calculated by first finding the SVD, which gives U, S and V as results
    U, S, V = np.linalg.svd(H) 
    # rotation matrix R is calculated by transpose of V x transpose of U 
    Vt = V.T
    R = np.matmul(Vt, U.T)
    
    return R


def apply_rotation(A_translated, R):

    '''Rotate A to align with B using the calculated optimal rotation matrix (R)
    
    :param A_translated: Translated matrix A
    :type A_translated: np.array
    "param R: Optimal rotation matrix
    :type R: np.array
    
    :return: Matrix of rotated X,Y,Z coordinates for A to align with matrix B
    :rtype: np.array
    '''

    # apply rotation to translated A matrix by multiplying A by the rotation matrix
    A_rotated = np.matmul(A_translated, R)

    return A_rotated


# 4 translate A back to where B originally was centered (add back averages to matrix columns)
def revert_translation(A_rotated, B_translated, A_means, B_means):

    '''To finalise mapping of A to B, translate both matrices back from origin to original centroid placement

    :param A_rotated: Use translated and rotated matrix of A coordinates
    :type A_rotated: np.array
    :param B_translated: Translated matrix B
    :type B_translated: np.array
    :param A_means: Mean X, Y and Z coordinates for matrix A
    :type A_means: float
    :param B_means: Mean X, Y and Z coordinates for matrix B
    :type B_means: float

    :return: Updated matrix for A, translated back to original centroid placement
    :rtype: np.array
    :return: Updated matrix for B, translated back to original centroid placement (identical to original matrix B)
    :rtype: np.array
    '''


    # for each of matrices A and B, revert the centroids back to their original position
    # do opposite of translate_to_origin() and add means of each column back to coordinate values

    A_reverted = A_rotated.copy()
    for axis in range(len(A_means)):
        A_reverted[axis] += A_means[axis]

    B_reverted = B_translated.copy()
    for axis in range(len(B_means)):
        B_reverted[axis] += B_means[axis]

    return A_reverted, B_reverted


def plot_to_compare(results_1, results_2, label_1, label_2, plot_name, file_name):

    '''Generate 3D plots to visualise results and confirm algorithm working as expected

    :param results_1: First matrix
    :type results_1: np.array
    :param results_2: Second matrix to compare to first
    :type results_2: np.array
    :param label_1: Annotation for first matrix
    :type label_1: str
    :param label_2: Annotation for second matrix
    :type label_2: str
    :param plot_name: Title for plot
    :type plot_name: str
    :param file_name: Name for image of plot to be saved as
    :type file_name: str

    :return: none
    :type: none
    
    '''

    # create figure 
    ax = plt.figure(figsize=(15,15)).add_subplot(projection='3d')

    # plot matrices for comparison 
    ax.plot(results_1[0], results_1[1], results_1[2], label=label_1, color='r')
    ax.plot(results_2[0], results_2[1], results_2[2], label=label_2, color='b')

    # set plot titles and labels
    ax.legend(loc='upper left', fontsize=20)
    ax.set_xlabel('X', fontsize=20)
    ax.set_ylabel('Y', fontsize=20)
    ax.set_zlabel('Z', fontsize=20)
    plt.title(plot_name, fontsize=20)

    # save plot to file & close 
    plt.savefig(file_name)
    plt.close()

    return