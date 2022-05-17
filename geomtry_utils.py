import numpy as np

def rot_ar_x(radi):
    return  np.array([[1, 0, 0],
            [0, np.cos(radi), -np.sin(radi)],
            [0, np.sin(radi), np.cos(radi)],],
            dtype=np.double)

def rot_ar_y(radi):
    return  np.array([[np.cos(radi), 0, np.sin(radi)],
            [0, 1, 0],
            [-np.sin(radi), 0, np.cos(radi)],],
            dtype=np.double)

def rot_ar_z(radi):
    return  np.array([[np.cos(radi), -np.sin(radi), 0],
            [np.sin(radi), np.cos(radi), 0],
            [0, 0, 1],],
            dtype=np.double)

def transform_ph4s(list_of_ph4Matrices, angleX=np.pi, angleY=np.pi, angleZ=np.pi, translateX=0, translateY=0,
                   translateZ=0):

    # transform fragment test coords by applying rotation matrices ?
    rot_matrix_x = rot_ar_x(angleX)
    rot_matrix_y = rot_ar_y(angleY)
    rot_matrix_z = rot_ar_y(angleZ)
    # rot_matrices = [rot_matrix_x, rot_matrix_y, rot_matrix_z]

    # frag_transformed = frag_test_mol_coords @ rot_matrix_x
    # frag_transformed_1 = frag_transformed @ rot_matrix_y
    # frag_transformed_2 = frag_transformed_1 @ rot_matrix_z

    _list_of_ph4Matrices=[]
    for mat in list_of_ph4Matrices:
        mat  = mat @ rot_matrix_x  @ rot_matrix_y @ rot_matrix_z
        mat += np.array([translateX,translateY,translateZ])
        _list_of_ph4Matrices.append(mat)
    return _list_of_ph4Matrices

def test_transform_ph4s():
    list_of_ph4Matrices = [np.array([[0,0,0], [5,0,0], [0,5,0], [0,0,5]]),]
                           # np.random.randn(3,3)*(-10)]
    _list_of_ph4Matrices = transform_ph4s(list_of_ph4Matrices, angleX=np.pi, angleY=0, angleZ=0, translateX=1)

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    labels = ['Donor'] #, 'Acceptor']
    for (coordsOri,coordsRot), label in zip(zip(list_of_ph4Matrices, _list_of_ph4Matrices), labels):
        print(coordsOri)
        print(coordsRot)
        ax.scatter3D(coordsOri[:, 0], coordsOri[:, 1], coordsOri[:, 2], label=label)
        ax.scatter3D(coordsRot[:, 0], coordsRot[:, 1], coordsRot[:, 2], label=label+"_rot")

    plt.legend()
    plt.show()