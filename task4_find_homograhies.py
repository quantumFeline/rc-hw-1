import numpy as np
import task2_3_projective_transformation as t2p

matrix = np.array([#[[420, 728], [339, 635],  [400, 386],      [0,0],  [397, 787]],
                   #[[515, 568], [430, 491],  [484, 239],      [0,0],  [486, 630]],
                   #[[828, 214], [725, 148],       [0,0], [743, 612],  [785, 300]],
                   [[413, 509], [332, 423],  [384, 168], [339, 912],  [390, 577]],
                   [[850, 386], [743, 315],   [809, 45], [770, 768],  [804, 457]],
                   [[530, 392], [445, 321],   [493, 58], [462, 778],  [503, 463]],
                   [[1070, 552],[947, 433], [1024, 169], [985, 956], [1006, 632]]])

transformer = t2p.Transformer("./images/")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[0]):
        corners = matrix[[i, j], :, :]
        print(corners)

        proj_matrix = transformer.find_homography(list(corners[0]), list(corners[1]))
        print(f"From {i} to {j}:")
        print(proj_matrix)