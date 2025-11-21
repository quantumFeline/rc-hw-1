import numpy as np
import task2_3_projective_transformation as t2p

corners_from = [(340, 638), (743, 633), (261, 695), (270, 793), (169, 800), (165, 696), (335, 421), (337, 615), (157, 623), (152, 428)]
corners_to = [(399, 788), (795, 797), (322, 847), (333, 937), (240, 941), (230, 850), (392, 585), (395, 765), (225, 763), (220, 581)]

print(t2p.Transformer.find_homography(corners_from, corners_to).tolist())