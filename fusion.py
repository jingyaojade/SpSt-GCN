import numpy as np


import pickle

label = open('./data/npy_dataset/transformed/ntu-xview/eval_label.pkl', 'rb')
_, label, _ = np.array(pickle.load(label))

J = open('./fusion/ntu60-xsub-spatial/2001_J_score.pkl', 'rb')
J = list(pickle.load(J))

V = open('./fusion/ntu60-xsub-spatial/2001_V_score.pkl', 'rb')
V = list(pickle.load(V))

B = open('./fusion/ntu60-xsub-spatial/2001_B_score.pkl', 'rb')
B = list(pickle.load(B))

print(len(label), len(J), len(V), len(B))

right_num = total_num = right_num_5 = 0

for i in range(len(J)):
    l = label[i]
    r11 = J[i]
    r22 = V[i]
    r33 = B[i]
    r = r22 + r33  #+ r22 #
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1

acc = right_num / total_num
acc5 = right_num_5 / total_num
print("%.4f"%acc, "%.4f"%acc5)