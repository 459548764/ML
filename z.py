
#
# import numpy as np
#
# G = np.array([[0,2,2,2,2],
#               [2,0,2,2,2],
#               [2,2,0,2,2],
#               [2,2,2,0,2],
#               [2,2,2,2,0]])
#
# G_inv = np.linalg.inv(G)
#
# data = np.array([[1.41/2,1.41/2],
#                  [1.45/2,1.39/2],
#                  [1.40/2,1.42/2],
#                  [1.50/2,1.30/2],
#                  [1.44/2,1.35/2]])
#
# data = data / np.linalg.norm(data, axis=1, keepdims=True)
#
# new_data = (0.1 * G_inv) @ data
#
# new_data_norm = new_data / np.linalg.norm(new_data, axis=1, keepdims=True)
#
# print(data)
#
# print(new_data_norm)


# import torch
# from sklearn.datasets import make_moons
# from torch.autograd import Variable
#
# batch_size = 200
# cluster = 20
# feat_dim = 2
# gamma = 0.01
# beta = 0.01
# lr = 0.01
#
# s = torch.rand(batch_size,cluster).double()
# s = Variable(s,requires_grad = True)
#
# c = torch.rand(cluster,feat_dim).double()
# c = Variable(c,requires_grad = True)
#
# moon_data, y = make_moons(n_samples=batch_size, noise=0.15, random_state=0)
#
# x_max = max(moon_data[:,0])
# x_min = min(moon_data[:,0])
#
# y_max = max(moon_data[:,1])
# y_min = min(moon_data[:,1])
#
# def get_euclid(x,y):
#     xsquare = torch.sum(x * x, dim=1, keepdim=True)
#     ysquare = torch.sum(y * y, dim=1, keepdim=True)
#     xyinner = x @ y.t()
#     return xsquare + ysquare.t() - 2*xyinner
#
#
# for epoch in range(1,1000000):
#     '''
#     data: [Batch * Dim]
#     dis_mat: [Batch * Cluster]
#     out: [Batch * Cluster]
#     '''
#     data = torch.from_numpy(moon_data).double()
#     out = torch.nn.functional.softmax(s, dim=1)
#
#     dis_mat = get_euclid(data, c)
#     ctr_mat = get_euclid(c,c)
#
#     loss_means = torch.sum(out * dis_mat)
#     loss_reg = torch.norm(out, p=2)
#     loss_div = torch.sum(ctr_mat)
#
#     loss = loss_means + gamma * loss_reg - beta * loss_div
#     if epoch % 10000 == 0:
#         print(epoch, loss)
#         print(c)
#     loss.backward()
#
#     s.data.sub_(lr * s.grad.data)
#     c.data.sub_(lr * c.grad.data)
#
#     s.grad.data.zero_()
#     c.grad.data.zero_()
#
#     for i in range(cluster):
#         if c[i][0] > x_max:
#             c[i][0] = x_max
#         elif c[i][0] < x_min:
#             c[i][0] = x_min
#
#     for i in range(cluster):
#         if c[i][1] > y_max:
#             c[i][1] = y_max
#         elif c[i][1] < y_min:
#             c[i][1] = y_min

# import numpy as np
# import random
#
# user = 10000
# item = 2000
# ratio = 0.4
#
# item_cls = 5
#
# key = np.random.randint(0,item_cls,user)
#
#
# def random_pack(idx,num):
#     idxs = list(range(0,idx))
#     random.shuffle(idxs)
#
#     num_item = idx//num
#     num_list = []
#
#     for i in range(num):
#         num_list.append(idxs[i * num_item: (i+1) * num_item])
#
#     return num_list
#
#
# def split_train_test(user,item,cls,ratio):
#     for i in range(user):
#         print(i)
#         like = cls[i]
#         dislike = cls[(i+1) % item_cls]
#
#         for j in range(len(item[like])):
#             toss = np.random.random()
#             if toss < ratio:
#                 info = str(i) + " " + str(item[like][j]) + " " + "1" + "\n"
#                 with open('pu_train3.txt', 'a+', errors='ignore') as f:
#                     f.write(info)
#             else:
#                 info = str(i) + " " + str(item[like][j]) + " " + "1" + "\n"
#                 with open('pu_test3.txt', 'a+', errors='ignore') as f:
#                     f.write(info)
#
#         for j in range(len(item[dislike])):
#             info = str(i) + " " + str(item[like][j]) + " " + "0" + "\n"
#             with open('pu_test3.txt', 'a+', errors='ignore') as f:
#                 f.write(info)
#
# item_list = random_pack(item,item_cls)
#
# print(item_list)
#
# split_train_test(user,item_list,key,ratio)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
