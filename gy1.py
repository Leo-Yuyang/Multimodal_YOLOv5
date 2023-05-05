import torch
import random

# torch.manual_seed(1)
# x = torch.randn([2, 3, 1, 1])
# y = torch.randperm(3)
# x_ = x[:,y]

x = torch.randn([2, 3, 1, 1])
sorted, indices = torch.sort(x, dim=1, descending=True)
b, rank = torch.sort(indices, dim=1)
x_c = torch.clone(x)


print(x)
len = x.size()[0]
for i in range(len):
    ranki = rank[i,:,:,:].squeeze()
    # print(ranki)
    x_c[i,:,:,:] = sorted[i,:,:,:].index_select(0,ranki)

print(x_c)

# print(c)


# data = torch.tensor([0.5,-0.7,1.46])
# a,indx1 = torch.sort(data)
# b,indx2 = torch.sort(indx1)
# c = a.index_select(0,indx2)
# print(a)
# print(indx1)
# print(b)
# print(indx2)
# print(c)




# x_abs = x.abs()
# sorted, indices = torch.sort(x_abs,dim=1,descending=True)
# new_x = y.gather(dim=1,index=indices)
#
# print(x)
# print(indices)
# print(y)
# print(new_x)
# print(x_abs)
# print(sorted, indices)



# print(new_x[:,:2,:,:])









