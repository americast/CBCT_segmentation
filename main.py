import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0,8.0,9.0]], requires_grad=True)
min_ = torch.ones(x.size(), requires_grad=True)
max_ = torch.ones(x.size(), requires_grad=True)

min_ = min_ * 3
max_ = max_ * 5

# From 3 to 5 (excluding 5)

# This enables thresholding to be different for different areas of the image

o = torch.ones(3, 3)
z = torch.zeros(3, 3)

x_min = x - min_
x_min = torch.where(x_min>0, o, z)

x_max = x - max_
x_max = torch.where(x_max<0, o, z)


x_net = torch.reshape(x_min, (9,)) * torch.reshape(x_max, (9,))
x_net = torch.reshape(x_net, (3, 3))

x_net = torch.reshape(x_net, (9,))

x_out = torch.nn.Linear(9, 1)(x_net)
# print(x_out)

error = x_out

# print(error)
x_out.backward()

print(min_.grad)

