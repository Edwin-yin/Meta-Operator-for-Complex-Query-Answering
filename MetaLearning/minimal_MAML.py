import torch
import torch.nn as nn


x_support = torch.tensor(1).cuda()
y_support = torch.tensor(9).cuda()
x_query = torch.tensor(2).cuda()
y_query = torch.tensor(18).cuda()


class Multiple_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiple_factor = nn.Embedding(num_embeddings=1, embedding_dim=1)

    def forward(self, x, params_now=None):
        if params_now:
            return (params_now['multiple_factor.weight'] ** 2) * x
        else:
            return (self.multiple_factor(torch.tensor(0).cuda())**2) * x


model = Multiple_Model()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.L1Loss()
adapt_lr = 1
print(model.multiple_factor.weight)
y_support_guess = model(x_support)
now_params = {}
for name, param in model.named_parameters():
    now_params[name] = param
loss_support = loss_function(y_support_guess, y_support)
adapt_grad = torch.autograd.grad(loss_support, now_params.values(), create_graph=True, allow_unused=True)
print(adapt_grad)
adapt_params = {}
for (name, param), grad in zip(now_params.items(), adapt_grad):
    adapt_params[name] = param - grad * adapt_lr
print(adapt_params)
y_query_guess = model(x_query, adapt_params)
loss_query = loss_function(y_query_guess, y_query)
optimizer.zero_grad()
loss_query.backward()
optimizer.step()
print(model.multiple_factor.weight)





