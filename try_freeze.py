import torch
import torch.nn as nn
import random

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

x = torch.tensor([[1.0]]).cuda()
y_ans = torch.tensor([[2.0]]).cuda()

x_meta = torch.tensor([[2.0]]).cuda()
y_meta = torch.tensor([[4.0]]).cuda()

my_meta_weight, my_meta_bias = torch.tensor([[1.0]]).cuda(), torch.tensor([[1.0]]).cuda()


class meta_one_layer_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def meta_forward(self, x, meta_bias, meta_weight):
        return meta_weight * x + meta_bias


class Multiple_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiple_factor = nn.Embedding(num_embeddings=1, embedding_dim=1)
        self.inner_net = meta_one_layer_net()

    def forward(self, x, freeze_inner=False, freeze_outer=False, use_meta=False, meta_bias=None, meta_weight=None):
        if not freeze_inner:
            if use_meta:
                x = self.inner_net.meta_forward(x, meta_bias, meta_weight)
            else:
                x = self.inner_net(x)
        else:
            freeze_net = torch.jit.freeze(torch.jit.script(self.inner_net.eval()))
            x = freeze_net(x)
        if freeze_outer:
            return self.multiple_factor(torch.tensor(0).cuda()).data * x
        else:
            return self.multiple_factor(torch.tensor(0).cuda()) * x


model = Multiple_Model()
model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_function = nn.L1Loss()
print(model.multiple_factor.weight, model.inner_net.layer.weight, model.inner_net.layer.bias)
y_guess = model(x, True, False)
loss = loss_function(y_guess, y_ans)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(model.multiple_factor.weight, model.inner_net.layer.weight, model.inner_net.layer.bias)

y_2_guess = model(x, False, True)
loss = loss_function(y_2_guess, y_ans)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(model.multiple_factor.weight, model.inner_net.layer.weight, model.inner_net.layer.bias)

adapt_lr = 0.1
y_support = model(x, False, False)
now_params = {}
for name, param in model.named_parameters():
    now_params[name] = param
loss_support = loss_function(y_support, y_ans)
adapt_grad = torch.autograd.grad(loss_support, now_params.values(), create_graph=True, allow_unused=True)

adapt_params = {}
for (name, param), grad in zip(now_params.items(), adapt_grad):
    adapt_params[name] = param - grad * adapt_lr

print(adapt_params)
y_meta_guess = model(x_meta, False, False, True, adapt_params['inner_net.layer.bias'], adapt_params['inner_net.layer.weight'])
loss = loss_function(y_meta_guess, y_meta)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(model.multiple_factor.weight, model.inner_net.layer.weight, model.inner_net.layer.bias)