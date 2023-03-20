import torch
mix_num = 2
r1 = 0.3 
r2 = 1 - r1
w = (r1 - r2) * torch.rand(mix_num, mix_num) + r2
w = w / w.sum(dim=1, keepdim=True)
inv_w = torch.inverse(w)
hidden_states = torch.randn(128, 55, 768)
# result = torch.einsum('bi,aijk->abjk', w, hidden_states.view(64, 2, 55, 768)).view(128, 55, 768)
x = torch.clone(hidden_states)
z = torch.clone(hidden_states)

for idx in range(0, 128, mix_num):
    hidden_states[idx] = x[idx] * w[0][0] + x[idx+1] * w[0][1]
    hidden_states[idx+1] = x[idx] * w[1][0] + x[idx+1] * w[1][1]
    z[idx:idx+mix_num] = torch.einsum('bi,ijk->bjk', w, x[idx:idx+mix_num])
y = torch.clone(hidden_states)

for idx in range(0, 128, 2):
    hidden_states[idx] = y[idx] * inv_w[0][0] + y[idx+1] * inv_w[0][1]
    hidden_states[idx+1] = y[idx] * inv_w[1][0] + y[idx+1] * inv_w[1][1]
    z[idx:idx+mix_num] = torch.einsum('bi,ijk->bjk', inv_w, y[idx:idx+mix_num])


pass