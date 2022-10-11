import torch
def loss_KLD(z_mean, z_logvar, z_mean_p=torch.tensor([[0.2581,-0.6142]]), z_logvar_p=torch.tensor([[-0.5881, -1.1092]])):
    ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()+1e-10))
    return ret
mean = torch.tensor([[0.2454,0.6071]])
logvar = torch.tensor([[-0.0015, 0.0084]])
print(loss_KLD(mean, logvar))