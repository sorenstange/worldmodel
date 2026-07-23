import torch
import torch.nn.functional as F

if __name__ == '__main__':
    logits = torch.tensor([[1.,2.,3.,4.,5.]])
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample() # [B]
    log_prob = dist.log_prob(action) # [B]
    print(action)
    print(log_prob)

