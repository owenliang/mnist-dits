import torch 
from config import T

# 前向diffusion计算参数
betas=torch.linspace(0.0001,0.02,T) # (T,)
alphas=1-betas  # (T,)
alphas_cumprod=torch.cumprod(alphas,dim=-1) # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
alphas_cumprod_prev=torch.cat((torch.tensor([1.0]),alphas_cumprod[:-1]),dim=-1) # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....]
variance=(1-alphas)*(1-alphas_cumprod_prev)/(1-alphas_cumprod)  # denoise用的方差   (T,)

# 执行前向加噪
def forward_add_noise(x,t): # batch_x: (batch,channel,height,width), batch_t: (batch_size,)
    noise=torch.randn_like(x)   # 为每张图片生成第t步的高斯噪音   (batch,channel,height,width)
    batch_alphas_cumprod=alphas_cumprod[t].view(x.size(0),1,1,1) 
    x=torch.sqrt(batch_alphas_cumprod)*x+torch.sqrt(1-batch_alphas_cumprod)*noise # 基于公式直接生成第t步加噪后图片
    return x,noise

if __name__=='__main__':
    import matplotlib.pyplot as plt 
    from dataset import MNIST
    
    dataset=MNIST()
    
    x=torch.stack((dataset[0][0],dataset[1][0]),dim=0) # 2个图片拼batch, (2,1,48,48)

    # 原图
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(x[0].permute(1,2,0))
    plt.subplot(1,2,2)
    plt.imshow(x[1].permute(1,2,0))
    plt.show()

    # 随机时间步
    t=torch.randint(0,T,size=(x.size(0),))
    print('t:',t)
    
    # 加噪
    x=x*2-1 # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配
    x,noise=forward_add_noise(x,t)
    print('x:',x.size())
    print('noise:',noise.size())

    # 加噪图
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(((x[0]+1)/2).permute(1,2,0))   
    plt.subplot(1,2,2)
    plt.imshow(((x[0]+1)/2).permute(1,2,0))
    plt.show()