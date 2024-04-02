import torch 
from config import T
from dit import DiT
import matplotlib.pyplot as plt 
from diffusion import *

DEVICE='cuda' if torch.cuda.is_available() else 'cpu' # 设备

def backward_denoise(model,x,y):
    steps=[x.clone(),]

    global alphas,alphas_cumprod,variance

    x=x.to(DEVICE)
    alphas=alphas.to(DEVICE)
    alphas_cumprod=alphas_cumprod.to(DEVICE)
    variance=variance.to(DEVICE)
    y=y.to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        for time in range(T-1,-1,-1):
            t=torch.full((x.size(0),),time).to(DEVICE) 

            # 预测x_t时刻的噪音
            noise=model(x,t,y)    
            
            # 生成t-1时刻的图像
            shape=(x.size(0),1,1,1) 
            mean=1/torch.sqrt(alphas[t].view(*shape))*  \
                (
                    x-
                    (1-alphas[t].view(*shape))/torch.sqrt(1-alphas_cumprod[t].view(*shape))*noise
                )
            if time!=0:
                x=mean+ \
                    torch.randn_like(x)* \
                    torch.sqrt(variance[t].view(*shape))
            else:
                x=mean
            x=torch.clamp(x, -1.0, 1.0).detach()
            steps.append(x)
    return steps

model=DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4).to(DEVICE) # 模型
model.load_state_dict(torch.load('model.pth'))

# 生成噪音图
batch_size=10
x=torch.randn(size=(batch_size,1,28,28))  # (5,1,24,24)
y=torch.arange(start=0,end=10,dtype=torch.long)   # 
# 逐步去噪得到原图
steps=backward_denoise(model,x,y)
# 绘制数量
num_imgs=20
# 绘制还原过程
plt.figure(figsize=(15,15))
for b in range(batch_size):
    for i in range(0,num_imgs):
        idx=int(T/num_imgs)*(i+1)
        # 像素值还原到[0,1]
        final_img=(steps[idx][b].to('cpu')+1)/2
        # tensor转回PIL图
        final_img=final_img.permute(1,2,0)
        plt.subplot(batch_size,num_imgs,b*num_imgs+i+1)
        plt.imshow(final_img)
plt.show()