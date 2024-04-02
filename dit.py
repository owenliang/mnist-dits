from torch import nn 
import torch 
from time_emb import TimeEmbedding
from dit_block import DiTBlock
from config import T 

class DiT(nn.Module):
    def __init__(self,img_size,patch_size,channel,emb_size,label_num,dit_num,head):
        super().__init__()
        
        self.patch_size=patch_size
        self.patch_count=img_size//self.patch_size
        self.channel=channel
        
        # patchify
        self.conv=nn.Conv2d(in_channels=channel,out_channels=channel*patch_size**2,kernel_size=patch_size,padding=0,stride=patch_size) 
        self.patch_emb=nn.Linear(in_features=channel*patch_size**2,out_features=emb_size) 
        self.patch_pos_emb=nn.Parameter(torch.rand(1,self.patch_count**2,emb_size))
        
        # time emb
        self.time_emb=nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size,emb_size)
        )

        # label emb
        self.label_emb=nn.Embedding(num_embeddings=label_num,embedding_dim=emb_size)
        
        # DiT Blocks
        self.dits=nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(emb_size,head))
        
        # layer norm
        self.ln=nn.LayerNorm(emb_size)
        
        # linear back to patch
        self.linear=nn.Linear(emb_size,channel*patch_size**2)
        
    def forward(self,x,t,y): # x:(batch,channel,height,width)   t:(batch,)  y:(batch,)
        # label emb
        y_emb=self.label_emb(y) #   (batch,emb_size)
        # time emb
        t_emb=self.time_emb(t)  #   (batch,emb_size)
        
        # condition emb
        cond=y_emb+t_emb
        
        # patch emb
        x=self.conv(x)  # (batch,new_channel,patch_count,patch_count)
        x=x.permute(0,2,3,1)    # (batch,patch_count,patch_count,new_channel)
        x=x.view(x.size(0),self.patch_count*self.patch_count,x.size(3)) # (batch,patch_count**2,new_channel)
        
        x=self.patch_emb(x) # (batch,patch_count**2,emb_size)
        x=x+self.patch_pos_emb # (batch,patch_count**2,emb_size)
        
        # dit blocks
        for dit in self.dits:
            x=dit(x,cond)
        
        # # layer norm
        x=self.ln(x)    #   (batch,patch_count**2,emb_size)
        
        # # linear back to patch
        x=self.linear(x)    # (batch,patch_count**2,channel*patch_size*patch_size)
        
        # reshape
        x=x.view(x.size(0),self.patch_count,self.patch_count,self.channel,self.patch_size,self.patch_size)  # (batch,patch_count,patch_count,channel,patch_size,patch_size)
        x=x.permute(0,3,1,2,4,5)    # (batch,channel,patch_count(H),patch_count(W),patch_size(H),patch_size(W))
        x=x.permute(0,1,2,4,3,5)    # (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x=x.reshape(x.size(0),self.channel,self.patch_count*self.patch_size,self.patch_count*self.patch_size)   # (batch,channel,img_size,img_size)
        return x
    
if __name__=='__main__':
    dit=DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4)
    x=torch.rand(5,1,28,28)
    t=torch.randint(0,T,(5,))
    y=torch.randint(0,10,(5,))
    outputs=dit(x,t,y)
    print(outputs.shape)