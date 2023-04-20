import torch

#have potential bug that occurs when batch_size>1

class FALoss(torch.nn.Module):
    def __init__(self,subscale=0.0625*8):
        super(FALoss,self).__init__()
        self.subscale=int(1/subscale) #16

    def forward(self,feature1,feature2):  #feature2是分割的主特征，feature1是辅助特征
        # feature1 = torch.nn.Conv2d(3,3,1)(feature1),
        # feature1=torch.nn.BatchNorm2d(3)(feature1),  #添加了BN层
        # feature1=torch.nn.ReLU(inplace=True)(feature1)

        feature1=torch.nn.AvgPool2d(self.subscale)(feature1) #平均池化，采样
        feature2=torch.nn.AvgPool2d(self.subscale)(feature2)

        m_batchsize, C, height, width = feature1.size() #feature1=[B,C,H1,W1]
        #print(feature1.size())

        feature1 = feature1.view(m_batchsize, -1, width*height)  #[B,C,W1*H1]
        #print(feature1.size())
        L1norm=torch.norm(feature1,2,1,keepdim=True).repeat(1,C,1)   #[B,1,W1*H1]
        #print(L1norm.size())
        # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        feature1=torch.div(feature1,L1norm)
        #print(feature1.size())
        mat1 = torch.bmm(feature1.permute(0,2,1),feature1) #[B,W1*H1,W1*H1]    feature1矩阵 * feature1矩阵 = #[B,W1*H1,C]*[B,C,W1*H1]

        m_batchsize, C, height, width = feature2.size() #feature2=[B,C,H2,W2]
        feature2 = feature2.view(m_batchsize, -1, width*height)  #[B,C,W2*H2]
        L2norm=torch.norm(feature2,2,1,keepdim=True).repeat(1,C,1)
        # L2norm=torch.repeat_interleave(L2norm, repeats=C, dim=1)
        feature2=torch.div(feature2,L2norm)
        mat2 = torch.bmm(feature2.permute(0,2,1),feature2) #[B,W2*H2,W2*H2]    feature2矩阵 * feature2矩阵 = #[B,W2*H2,C]*[B,C,W2*H2]

        Lnorm=torch.norm(mat1-mat2,1) # mat1=[B,W1*H1,W1*H1] , mat2=#[B,W2*H2,W2*H2]

        return Lnorm/((height*width)**2)



if __name__ == "__main__":
    a = torch.rand(2, 1, 160, 160)
    b = torch.rand(2, 1, 160, 160)
    fa = FALoss()
    loss = fa(a,b)
    print(loss)