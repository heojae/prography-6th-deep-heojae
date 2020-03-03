# prography-6th-deep-heojae

우선 이것을 시작하기전에 이곳 [제 구글드라이브 모델저장소](https://drive.google.com/open?id=16QzPEIepI5gm5lmhgOVB7goNtTwO8RNX)에 들어가서   best_vgg_mnist.pt를 들고 와서   
test.py를 실행을 시켜주시면 됩니다. 


구글 드라이브에서 best_vgg_mnist.pt를 들고와서 test.py를 돌리면   
best accuracy 99.1이 나옵니다.   
Test set- best Accuracy: 9910/10000 (99%)



VGG-16으로 네트워크를 구성하고, 
=>model.py 에 VGG16을 구현을 해두었습니다.  

MNIST 데이터를 RGB 채널로 변경해주세요.
=> transform에서, 이렇게 구현을 해두어습니다.
transform = transforms.Compose([  
                transforms.Resize((224,224)),  
                transforms.ToTensor(),  
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),  
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  
                 ])  
change input size from [1,28,28] -> [1,224,224]  
change 1 dimension to 3 dimension [1,224,224] ->[3,224,224]
  
(1)의 모델 구조에서 model initialization, inference 부분을 함수형태로 작성해주세요. (객체 형태로 작성하여도 무관합니다)  
model initialization   
from model import VGG16        # i initialize model in model.py as VGG16  
----------------------------------------------------------------------------------------------------------
Inference   
inference.py 에서, image_name = "sample1.jpg"라고 되어 있는데, 이 이미지 이름만 바꾸어서 실행을 시키면 결과를 출력을 해줍니다.   

(2)의 구조에서 Conv2_1의 입력을 첫번째 Dense 입력에 추가해주는 구조를 추가해주세요. (Skip connection 구조)  
model.py에 VGG16 network 를 구현을 해두었으며,  

self.skip1 = nn.Conv2d(64,512,kernel_size=6,padding=1,stride=18) 을 통해서, conv2_1에서 온 input을 받아서,   

Dense layer 에 들어가기 전에 torch.cat(x,skip_connection)으로 짜두어 구현을 해두었습니다.   


(3)에서 나온 모델을 RGB체널로 바꾼 MNIST로 학습해주세요.

RGB 채널로 학습을 시켜두었습니다


python test.py을 통해 테스트 코드를 실행시켜 정확도를 출력해주세요.

구글 드라이브에서 best_vgg_mnist.pt를 들고와서 test.py를 돌리면 

best accuracy 99.1이 나옵니다. 

Test set- best Accuracy: 9910/10000 (99%)

구현한 모델의 ADT를 README.md에 간단히 요약해주세요.

기본 VGG net을 기반으로, 아래와 같이 추가를 해주어서 network를 구성을 하였습니다.

#######################################################################################
i think i need to torch.cat before to first Dense layer and

i need to change [1,64,112,112] -> [1,512,7,7] 

so i implement skip1 for this.

self.skip1 = nn.Conv2d(64,512,kernel_size=6,padding=1,stride=18) 

#######################################################################################  
이 skip1을 새로운 conv2_1의 input에 넣고, 나중에 Dense (fc6)에 들어가기전에 x = torch.cat((x, skip_connection), dim=1)으로 함께 들어가게 됩니다.



프레임워크(Tensorflow, Keras, Pytorch 등) 은 원하는 대로 선택해주시면 됩니다.

=> pytorch로 구현을 해두었습니다. 

torch, torchvision, PIL, argparser, numpy, matplotlib 정도 library가 있으면 돌아갈 수 있을 것 입니다. 












