# MIBench.github.io
MIBench: A Comprehensive Benchmark for Membership Inference Attacks

MIBench is a comprehensive benchmark for comparing different MI attacks, which consists not only the evaluation metric module, but also the evaluation scenario module. And we design the evaluation scenarios from four perspectives: data distribution, the distance between members and nonmembers in the target dataset, the differential distance between two datasets (i.e., the target dataset and a generated dataset with only nonmembers), and the ratio of the samples that are made no inferences by an MI attack. The evaluation metric module consists of six typical evaluation metrics (e.g., precision, recall, f1-score, false positive rate (FPR), 
false negative rate (FNR), membership advantage (MA)). We have identified three principles for the proposed “comparing different MI attacks” methodology, and we have designed and implemented the MIBench benchmark with 84 evaluation scenarios for each dataset. In total, we have used our benchmark to fairly and systematically compare 13 state-of-the-art MI attack algorithms across 588 evaluation scenarios, and these evaluation scenarios cover 7 widely used datasets and 7 representative types of models.

MI attacks:
NN_attack,
Loss-Threshold,
Lable-only,
Top3-NN attack,
Top1-Threshold,
BlindMI-Diff,
Top2+True,
Privacy Risk Scores,
Shapley Values,
Positive Predictive Value ,
Calibrated Score,
Distillation-based Thre.,
Likelihood ratio attack

Datasets: CH_MNIST, CIFAR10, CIFAR100, ImageNet, Location30, Purchase100, ImageNet, Texas100

Models: MLP, StandDNN, VGG16, VGG19, ResNet50, ResNet101, DenseNet121


Requirements:
You can run the following script to configurate necessary environment
sh ./sh/install.sh

Usage:
   Please first to make a folder for record, all experiment results with save to record folder as default. And make folder for data to put supported datasets.
   XXX  XXX

Attack:
This is a demo script of running NN_attack on CIFAR100.
python ./attack/NN_attack.py --yaml_path ../config/attack/NN/CIFAR100.yaml --dataset CIFAR100 --dataset_path ../data --save_folder_name CIFAR100_0_1

Selected attacks:
![Selected attacks](https://user-images.githubusercontent.com/124696836/217288040-c38bb11e-6b8a-48ae-bc1b-1c7639e886c3.png)

Evaluation Framework:
    MIBench is a comprehensive benchmark for comparing different MI attacks, which consists not only the evaluation metric module, but also the evaluation scenario module.
Part I: Evaluation Scenarios
In this work, we have designed and implemented the MIBench benchmark with 84 evaluation scenarios for each dataset. In total, we have used our benchmark to fairly and systematically compare 13 state-of-the-art MI attack algorithms across 588 evaluation scenarios, and these evaluation scenarios cover 7 widely used datasets and 7 representative types of models.

(a) Evaluation Scenarios of CIFAR100.
![Evaluation Scenarios of CIFAR100_V1](https://user-images.githubusercontent.com/124696836/217288244-4e5e3a64-6649-457d-ae69-a8c8ad566322.png)


(b) Evaluation Scenarios of CIFAR10.
![Evaluation Scenarios of CIFAR10](https://user-images.githubusercontent.com/124696836/217288349-85c55eed-ae51-4721-a063-618b45a4eef4.png)

(c) Evaluation Scenarios of CH_MNIST.
![Evaluation Scenarios of CH_MNIST](https://user-images.githubusercontent.com/124696836/217288415-6eba7f41-7daf-4d9e-8932-8b1a76a3d057.png)

(d) Evaluation Scenarios of ImageNet.
![Evaluation Scenarios of ImageNet](https://user-images.githubusercontent.com/124696836/217289221-4b7301ff-e9f4-4014-8d72-36f2448fab87.png)


(e) Evaluation Scenarios of Location30.
![Evaluation Scenarios of Location30](https://user-images.githubusercontent.com/124696836/217289254-18671ab0-5352-4e97-a437-22da51d657b4.png)


(f) Evaluation Scenarios of Purchase100.
![Evaluation Scenarios of Purchase100](https://user-images.githubusercontent.com/124696836/217289288-09ce666a-8d8c-4fc2-b47b-f1d9e3555f6a.png)


(g)  Evaluation Scenarios of Texas100.
![Evaluation Scenarios of Texas100](https://user-images.githubusercontent.com/124696836/217289305-3d6e9a52-af89-457b-9b9f-ca6810ecc833.png)


Part II: Evaluation Metrics
We mainly use attacker-side precision, recall, f1-score, false positive rate (FPR), false negative rate (FNR), membership advantage (MA), as our evaluation metrics.  The details of the evaluation metrics are shown as follows.
(a) precision: the ratio of real-true members predicted among all the positive membership predictions made by an adversary; 
(b) recall: the ratio of true members predicted by an adversary among all the real-true members; 
(c) f1-score: the harmonic mean of precision and recall; 
(d) false positive rate (FPR): the ratio of nonmember samples are erroneously predicted as members; 
(e) false negative rate (FNR): the difference of the 1 and recall (e.g., FNR=1-recall);
(f) membership advantage (MA)： is the difference between the true positive rate and the false positive rate (e.g., MA = TPR - FPR).

Results:
(1) CIFAR100:
RQ1: Effect of data Distributions
ES01: CIFAR100_Normal + 2.893 + 0.085 + 20%
ES29: CIFAR100_Uniform + 2.893 + 0.085 + 20% 
ES57: CIFAR100_Bernoulli + 2.893 +0.085 + 20%
![CIFAR100_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/219857423-023f397c-8b77-4406-8f17-344609c83721.png)
   

RQ2: Effect of Distance between members and nonmembers
ES02: CIFAR100_Normal + 2.893 + 0.085 + 40% 
ES10: CIFAR100_Normal + 3.813 + 0.085 + 40%
ES22: CIFAR100_Normal + 4.325 + 0.085 + 40%
![CIFAR100_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/219857440-b9a22ebc-8f68-48b8-af90-57f390781d13.png)



RQ3: Effect of Differential Distances between two datasets
ES03: CIFAR100_Normal + 2.893 + 0.085 + 45%
ES05: CIFAR100_Normal + 2.893 + 0.119 + 45%
ES07: CIFAR100_Normal + 2.893 + 0.157 + 45%                     
![CIFAR100_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/219857466-c0857043-b7d5-4bfb-a1a2-709d26b5b7a1.png)


RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack 
ES37: CIFAR100_Uniform + 3.813 + 0.085 + 20%
ES38: CIFAR100_Uniform + 3.813 + 0.085 + 40%
ES39: CIFAR100_Uniform + 3.813 + 0.085 + 45% 
ES40: CIFAR100_Uniform + 3.813 + 0.085 + 49% 
![CIFAR100_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/219857475-56cccc31-1e56-4f75-8f6a-f60cdf096396.png)


(2) CIFAR10:
RQ1: Effect of data Distributions
ES13: CIFAR10_Normal + 2.501 + 0.213 + 20%
ES41: CIFAR10_Uniform + 2.501 + 0.213 + 20%
ES69: CIFAR10_Bernoulli + 2.501 + 0.213 + 20%
        
![CIFAR10_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/219933681-e0ab7da6-63e6-4d77-90c8-8bfa4e69c67c.png)
 

RQ2: Effect of Distance between members and nonmembers
ES02: CIFAR10_Normal + 1.908 + 0.155 + 40%
ES10: CIFAR10_Normal + 2.501 + 0.155 + 40%
ES22: CIFAR10_Normal + 3.472 + 0.155 + 40%
                    
![CIFAR10_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/219933690-c9b04621-9d44-43c5-8dbe-3e6474dcd781.png)


RQ3: Effect of Differential Distances between two datasets
ES51: CIFAR10_Uniform + 3.472 + 0.155 + 45%
ES53: CIFAR10_Uniform + 3.472 + 0.213 + 45% 
ES55: CIFAR10_Uniform + 3.472 + 0.291 + 45% 
                    
![CIFAR10_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/219933691-86b1ae51-f4b4-4c96-8428-fa8c1e83b5fd.png)


RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack 
ES57: CIFAR10_Bernoulli + 1.908 +0.155 + 20% 
ES58: CIFAR10_Bernoulli + 1.908 + 0.155 + 40%
ES59: CIFAR10_Bernoulli + 1.908 + 0.155 + 45%
ES60: CIFAR10_Bernoulli + 1.908 + 0.155 + 49% 
        
![CIFAR10_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/219933699-f10df496-f69b-47a2-8fa8-d31f4c0a2cfb.png)
   

(3) CH_MNIST:
RQ1: Effect of data Distributions
ES21: CH_MNIST_Normal + 1.720 +0.083 + 20%
ES49 : CH_MNIST_Uniform + 1.720 +0.083 + 20%
ES77: CH_MNIST_Bernoulli + 1.720 +0.083 + 20%
![CH_MNIST_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220026897-2b7bc960-dfeb-4eee-bbe4-4d1c995aa7b8.png)


RQ2: Effect of Distance between members and nonmembers
ES04: CH_MNIST_Uniform + 0.954 + 0.108 + 40%
ES14: CH_MNIST_Uniform + 1.355 + 0.108 + 40%
ES24: CH_MNIST_Uniform + 1.720 + 0.108 + 40%
                    
![CH_MNIST_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220026979-f6893d18-4fb7-4cbb-b1dc-7e33fe4b5fa0.png)


RQ3: Effect of Differential Distances between two datasets
ES03: CH_MNIST_Normal + 0.954 + 0.083 + 45%
ES05: CH_MNIST_Normal + 0.954 + 0.108 + 45%
ES07: CH_MNIST_Normal + 0.954 + 0.133 + 45%
![CH_MNIST_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220027031-de965164-7907-4ff6-8f4d-78dbdbc0409e.png)
          


RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack 
ES73: CH_MNIST_Bernoulli + 1.355 + 0.133 + 20%
ES74: CH_MNIST_Bernoulli + 1.355 + 0.133 + 40%
ES75: CH_MNIST_Bernoulli + 1.355 + 0.133 + 45%
ES76: CH_MNIST_Bernoulli + 1.355 + 0.133 + 49%
           
![CH_MNIST_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220027085-3a6964e6-edb5-4d55-b919-1a909f3c300c.png)

(4) ImageNet:
RQ1: Effect of data Distributions
ES02: ImageNet_Normal + 0.934 + 0.046 + 40%
ES30: ImageNet_Uniform + 0.934 + 0.046 + 40%
ES58: ImageNet_Bernoulli + 0.934 + 0.046 + 40%
![ImageNet_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220027209-2ac37e6d-4068-4055-a7db-ca5c35de9201.png)


RQ2: Effect of Distance between members and nonmembers
ES34: ImageNet_Uniform + 0.934 + 0.08 + 49% 
ES44: ImageNet_Uniform + 1.130 + 0.08 + 49%
ES54: ImageNet_Uniform + 1.388 + 0.08 + 49%
![ImageNet_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220027299-2748512f-2418-4a75-969f-422d39ee38d4.png)


RQ3: Effect of Differential Distances between two datasets
ES79: ImageNet_Bernoulli + 1.388 + 0.046 + 45% 
ES81: ImageNet_Bernoulli + 1.388 + 0.080 + 45%
ES83: ImageNet_Bernoulli + 1.388 + 0.145 + 45%
![ImageNet_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220027343-97532c83-2c71-4992-b3f8-dd9fc1789ba7.png)

RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack 
ES13: ImageNet_Normal + 1.130 + 0.080 + 20%
ES14: ImageNet_Normal + 1.130 + 0.080 + 40%
ES15: ImageNet_Normal + 1.130 + 0.080 + 45%
ES16: ImageNet_Normal + 1.130 + 0.080 + 49%
           
![ImageNet_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220027421-19592d0a-beea-4dba-9eb8-45009c09dc1d.png)


(5) Location30:
RQ1: Effect of data Distributions
ES01: Location30_Normal + 0.570 + 0.041 + 4%
ES29: Location30_Uniform + 0.570 + 0.041 + 4%
ES57: Location30_Bernoulli + 0.570 + 0.041 + 4% 
![Location30_RQ1 RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220057919-b581ec9e-80cd-4a0a-8efd-021c78c5515b.png)
 


RQ2: Effect of Distance between members and nonmembers
ES32: Location30_Uniform + 0.57 + 0.076 + 8%
ES42: Location30_Uniform + 0.724 + 0.076 + 8%
ES52: Location30_Uniform + 0.801 + 0.076 + 8%
                    
![Location30_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220057949-2f0389c7-738e-4ad2-ae44-56c388d20a9f.png)

RQ3: Effect of Differential Distances between two datasets
ES23: Location30_Normal + 0.801 + 0.041 + 12%
ES25: Location30_Normal + 0.801 + 0.076 + 12%
ES27: Location30_Normal + 0.801 + 0.094 + 12%
![Location30_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220058019-2694ad4a-f661-4e9a-8fea-ef5f7bba054a.png)
         
RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack
ES73: Location30_Bernoulli + 0.724 + 0.094 + 4%
ES74: Location30_Bernoulli + 0.724 + 0.094 + 8%
ES75: Location30_Bernoulli + 0.724 + 0.094 + 12%
ES76: Location30_Bernoulli + 0.724 + 0.094 + 16% 
           
![Location30_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220058062-812b8d3e-0cb1-425b-a94f-6896b3c98e8b.png)


(6) Purchase100:
RQ1: Effect of data Distributions
ES01: Purchase100_Normal + 0.550 + 0.087 + 2%
ES29: Purchase100_Uniform + 0.550 + 0.087 + 2%
ES57: Purchase100_Bernoulli + 0.550 + 0.087 + 2%
           
![Purchase100_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220282330-bde5d136-3dea-4ef6-9f08-833e58816532.png)

RQ2: Effect of Distance between members and nonmembers
ES04: Purchase100_Normal + 0.550 + 0.110 + 4%
ES14: Purchase100_Normal + 0.625 + 0.110 + 4%
ES24: Purchase100_Normal + 0.729 + 0.110 + 4%
![Purchase100_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220282362-9ae055f3-8e3a-4741-9781-683199aa5618.png)

RQ3: Effect of Differential Distances between two datasets
ES51: Purchase100_Uniform + 0.729 + 0.087 + 10%
ES53: Purchase100_Uniform + 0.729 + 0.110 + 10%
ES55: Purchase100_Uniform + 0.729 + 0.156 + 10%

                    
![Purchase100_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220282410-f5573edf-cb65-4a37-8d75-351488c676c2.png)

RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack
ES65: Purchase100_Bernoulli + 0.625 + 0.087 + 2%
ES66: Purchase100_Bernoulli + 0.625 + 0.087 + 4%
ES67: Purchase100_Bernoulli + 0.625 + 0.087 + 10%
ES68: Purchase100_Bernoulli + 0.625 + 0.087 + 12%
           
![Purchase100_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220282466-ee7ce129-b8e6-4705-9f52-cbe361e5a199.png)

(7) Texas100:
RQ1: Effect of data Distributions
ES01: Texas100_Normal + 0.530 + 0.038 + 2%
ES29: Texas100_Uniform + 0.530 + 0.038 + 2%
ES57: Texas100_Bernoulli + 0.530 + 0.038 + 2%
        
![Texas100_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220282556-caaea300-e406-4a46-bbb6-2eae9905fa23.png)
RQ2: Effect of Distance between members and nonmembers
ES02: Texas100_Normal + 0.530 + 0.038 + 4%
ES10: Texas100_Normal + 0.641 + 0.038 + 4%
ES22: Texas100_Normal + 0.734 + 0.038 + 4%
![Texas100_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220282619-c9f17da0-758c-4586-a8fa-7a2edba53f5d.png)
                

RQ3: Effect of Differential Distances between two datasets
ES51: Texas100_Uniform + 0.734 + 0.038 + 10%
ES53: Texas100_Uniform + 0.734 + 0.073 + 10%
ES55: Texas100_Uniform + 0.734 + 0.107 + 10%
                    
![Texas100_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220282636-4da0db4a-5f2f-47c0-910c-a84b9619c703.png)

RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack
ES65: Texas100_Bernoulli + 0.641 + 0.038 + 2%
ES66: Texas100_Bernoulli + 0.641 + 0.038 + 4%
ES67: Texas100_Bernoulli + 0.641 + 0.038 + 10%
ES68: Texas100_Bernoulli + 0.641 + 0.038 + 12%
           
![Texas100_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220282670-119c413a-6803-422e-9391-45eb2031d166.png)



