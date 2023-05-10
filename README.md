# MIBench.github.io
MIBench: A Comprehensive Benchmark for Membership Inference Attacks

MIBench is a comprehensive benchmark for comparing different MI attacks, which consists not only the evaluation metric module, but also the evaluation scenario module. And we design the evaluation scenarios from four perspectives: the distance distribution of data samples in the target dataset, the distance between data samples of the target dataset, the differential distance between two datasets (i.e., the target dataset and a generated dataset with only nonmembers), and the ratio of the samples that are made no inferences by an MI attack. The evaluation metric module consists of ten typical evaluation metrics (e.g., accuracy, precision, recall, f1-score, false positive rate (FPR), false negative rate (FNR), membership advantage (MA), the Area Under the Curve (AUC) of attack Receiver Operating Characteristic (ROC) curve, TPR @ fixed (low) FPR, threshold at maximum MA). We have identified three principles for the proposed “comparing different MI attacks” methodology, and we have designed and implemented the MIBench benchmark with 84 evaluation scenarios for each dataset. In total, we have used our benchmark to fairly and systematically compare 15 state-of-the-art MI attack algorithms across 588 evaluation scenarios, and these evaluation scenarios cover 7 widely used datasets and 7 representative types of models.

**MI attacks**:  
*    NN_attack  
*   Loss-Threshold  
* Label-only  
* Top3-NN attack  
* Top1-Threshold  
* BlindMI-Diff-w  
* BlindMI-Diff-w/o  
* BlindMI-Diff-1CLASS   
* Top2+True  
* Privacy Risk Scores  
* Shapley Values  
* Positive Predictive Value   
* Calibrated Score  
* Distillation-based Thre.  
* Likelihood ratio attack  

	
**Datasets**: CIFAR100, CIFAR10, CH_MNIST, ImageNet, Location30, 
Purchase100, Texas100


**Models**: MLP, StandDNN, VGG16, VGG19, ResNet50, ResNet101, DenseNet121


**Requirements**:
You can run the following script to configurate necessary environment
sh ./sh/install.sh

**Usage**:
   Please first to make a folder for record, all experiment results with save to record folder as default. And make folder for data to put supported datasets.
   XXX  XXX

**Attack**:
This is a demo script of running NN_attack on CIFAR100.
python ./attack/NN_attack.py --yaml_path ../config/attack/NN/CIFAR100.yaml --dataset CIFAR100 --dataset_path ../data --save_folder_name CIFAR100_0_1

Selected attacks:
![18c218f23f733985d975e2e89c486bd](https://user-images.githubusercontent.com/124696836/235963324-7ed2a0da-705c-473c-8009-be0320a65d4d.png)



**Evaluation Framework**:  
      MIBench is a comprehensive benchmark for comparing different MI attacks, which consists not only the evaluation metric module, but also the evaluation scenario module.
    
* Part I: Evaluation Scenarios

In this work, we have designed and implemented the MIBench benchmark with 84 evaluation scenarios for each dataset. In total, we have used our benchmark to fairly and systematically compare 15 state-of-the-art MI attack algorithms across 588 evaluation scenarios, and these evaluation scenarios cover 7 widely used datasets and 7 representative types of models.


(a) Evaluation Scenarios of CIFAR100.
<img width="1772" alt="CIFAR100" src="https://user-images.githubusercontent.com/124696836/236993194-eceadb11-28d1-42d0-9d16-e1aaf44bd20a.png">


(b) Evaluation Scenarios of CIFAR10.
<img width="1772" alt="CIFAR10" src="https://user-images.githubusercontent.com/124696836/236993585-562b528e-409c-4b61-8839-518e1d53b22d.png">


(c) Evaluation Scenarios of CH_MNIST.
<img width="1772" alt="CH_MNIST" src="https://user-images.githubusercontent.com/124696836/236993629-d890449e-1aaa-4bce-86ec-d23231842faa.png">


(d) Evaluation Scenarios of ImageNet.
<img width="1772" alt="ImageNet" src="https://user-images.githubusercontent.com/124696836/236993671-c0a1df25-311e-465f-9b72-760131c7c737.png">


(e) Evaluation Scenarios of Location30.
<img width="1772" alt="Location30" src="https://user-images.githubusercontent.com/124696836/236993708-eaa8d227-6b18-484c-9fb6-e97a350e049c.png">


(f) Evaluation Scenarios of Purchase100.
<img width="1772" alt="Purchase100" src="https://user-images.githubusercontent.com/124696836/236993768-337a8604-125e-4af5-92df-4c2e3769badb.png">


(g)  Evaluation Scenarios of Texas100.
<img width="1772" alt="Texas100" src="https://user-images.githubusercontent.com/124696836/236993798-513f11dc-0b56-463b-875e-15f3ead69010.png">


* Part II: Evaluation Metrics

    We mainly use attacker-side accuracy, precision, recall, f1-score, false positive rate (FPR), false negative rate (FNR), membership advantage (MA), the Area Under 
the Curve (AUC) of attack Receiver Operating Characteristic (ROC) curve, TPR @ fixed (low) FPR, threshold at maximum MA, as our evaluation metrics. The details of the evaluation metrics are shown as follows.

 (a) **accuracy**: the percentage of data samples with correct membership predictions by MI attacks;  
 (b) **precision**: the ratio of real-true members predicted among all the positive membership predictions made by an adversary;   
 (c) **recall**: the ratio of true members predicted by an adversary among all the real-true members;   
 (d) **f1-score**: the harmonic mean of precision and recall;   
 (e) **false positive rate (FPR)**: the ratio of nonmember samples are erroneously predicted as members;   
 (f) **false negative rate (FNR):** the difference of the 1 and recall (e.g., FNR=1-recall);      
 (g) **membership advantage (MA)**：the difference between the true positive rate and the false positive rate (e.g., MA = TPR - FPR);  
 (h) **Area Under the Curve (AUC)**: computed as the Area Under the Curve of attack Receiver Operating Characteristic (ROC);  
 (i) **TPR @ fixed (low) FPR**: an attack’s truepositive rate at (fixed) low false-positive rates;  
 (j) **threshold at maximum MA**: a threshold to achieve maximum MA.



**Results**:

   The results section consists of two parts: the results of 84 evaluation scenarios (ES) and the results of 4 research questions (RQ). And in each part, we identify the evaluation results of 15 state-of-the-art MI attacks by ten evaluation metrics (e.g., attacker-side accuracy, precision, recall, f1-score, FPR, FNR, MA, AUC, TPR @ fixed (low) FPR (T@0.01%F and T@0.1%F), threshold at maximum MA).
   
   * Part I: The Results of 84 Evaluation Scenarios

**1. Distillation-based**: 

   **(1) CIFAR100**:
   <img width="2144" alt="2023 5 9_Distillation-based_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/810dae09-cf18-42bb-9487-5893697dfb29">
   <img width="2144" alt="2023 5 9_Distillation-based_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4514c405-a92a-4b47-b160-10bd6cfab7b2">
   **(2) CIFAR10**:
   <img width="2144" alt="2023 5 9_Distillation-based_CIFAR10_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/315b99cd-ad47-4666-b15a-ae51d6a05671">
<img width="2144" alt="2023 5 9_Distillation-based_CIFAR10_不同评估场景_实验结果06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/20762444-650c-40bb-a4c5-217635ac1511">
   **(3) CH_MNIST**:
   <img width="2144" alt="2023 5 9_Distillation-based_CH_MNIST_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/cfdf53c2-3f2c-497b-a30e-fd77c4455a65">
<img width="2144" alt="2023 5 9_Distillation-based_CH_MNIST_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/ed4b22f7-1bcf-42fe-9a43-bfe4e4916333">
   **(4) ImageNet**: 
   <img width="2144" alt="2023 5 9_Distillation-based_ImageNet_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/b8c2f41c-1804-4dcb-a82c-3ebe197dd964">
<img width="2144" alt="2023 5 9_Distillation-based_ImageNet_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/aece8f19-1fc8-4887-8344-7c9d65b37285">
   **(5) Location30**: 
   <img width="2144" alt="2023 5 9_Distillation-based_Location30_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e7cb7225-4f0d-492b-b1db-8cd56f17a26c">
<img width="2144" alt="2023 5 9_Distillation-based_Location30_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/9044f8d0-17c1-48d2-b2f1-d4c07952be04">   
   **(6) Purchase100**:  
   <img width="2144" alt="2023 5 9_Distillation-based_Purchase100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/092f6fd1-ea33-40c3-b5f5-dd4749eb97f4">
<img width="2144" alt="2023 5 9_Distillation-based_Purchase100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/2a391173-87c4-4870-988b-5a0d36083082">
   **(7) Texas100**:  
   <img width="2144" alt="2023 5 9_Distillation-based_Texas100_不同评估场景_实验结果_15" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3357bb10-425e-4c16-93b9-66f5b0b127df">
<img width="2144" alt="2023 5 9_Distillation-based_Texas100_不同评估场景_实验结果_16" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3b40d920-7bcd-414f-a243-ce1b37734ae1">
   
   **2. Calibrated Score**:  
  
  **(1) CIFAR100**:
  <img width="2055" alt="2023 4 23_Calibrated Score_CIFAR100_不同评估场景_实验结果(2)_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/9d8bb788-610d-4abe-b6e9-cbd771c8f832">
<img width="2055" alt="2023 4 23_Calibrated Score_CIFAR100_不同评估场景_实验结果(2)_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6cbd7bce-cfd0-41b7-9f94-6b323b552bc4">
   **(2) CIFAR10**:
   <img width="2055" alt="2023 4 23_Calibrated Score_CIFAR10_不同评估场景_实验结果(2)_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/269717f4-a6da-4900-b06b-de1e53a594f6">
<img width="2055" alt="2023 4 23_Calibrated Score_CIFAR10_不同评估场景_实验结果(2)_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1ed746bc-3cfd-4584-af74-a05299ac7510">
   **(3) CH_MNIST**:
   <img width="2055" alt="2023 4 23_Calibrated Score_CH_MNIST_不同评估场景_实验结果(2)_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6f87b3eb-3c20-41aa-8fa5-46b6adbb55ca">
<img width="2055" alt="2023 4 23_Calibrated Score_CH_MNIST_不同评估场景_实验结果(2)_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/61fb8ef7-184a-4bf8-9d35-847f5bac4977">
   **(4) ImageNet**: 
   <img width="2055" alt="2023 4 23_Calibrated Score_ImageNet_不同评估场景_实验结果(2)_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/18469773-03cf-4a68-a0ba-08ff277d82c9">
<img width="2055" alt="2023 4 23_Calibrated Score_ImageNet_不同评估场景_实验结果(2)_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/175087b1-bb8a-40a2-9033-d7b105ff81d7">
   **(5) Purchase100**:  
   <img width="2055" alt="2023 4 23_Calibrated Score_Purchase100_不同评估场景_实验结果(2)_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c7d4dfdd-4a77-44df-b704-9d7de50230d4">
<img width="2055" alt="2023 4 23_Calibrated Score_Purchase100_不同评估场景_实验结果(2)_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/05245229-8f6a-4623-839a-5af3712500db">
   **(6) Texas100**:  
   <img width="2055" alt="2023 4 23_Calibrated Score_Texas100_不同评估场景_实验结果(2)_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/8c82861c-008d-4f41-b36a-8e21cb614fe4">
<img width="2055" alt="2023 4 23_Calibrated Score_Texas100_不同评估场景_实验结果(2)_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/52ddb19c-70ee-4cc3-b5db-cbbd8b877ec4">

   **3. Label-only**:  
  
  **(1) CIFAR100**:
  <img width="2144" alt="2023 4 23_Label-only_CIFAR100_不同评估场景_实验结果4 29_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7b752c90-961e-4aa8-a088-d1cb25fd07db">
<img width="2144" alt="2023 4 23_Label-only_CIFAR100_不同评估场景_实验结果4 29_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a4249303-8371-486d-ab52-85ede2c751d5">
   **(2) CIFAR10**:
   <img width="2144" alt="2023 4 23_Label-only_CIFAR10_不同评估场景_实验结果4 29_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/521c6462-a6c8-45c1-ac74-9584cdd4b078">
<img width="2144" alt="2023 4 23_Label-only_CIFAR10_不同评估场景_实验结果4 29_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/995eea40-55a6-46c6-bd6d-ad4ac5ef76ec">
   **(3) CH_MNIST**:
   <img width="2144" alt="2023 4 23_Label-only_CH_MNST_不同评估场景_实验结果4 29_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3e7d28f0-55f5-46e5-a4e4-545cfd3d16f0">
<img width="2144" alt="2023 4 23_Label-only_CH_MNST_不同评估场景_实验结果4 29_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/ad3d5fa6-685d-457c-a0ee-ff08f66eb0c4">
   **(4) ImageNet**:
   <img width="2144" alt="2023 4 23_Label-only_ImageNet_不同评估场景_实验结果4 29_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/23a3929a-ff35-4aa0-9611-ffcb6045cc57">
<img width="2144" alt="2023 4 23_Label-only_ImageNet_不同评估场景_实验结果4 29_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1fff91f9-f465-4f21-8e63-9e78f00da047">
   **(5) Location30**:  
   <img width="2144" alt="2023 4 23_Label-only_Location30_不同评估场景_实验结果4 29_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a1637097-ed6e-4438-9150-0d37d1159e18">
<img width="2144" alt="2023 4 23_Label-only_Location30_不同评估场景_实验结果4 29_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/419fe47f-ccd2-4efa-8235-b3d9d71d0206">
   **(6) Purchase100**:  
   <img width="2144" alt="2023 4 23_Label-only_Purchase100_不同评估场景_实验结果4 29_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c6b11a65-28b8-432d-a06f-1063bb4f955a">
<img width="2144" alt="2023 4 23_Label-only_Purchase100_不同评估场景_实验结果4 29_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/b714d3b9-62e7-41fd-8014-bd82f134eaf5">
   **(7) Texas100**:  
   <img width="2144" alt="2023 4 23_Label-only_Texas100_不同评估场景_实验结果4 29_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/879ad9bf-f3d5-4551-89d2-ed250000faee">
<img width="2144" alt="2023 4 23_Label-only_Texas100_不同评估场景_实验结果4 29_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/cd748f60-3ab1-4391-aeed-588c8135b965">

   **4. NN_attack**:  
   
  **(1) CIFAR100**:
  <img width="2191" alt="2023 4 23_NN_attack_CIFAR100_不同评估场景_实验结果4 29_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/18eb8657-b783-455e-bbad-51ad12839485">
<img width="2191" alt="2023 4 23_NN_attack_CIFAR100_不同评估场景_实验结果4 29_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a28c16a8-6793-49eb-a7b9-874591790873">
   **(2) CIFAR10**:
<img width="2191" alt="2023 4 23_NN_attack_CIFAR10_不同评估场景_实验结果4 29_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/fd93b88d-acb0-4596-95c9-a92100a4cb46">
<img width="2191" alt="2023 4 23_NN_attack_CIFAR10_不同评估场景_实验结果4 29_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/acaa310c-5105-43b3-819e-2e6b216a6cef">
   **(3) CH_MNIST**:
   <img width="2191" alt="2023 4 23_NN_attack_CH_MINST_不同评估场景_实验结果4 29_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/bdcb9c06-b6db-43d5-a816-4cfc0e653b59">
<img width="2191" alt="2023 4 23_NN_attack_CH_MINST_不同评估场景_实验结果4 29_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d5e1358e-1c23-4e85-8823-fcdb1273de9b">
   **(4) ImageNet**:   
<img width="2191" alt="2023 4 23_NN_attack_ImageNet_不同评估场景_实验结果4 29_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d95149d4-e5a7-4461-8398-e6f094dbc4cf">
<img width="2191" alt="2023 4 23_NN_attack_ImageNet_不同评估场景_实验结果4 29_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e417fc99-a408-482d-83f7-920957761f78">
   **(5) Location30**: 
   <img width="2191" alt="2023 4 23_NN_attack_Location30_不同评估场景_实验结果4 29_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/62e65734-c51c-481f-bb6a-1cb664f2c844">
<img width="2191" alt="2023 4 23_NN_attack_Location30_不同评估场景_实验结果4 29_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d6ea1a15-159b-4080-b6bf-d6df9a052fd8">
   **(6) Purchase100**:  
   <img width="2191" alt="2023 4 23_NN_attack_Purchase100_不同评估场景_实验结果4 29_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/531159e5-1adc-4a7f-b3b2-f28656987041">
<img width="2191" alt="2023 4 23_NN_attack_Purchase100_不同评估场景_实验结果4 29_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/bdece8c9-a6ed-4fa4-bad3-249c0de13b2d">
   **(7) Texas100**:  
   <img width="2191" alt="2023 4 23_NN_attack_Texas100_不同评估场景_实验结果4 29_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4eccc883-5d06-41fc-bfe8-e516198301c9">
<img width="2191" alt="2023 4 23_NN_attack_Texas100_不同评估场景_实验结果4 29_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/b15443cd-5716-4ba9-8eae-f36905ab072e">

   **5. PPV**:  
  
  **(1) CIFAR100**:
  <img width="2115" alt="2023 4 23_PPV_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d3d98ae7-839b-47f6-b880-4335163d8200">
<img width="2115" alt="2023 4 23_PPV_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/53eeb696-cc02-4c7d-9ebb-21b7d864322b">
   **(2) CIFAR10**:
<img width="2115" alt="2023 4 23_PPV_CIFAR10_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/99adf1ff-89d4-4df8-8134-15c456c806a9">
<img width="2115" alt="2023 4 23_PPV_CIFAR10_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/83a8f649-52e4-4b04-b74c-25d0667dbab0">
   **(3) CH_MNIST**:
   <img width="2115" alt="2023 4 23_PPV_CH_MINST_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/ef769106-0f21-4021-aa0d-029f2b631d6d">
<img width="2115" alt="2023 4 23_PPV_CH_MINST_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/39e6d0e4-4937-446f-9316-2b4b0dc10875">
   **(4) ImageNet**:   
<img width="2115" alt="2023 4 23_PPV_ImageNet_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/9779f51d-f863-4cd4-931e-d86699e7bd14">
<img width="2115" alt="2023 4 23_PPV_ImageNet_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/48810c69-7315-4e0b-a2fe-bb9106493262">
   **(5) Location30**:  
   <img width="2115" alt="2023 4 23_PPV_Location30_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/f3b1bca8-e693-41be-a46b-cf767c967ee9">
<img width="2115" alt="2023 4 23_PPV_Location30_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/02d6e06d-deba-4c89-881f-64a0349fb61f">
   **(6) Purchase100**:  
   <img width="2115" alt="2023 4 23_PPV_Purchase100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1a6a81b3-114c-43a5-b260-31c29ab540fd">
<img width="2115" alt="2023 4 23_PPV_Purchase100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/9579160d-35ac-4d39-8428-a54037793e3a">
   **(7) Texas100**:  
   <img width="2115" alt="2023 4 23_PPV_Texas100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/0f6673f7-226f-48ee-82fd-d3822f41538d">
<img width="2115" alt="2023 4 23_PPV_Texas100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/31c4d45d-1e65-49f3-ad91-26ff3a6e392f">

   **6. Risk score**:  
  
  **(1) CIFAR100**:
  <img width="2127" alt="2023 4 23_Risk score_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7aaa6735-9b41-40a7-bc46-6e63abc4a000">
<img width="2127" alt="2023 4 23_Risk score_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/75c1c7bc-d544-45d2-9112-ea7f282f1b4d">
   **(2) CH_MNIST**:
   <img width="2127" alt="2023 4 23_Risk score_CH_MNST_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3e1c3d98-4f89-4788-a2a9-90660615c303">
<img width="2127" alt="2023 4 23_Risk score_CH_MNST_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/b15919e9-e19d-4094-b52e-d6767327cbf4">
   **(3) ImageNet**:
   <img width="2127" alt="2023 4 23_Risk score_ImageNet_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/cafc0e76-b109-46d6-b39e-6552d9968c29">
<img width="2127" alt="2023 4 23_Risk score_ImageNet_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/fd1d0de4-ec38-4761-a4e8-8267933ad643">
   **(4) Location30**:   
<img width="2127" alt="2023 4 23_Risk score_Location30_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e5de4041-f847-43e6-a6a3-b8c1a3dfc769">
<img width="2127" alt="2023 4 23_Risk score_Location30_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6971df83-d6b3-467f-b0fb-11053daa8333">
   **(5) Purchase100**:  
   <img width="2127" alt="2023 4 23_Risk score_Purchase100_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c1e23aaa-09d3-465f-a5dc-3c70033917a4">
<img width="2127" alt="2023 4 23_Risk score_Purchase100_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/b5c167b0-75ac-4f09-ac82-a20387107fd7">
   **(6) Texas100**:  
  <img width="2127" alt="2023 4 23_Risk score_Texas100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/171b6837-a763-4f5c-ba9a-c5e99100fb76">
<img width="2127" alt="2023 4 23_Risk score_Texas100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d365d4d2-c409-4b24-8168-61fe01222830">

   **7. Shapley values**:  
  
  **(1) CIFAR100**:
  <img width="2142" alt="2023 4 23_Shapley values_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/06b55895-8081-4292-a31e-0d456555d3d6">
<img width="2142" alt="2023 4 23_Shapley values_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/0e8c6ade-ddb0-46ab-bef5-c701d158ddaf">
   **(2) CIFAR10**:
<img width="2142" alt="2023 4 23_Shapley values_CIFAR10_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/fe82f861-6e9f-4bd8-9920-1d5f3d07ffcd">
<img width="2142" alt="2023 4 23_Shapley values_CIFAR10_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/823c904e-14b4-4779-846e-d6e92732c67c">
   **(3) CH_MNIST**:
   <img width="2142" alt="2023 4 23_Shapley values_CH_MNST_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a6163099-f3f2-4bd4-a9f7-60556c157dc4">
<img width="2142" alt="2023 4 23_Shapley values_CH_MNST_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/ce531ec5-1ecf-45c1-9db6-1bd40e553dd2">
   **(4) ImageNet**:   
<img width="2142" alt="2023 4 23_Shapley values_ImageNet_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/17e45aad-1d47-4f8d-80cf-89155248b69e">
<img width="2142" alt="2023 4 23_Shapley values_ImageNet_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/91cc2e3b-ad99-444a-9182-09a444fc4b7d">
   **(5) Location30**:  
   <img width="2142" alt="2023 4 23_Shapley values_Location30_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/0401e1aa-9b9f-4783-802e-d127130a202f">
<img width="2142" alt="2023 4 23_Shapley values_Location30_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/97df89a4-47f2-4d2d-abc7-d95932fc9fb7">
   **(6) Purchase100**:  
   <img width="2142" alt="2023 4 23_Shapley values_Purchase100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a62ac06c-b2a8-465c-aa8b-47e16a986ac1">
<img width="2142" alt="2023 4 23_Shapley values_Purchase100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d99d4b63-0cff-47b2-82d8-f40f42a29456">
   **(7) Texas100**:  
   <img width="2142" alt="2023 4 23_Shapley values_Texas100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/9ad709ce-0291-4dab-850a-043acdf8609b">
<img width="2142" alt="2023 4 23_Shapley values_Texas100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/960a880b-3d1a-4ce7-aff1-184ffadabbcd">
   **8. Top1_Threshold**:  
  
  **(1) CIFAR100**:
   <img width="2190" alt="2023 4 23_Top1_Threshold_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d7fb3461-cfa8-42c4-9f71-62c65546a519">
<img width="2190" alt="2023 4 23_Top1_Threshold_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/2c3b2087-c793-450b-b2a9-ce10bc73e3a8">
   **(2) CIFAR10**:
<img width="2190" alt="2023 4 23_Top1_Threshold_CIFAR10_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d2869ee9-032c-4d22-8f85-8d4f3f160f11">
<img width="2190" alt="2023 4 23_Top1_Threshold_CIFAR10_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6cfef302-01f6-4060-9f95-129cbf952d09">
   **(3) CH_MNIST**:
   <img width="2190" alt="2023 4 23_Top1_Threshold_CH_MNST_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4efe50aa-b0a6-4f84-a63d-fb77100e2787">
<img width="2190" alt="2023 4 23_Top1_Threshold_CH_MNST_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/ad2a5a40-a27f-4595-bba8-ec05122937ad">
   **(4) ImageNet**:   
<img width="2190" alt="2023 4 23_Top1_Threshold_ImageNet_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a7c532f3-5eb1-4660-b8bf-9084c6ef2fd9">
<img width="2190" alt="2023 4 23_Top1_Threshold_ImageNet_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3ae8460a-c482-4e85-a2ae-be7c0cc7210f">
   **(5) Location30**:  
   <img width="2190" alt="2023 4 23_Top1_Threshold_Location30_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a5cacaf5-d448-4039-b674-d948ad73b00a">
<img width="2190" alt="2023 4 23_Top1_Threshold_Location30_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/937d020a-975a-407b-b356-7ea0718e97aa">
   **(6) Purchase100**:  
   <img width="2190" alt="2023 4 23_Top1_Threshold_Purchase100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6ec7470f-7712-48ab-ab0a-8b0517991d7f">
<img width="2190" alt="2023 4 23_Top1_Threshold_Purchase100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/74731d3e-157a-46da-b301-dfc56c4ebd4c">
   **(7) Texas100**:  
   <img width="2190" alt="2023 4 23_Top1_Threshold_Texas100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c7a14e35-b8f6-4cb7-a573-3b3390664b77">
<img width="2190" alt="2023 4 23_Top1_Threshold_Texas100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/74bd107b-6d8d-483e-8cee-d9f4faa61a78">

   **9. BlinMI-1CLASS**:  
  
  **(1) CIFAR100**:
  <img width="2202" alt="2023 4 27_BlinMI-1CLASS_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e4857763-e208-44ad-b317-458e4c62d7a8">
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a274e686-b6c3-4584-83fe-59aeb5ae00cd">
   **(2) CIFAR10**:
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_CIFAR10_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/5cab2f4c-b4f8-427e-be9c-d1d71048eef4">
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_CIFAR10_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/539a0d59-e3b3-4ae7-9630-576e419bef2f">
 **(3) CH_MNIST**:
   <img width="2202" alt="2023 4 27_BlinMI-1CLASS_CH_MNST_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/2ebf1a71-a51e-41a7-ae04-ee1738cdce25">
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_CH_MNST_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3c36731c-e6d1-4f0a-b36e-08ccec5356f2">
   **(4) ImageNet**:   
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_ImageNet_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e8c4a404-f36b-4445-a4d3-39baadd9f978">
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_ImageNet_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/16d22e44-6e20-4666-b41d-5c35e7ce5ea2">
   **(5) Location30**:  
   <img width="2202" alt="2023 4 27_BlinMI-1CLASS_Location30_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/cc605e5e-5917-4749-a928-7dfaca5c6054">
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_Location30_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a869f45a-b761-4df3-be20-61dc31f7cf7e">
   **(6) Purchase100**:  
   <img width="2202" alt="2023 4 27_BlinMI-1CLASS_Purchase100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d5aabb81-f3f7-41bd-bd39-61ea23458339">
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_Purchase100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/68a147e4-0b80-471c-983b-a2e3886c359e">
   **(7) Texas100**:  
   <img width="2202" alt="2023 4 27_BlinMI-1CLASS_Texas100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/43b40b9a-0618-42f9-b4bf-6454df531f9d">
<img width="2202" alt="2023 4 27_BlinMI-1CLASS_Texas100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/01fb9fc9-f327-4209-b4cf-f093700eda1b">

   **10. Top3_NN**:  
  
  **(1) CIFAR100**:
  <img width="2156" alt="2023 4 30_Top3_NN_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3cae7047-c11b-4190-bafa-eb189033b249">
<img width="2156" alt="2023 4 30_Top3_NN_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4b575bcb-555b-4a95-95e6-bc87e408a1de">
  **(2) CIFAR10**:
   <img width="2156" alt="2023 4 30_Top3_NN_CIFAR10_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1cbb901a-5d18-45e0-8e83-58cdc7c429a0">
<img width="2156" alt="2023 4 30_Top3_NN_CIFAR10_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/404d30d7-6afc-461a-9f8c-b999f57b1589">
   **(3) CH_MNIST**:
   <img width="2156" alt="2023 4 30_Top3_NN_CH_MNST_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/05ce8306-1161-4bdc-b7e7-86b3cff442d2">
<img width="2156" alt="2023 4 30_Top3_NN_CH_MNST_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/97868227-79d3-41ab-9431-a2bdb2cd51a6">
   **(4) ImageNet**:   
<img width="2156" alt="2023 4 30_Top3_NN_ImageNet_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e043b02e-3bdc-4788-acc5-3eb6578a1f4f">
<img width="2156" alt="2023 4 30_Top3_NN_ImageNet_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1d281608-7381-491d-a5c1-c123b7380176">
   **(5) Location30**:  
   <img width="2156" alt="2023 4 30_Top3_NN_Location30_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/57fddc03-c94a-44d3-b37f-f4ed120bfbb8">
<img width="2156" alt="2023 4 30_Top3_NN_Location30_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1626202c-43d0-4d8f-9982-66de34dbe6f6">
   **(6) Purchase100**:  
   <img width="2156" alt="2023 4 30_Top3_NN_Purchase100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/f7b97798-d33d-4193-94d5-5c3029849152">
<img width="2156" alt="2023 4 30_Top3_NN_Purchase100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/2082e9df-634f-4c32-9d48-bf86476f0542">
   **(7) Texas100**: 
   <img width="2156" alt="2023 4 30_Top3_NN_Texas100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a1b35592-1365-40bd-9ceb-1a7f2fd65cb0">
<img width="2156" alt="2023 4 30_Top3_NN_Texas100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/61acac43-39e2-4ba7-92db-f0330e777e26">

   **11. LiRA**:  
  
  **(1) CIFAR100**:
  <img width="2167" alt="2023 5 1_LiRA_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6b39bc15-40e8-4614-80de-944a568604aa">
<img width="2167" alt="2023 5 1_LiRA_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/2519152d-45ef-4f88-897b-697a181810ad">
  **(2) CIFAR10**:
   <img width="2167" alt="2023 5 1_LiRA_CIFAR10_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/02a85dca-cb9b-469f-b9e4-7328001d6c0a">
<img width="2167" alt="2023 5 1_LiRA_CIFAR10_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/93405a5e-70de-4405-8a8d-4eb244187874">
   **(3) CH_MNIST**:
   
   **(4) ImageNet**:   

   **(5) Location30**:  
   
   **(6) Purchase100**:  
   
   **(7) Texas100**: 
   
   **9. BlinMI-1CLASS**:  
  
  **(1) CIFAR100**:
  
  **(2) CIFAR10**:
   
   **(3) CH_MNIST**:
   
   **(4) ImageNet**:   

   **(5) Location30**:  
   
   **(6) Purchase100**:  
   
   **(7) Texas100**: 
   
   **9. BlinMI-1CLASS**:  
  
  **(1) CIFAR100**:
  
  **(2) CIFAR10**:
   
   **(3) CH_MNIST**:
   
   **(4) ImageNet**:   

   **(5) Location30**:  
   
   **(6) Purchase100**:  
   
   **(7) Texas100**: 
   
   **9. BlinMI-1CLASS**:  
  
  **(1) CIFAR100**:
  
  **(2) CIFAR10**:
   
   **(3) CH_MNIST**:
   
   **(4) ImageNet**:   

   **(5) Location30**:  
   
   **(6) Purchase100**:  
   
   **(7) Texas100**: 
   
   **9. BlinMI-1CLASS**:  
  
  **(1) CIFAR100**:
  
  **(2) CIFAR10**:
   
   **(3) CH_MNIST**:
   
   **(4) ImageNet**:   

   **(5) Location30**:  
   
   **(6) Purchase100**:  
   
   **(7) Texas100**: 
   
   
   * Part II: The Results of 4 Research Questions
   
**(1) CIFAR100**:

**RQ1: Effect of Distance Distribution of Data Damples in the Target Dataset**

ES01: CIFAR100_Normal + 2.893 + 0.085 + 20%                       
ES29: CIFAR100_Uniform + 2.893 + 0.085 + 20%                       
ES57: CIFAR100_Bernoulli + 2.893 +0.085 + 20%
![CIFAR100_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/219857423-023f397c-8b77-4406-8f17-344609c83721.png)
   

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES02: CIFAR100_Normal + 2.893 + 0.085 + 40%                        
ES10: CIFAR100_Normal + 3.813 + 0.085 + 40%                       
ES22: CIFAR100_Normal + 4.325 + 0.085 + 40%
![CIFAR100_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/219857440-b9a22ebc-8f68-48b8-af90-57f390781d13.png)



**RQ3: Effect of Differential Distances between two datasets**

ES03: CIFAR100_Normal + 2.893 + 0.085 + 45%                       
ES05: CIFAR100_Normal + 2.893 + 0.119 + 45%                       
ES07: CIFAR100_Normal + 2.893 + 0.157 + 45%
![CIFAR100_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/219857466-c0857043-b7d5-4bfb-a1a2-709d26b5b7a1.png)


**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES37: CIFAR100_Uniform + 3.813 + 0.085 + 20%                       
ES38: CIFAR100_Uniform + 3.813 + 0.085 + 40%                       
ES39: CIFAR100_Uniform + 3.813 + 0.085 + 45%                        
ES40: CIFAR100_Uniform + 3.813 + 0.085 + 49%
![CIFAR100_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/219857475-56cccc31-1e56-4f75-8f6a-f60cdf096396.png)


(2) CIFAR10:

**RQ1: Effect of Distance Distribution of Data Damples in the Target Dataset**

ES13: CIFAR10_Normal + 2.501 + 0.213 + 20%                       
ES41: CIFAR10_Uniform + 2.501 + 0.213 + 20%                       
ES69: CIFAR10_Bernoulli + 2.501 + 0.213 + 20%
![CIFAR10_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/219933681-e0ab7da6-63e6-4d77-90c8-8bfa4e69c67c.png)
 

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES02: CIFAR10_Normal + 1.908 + 0.155 + 40%                       
ES10: CIFAR10_Normal + 2.501 + 0.155 + 40%                       
ES22: CIFAR10_Normal + 3.472 + 0.155 + 40%
![CIFAR10_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/219933690-c9b04621-9d44-43c5-8dbe-3e6474dcd781.png)


**RQ3: Effect of Differential Distances between two datasets**

ES51: CIFAR10_Uniform + 3.472 + 0.155 + 45%                       
ES53: CIFAR10_Uniform + 3.472 + 0.213 + 45%                        
ES55: CIFAR10_Uniform + 3.472 + 0.291 + 45%
![CIFAR10_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/219933691-86b1ae51-f4b4-4c96-8428-fa8c1e83b5fd.png)


**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES57: CIFAR10_Bernoulli + 1.908 +0.155 + 20%                        
ES58: CIFAR10_Bernoulli + 1.908 + 0.155 + 40%                       
ES59: CIFAR10_Bernoulli + 1.908 + 0.155 + 45%                       
ES60: CIFAR10_Bernoulli + 1.908 + 0.155 + 49%
![CIFAR10_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/219933699-f10df496-f69b-47a2-8fa8-d31f4c0a2cfb.png)
   

(3) CH_MNIST:

**RQ1: Effect of Distance Distribution of Data Damples in the Target Dataset**

ES21: CH_MNIST_Normal + 1.720 +0.083 + 20%                       
ES49 : CH_MNIST_Uniform + 1.720 +0.083 + 20%                       
ES77: CH_MNIST_Bernoulli + 1.720 +0.083 + 20%
![CH_MNIST_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220026897-2b7bc960-dfeb-4eee-bbe4-4d1c995aa7b8.png)


**RQ2: Effect of Distance between data samples of the Target Dataset**

ES04: CH_MNIST_Uniform + 0.954 + 0.108 + 40%                       
ES14: CH_MNIST_Uniform + 1.355 + 0.108 + 40%                       
ES24: CH_MNIST_Uniform + 1.720 + 0.108 + 40%
![CH_MNIST_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220026979-f6893d18-4fb7-4cbb-b1dc-7e33fe4b5fa0.png)


**RQ3: Effect of Differential Distances between two datasets**

ES03: CH_MNIST_Normal + 0.954 + 0.083 + 45%                                              
ES05: CH_MNIST_Normal + 0.954 + 0.108 + 45%                                              
ES07: CH_MNIST_Normal + 0.954 + 0.133 + 45%
![CH_MNIST_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220027031-de965164-7907-4ff6-8f4d-78dbdbc0409e.png)
          


**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES73: CH_MNIST_Bernoulli + 1.355 + 0.133 + 20%                                              
ES74: CH_MNIST_Bernoulli + 1.355 + 0.133 + 40%                                              
ES75: CH_MNIST_Bernoulli + 1.355 + 0.133 + 45%                                              
ES76: CH_MNIST_Bernoulli + 1.355 + 0.133 + 49%
![CH_MNIST_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220027085-3a6964e6-edb5-4d55-b919-1a909f3c300c.png)

(4) ImageNet:

**RQ1: Effect of Distance Distribution of Data Damples in the Target Dataset**

ES02: ImageNet_Normal + 0.934 + 0.046 + 40%                                              
ES30: ImageNet_Uniform + 0.934 + 0.046 + 40%                                              
ES58: ImageNet_Bernoulli + 0.934 + 0.046 + 40%
![ImageNet_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220027209-2ac37e6d-4068-4055-a7db-ca5c35de9201.png)


**RQ2: Effect of Distance between data samples of the Target Dataset**

ES34: ImageNet_Uniform + 0.934 + 0.08 + 49%                                               
ES44: ImageNet_Uniform + 1.130 + 0.08 + 49%                                              
ES54: ImageNet_Uniform + 1.388 + 0.08 + 49%
![ImageNet_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220027299-2748512f-2418-4a75-969f-422d39ee38d4.png)


**RQ3: Effect of Differential Distances between two datasets**

ES79: ImageNet_Bernoulli + 1.388 + 0.046 + 45%                                               
ES81: ImageNet_Bernoulli + 1.388 + 0.080 + 45%                                              
ES83: ImageNet_Bernoulli + 1.388 + 0.145 + 45%
![ImageNet_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220027343-97532c83-2c71-4992-b3f8-dd9fc1789ba7.png)

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES13: ImageNet_Normal + 1.130 + 0.080 + 20%                       
ES14: ImageNet_Normal + 1.130 + 0.080 + 40%                       
ES15: ImageNet_Normal + 1.130 + 0.080 + 45%                       
ES16: ImageNet_Normal + 1.130 + 0.080 + 49%
![ImageNet_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220027421-19592d0a-beea-4dba-9eb8-45009c09dc1d.png)


(5) Location30:

**RQ1: Effect of Distance Distribution of Data Damples in the Target Dataset**

ES01: Location30_Normal + 0.570 + 0.041 + 4%                       
ES29: Location30_Uniform + 0.570 + 0.041 + 4%                       
ES57: Location30_Bernoulli + 0.570 + 0.041 + 4%
![Location30_RQ1 RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220057919-b581ec9e-80cd-4a0a-8efd-021c78c5515b.png)
 


**RQ2: Effect of Distance between data samples of the Target Dataset**

ES32: Location30_Uniform + 0.57 + 0.076 + 8%                       
ES42: Location30_Uniform + 0.724 + 0.076 + 8%                       
ES52: Location30_Uniform + 0.801 + 0.076 + 8%
![Location30_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220057949-2f0389c7-738e-4ad2-ae44-56c388d20a9f.png)

**RQ3: Effect of Differential Distances between two datasets**

ES23: Location30_Normal + 0.801 + 0.041 + 12%                       
ES25: Location30_Normal + 0.801 + 0.076 + 12%                       
ES27: Location30_Normal + 0.801 + 0.094 + 12%
![Location30_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220058019-2694ad4a-f661-4e9a-8fea-ef5f7bba054a.png)
         
**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES73: Location30_Bernoulli + 0.724 + 0.094 + 4%                       
ES74: Location30_Bernoulli + 0.724 + 0.094 + 8%                       
ES75: Location30_Bernoulli + 0.724 + 0.094 + 12%                       
ES76: Location30_Bernoulli + 0.724 + 0.094 + 16%
![Location30_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220058062-812b8d3e-0cb1-425b-a94f-6896b3c98e8b.png)


(6) Purchase100:

**RQ1: Effect of Distance Distribution of Data Damples in the Target Dataset**

ES01: Purchase100_Normal + 0.550 + 0.087 + 2%                       
ES29: Purchase100_Uniform + 0.550 + 0.087 + 2%                       
ES57: Purchase100_Bernoulli + 0.550 + 0.087 + 2%
![Purchase100_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220282330-bde5d136-3dea-4ef6-9f08-833e58816532.png)

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES04: Purchase100_Normal + 0.550 + 0.110 + 4%                       
ES14: Purchase100_Normal + 0.625 + 0.110 + 4%                       
ES24: Purchase100_Normal + 0.729 + 0.110 + 4%
![Purchase100_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220282362-9ae055f3-8e3a-4741-9781-683199aa5618.png)

**RQ3: Effect of Differential Distances between two datasets**

ES51: Purchase100_Uniform + 0.729 + 0.087 + 10%                       
ES53: Purchase100_Uniform + 0.729 + 0.110 + 10%                       
ES55: Purchase100_Uniform + 0.729 + 0.156 + 10%
![Purchase100_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220282410-f5573edf-cb65-4a37-8d75-351488c676c2.png)

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES65: Purchase100_Bernoulli + 0.625 + 0.087 + 2%                       
ES66: Purchase100_Bernoulli + 0.625 + 0.087 + 4%                       
ES67: Purchase100_Bernoulli + 0.625 + 0.087 + 10%                       
ES68: Purchase100_Bernoulli + 0.625 + 0.087 + 12%
![Purchase100_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220282466-ee7ce129-b8e6-4705-9f52-cbe361e5a199.png)

(7) Texas100:

**RQ1: Effect of Distance Distribution of Data Damples in the Target Dataset**

ES01: Texas100_Normal + 0.530 + 0.038 + 2%                       
ES29: Texas100_Uniform + 0.530 + 0.038 + 2%                       
ES57: Texas100_Bernoulli + 0.530 + 0.038 + 2%
![Texas100_RQ1 Effect of data Distributions](https://user-images.githubusercontent.com/124696836/220282556-caaea300-e406-4a46-bbb6-2eae9905fa23.png)

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES02: Texas100_Normal + 0.530 + 0.038 + 4%                       
ES10: Texas100_Normal + 0.641 + 0.038 + 4%                       
ES22: Texas100_Normal + 0.734 + 0.038 + 4%
![Texas100_RQ2 Effect of Distance between members and nonmembers](https://user-images.githubusercontent.com/124696836/220282619-c9f17da0-758c-4586-a8fa-7a2edba53f5d.png)
                

**RQ3: Effect of Differential Distances between two datasets**

ES51: Texas100_Uniform + 0.734 + 0.038 + 10%                       
ES53: Texas100_Uniform + 0.734 + 0.073 + 10%                       
ES55: Texas100_Uniform + 0.734 + 0.107 + 10%
![Texas100_RQ3 Effect of Differential Distances between two datasets](https://user-images.githubusercontent.com/124696836/220282636-4da0db4a-5f2f-47c0-910c-a84b9619c703.png)

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES65: Texas100_Bernoulli + 0.641 + 0.038 + 2%                       
ES66: Texas100_Bernoulli + 0.641 + 0.038 + 4%                       
ES67: Texas100_Bernoulli + 0.641 + 0.038 + 10%                       
ES68: Texas100_Bernoulli + 0.641 + 0.038 + 12%
![Texas100_RQ4 Effect of the Ratios of the samples that are made no inferences by an MI attack](https://user-images.githubusercontent.com/124696836/220282670-119c413a-6803-422e-9391-45eb2031d166.png)



