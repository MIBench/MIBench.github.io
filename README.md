# MIBench.github.io
Comparing Different Membership Inference Attacks with a Comprehensive Benchmark

Membership inference attacks pose a significant threat to user privacy in machine learning systems. While numerous attack mechanisms have been proposed in the literature, the
lack of standardized evaluation parameters and metrics has led to inconsistent and even conflicting comparison results. To address this issue and facilitate a systematic analysis of these disparate
findings, we introduce MIBench, a comprehensive benchmark for membership inference attacks. MIBench includes a suite of carefully designed evaluation scenarios and evaluation metrics
to provide a consistent framework for assessing the efficacy of various membership inference techniques. The evaluation scenarios are crafted to encompass four critical factors: intra-dataset distance distribution, inter-sample distance within the target dataset, differential distance analysis, and inference withholding ratio. In total, MIBench includes ten typical evaluation metrics and incorporates 84 distinct evaluation scenarios for each dataset. Using this robust framework, we conducted a thorough comparative analysis of 15 state-of-the-art membership inference attack algorithms across 588 evaluation scenarios, 7 widely adopted datasets, and 7 representative model architectures. Our analysis revealed 83 instances of Conflicting Comparison Results (CCR), providing substantial evidence for the CCR Phenomenon.
We identified two CCR types: Type 1 (single-factor) and Type 2 (dual-factor). The distribution of CCR instances across the four critical factors was: inter-sample distance (40.96%), differential distance (37.35%), inference withholding ratio (19.28%), and intra-dataset distance (2.41%). All codes and evaluations of MIBench are publicly available in the following link1.

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

   The results section consists of three parts: the results of 84 evaluation scenarios (ES), the thresholds at maximum MA of the Risk score and Shapley values attacks and the results of 4 research questions (RQ). And in part I and part III, we identify the evaluation results of 15 state-of-the-art MI attacks by ten evaluation metrics (e.g., attacker-side accuracy, precision, recall, f1-score, FPR, FNR, MA, AUC, TPR @ fixed (low) FPR (T@0.01%F and T@0.1%F), threshold at maximum MA).
   
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

   **9. BlindMI-1CLASS**:  
  
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
   <img width="2167" alt="2023 5 1_LiRA_CH_MNST_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1b712698-9cba-4c84-8fc6-71f609cc5da6">
<img width="2167" alt="2023 5 1_LiRA_CH_MNST_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7301ef9f-dd0a-43d6-8537-7b0e602aed8c">
   **(4) ImageNet**:   
<img width="2167" alt="2023 5 1_LiRA_ImageNet_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d2111bb3-6f7b-46bc-b81c-05af6e546d1a">
<img width="2167" alt="2023 5 1_LiRA_ImageNet_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/01127e6f-ab15-4d5c-a2de-2f7e7c49efe9">
   **(5) Location30**:  
   <img width="2167" alt="2023 5 1_LiRA_Location30_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/2c74bad3-ebb4-407d-8a24-539768892390">
<img width="2167" alt="2023 5 1_LiRA_Location30_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4c8f22f0-0fa7-4c59-ac92-a97682cbf71f">
   **(6) Purchase100**:  
   <img width="2167" alt="2023 5 1_LiRA_Purchase100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/f4f629f0-83ed-4348-a849-32a35ee37665">
<img width="2167" alt="2023 5 1_LiRA_Purchase100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/a2b0f645-40d2-489d-80a7-1712e5fd4bc1">
   **(7) Texas100**: 
   <img width="2167" alt="2023 5 1_LiRA_Texas100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4300f269-0b61-44ea-a52b-8672add4b718">
<img width="2167" alt="2023 5 1_LiRA_Texas100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/fdea58a8-7a70-4cb8-b2ee-de5a5953fbb6">

   **12. Top2+True**:  
  
  **(1) CIFAR100**:
  <img width="2248" alt="2023 5 1_Top2+True_CIFAR100_不同评估场景_实验结果_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3050229a-8762-4db7-8988-a2e803ed6aec">
<img width="2248" alt="2023 5 1_Top2+True_CIFAR100_不同评估场景_实验结果_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6bcb8b81-18ea-45d1-ad88-d3c0cfb47a05">
  **(2) CIFAR10**:
   <img width="2248" alt="2023 5 1_Top2+True_CIFAR10_不同评估场景_实验结果_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d916ad36-cda3-4a58-8ac6-fcb71602d5e0">
<img width="2248" alt="2023 5 1_Top2+True_CIFAR10_不同评估场景_实验结果_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/95df00bf-9fd1-46f4-876e-cffbf1bc9150">
   **(3) CH_MNIST**:
   <img width="2248" alt="2023 5 1_Top2+True_CH_MNST_不同评估场景_实验结果_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d7c21609-d072-4973-ac40-e1d1cd276576">
<img width="2248" alt="2023 5 1_Top2+True_CH_MNST_不同评估场景_实验结果_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/20b959fd-67e3-4486-8fd7-e686b42a3b4b">
   **(4) ImageNet**:   
<img width="2248" alt="2023 5 1_Top2+True_ImageNet_不同评估场景_实验结果_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/40ae1e03-1026-46f0-8f6a-7b566f9cd73c">
<img width="2248" alt="2023 5 1_Top2+True_ImageNet_不同评估场景_实验结果_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c41da7e3-045e-44cc-a4ac-f03c0c076fd4">
   **(5) Location30**:  
   <img width="2248" alt="2023 5 1_Top2+True_Location30_不同评估场景_实验结果_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/639156d2-901c-4994-b0e4-147487553e7a">
<img width="2248" alt="2023 5 1_Top2+True_Location30_不同评估场景_实验结果_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/0c476730-ffd5-49db-8e1b-b5a7cc6daef4">
   **(6) Purchase100**:  
   <img width="2248" alt="2023 5 1_Top2+True_Purchase100_不同评估场景_实验结果_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/abed8819-5976-4658-bd06-8b3c44c5e105">
<img width="2248" alt="2023 5 1_Top2+True_Purchase100_不同评估场景_实验结果_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/208dfd9a-be35-411c-9644-ff9095437d8c">
   **(7) Texas100**: 
   <img width="2248" alt="2023 5 1_Top2+True_Texas100_不同评估场景_实验结果_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/13acb5bd-346c-48e0-956a-40892478c3e1">
<img width="2248" alt="2023 5 1_Top2+True_Texas100_不同评估场景_实验结果_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/95c8adab-f4e3-42ec-9c2e-cbf204f745cb">

   **13. BlindMI-w**:  
  
  **(1) CIFAR100**:
  <img width="2156" alt="2023 5 2_BlinMI-w_CIFAR100_不同评估场景_实验结果_V2_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/0c754d66-75f7-416e-938d-1832091f94af">
<img width="2156" alt="2023 5 2_BlinMI-w_CIFAR100_不同评估场景_实验结果_V2_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/5ee6d948-a6f6-44fa-928c-7425544e4adc">
  **(2) CIFAR10**:
   <img width="2156" alt="2023 5 2_BlinMI-w_CIFAR10_不同评估场景_实验结果_V2_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/8dcf93a5-eecb-4ffd-adf8-63b432ddb3bc">
<img width="2156" alt="2023 5 2_BlinMI-w_CIFAR10_不同评估场景_实验结果_V2_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d7e48ef8-0a1c-414e-a627-e1328f72eaaa">
   **(3) CH_MNIST**:
   <img width="2156" alt="2023 5 2_BlinMI-w_CH_MNST_不同评估场景_实验结果_V2_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/ad119b4c-bbd3-4426-82f6-4920d02f17e4">
<img width="2156" alt="2023 5 2_BlinMI-w_CH_MNST_不同评估场景_实验结果_V2_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/facd591d-bcb8-4072-b41c-4640d9def09f">
   **(4) ImageNet**:   
<img width="2156" alt="2023 5 2_BlinMI-w_ImageNet_不同评估场景_实验结果_V2_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/20c8dfa0-5f78-4590-a359-ca74c40d065e">
<img width="2156" alt="2023 5 2_BlinMI-w_ImageNet_不同评估场景_实验结果_V2_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/30259bf4-b18e-4679-b89a-411ed0c4f330">
   **(5) Location30**:  
   <img width="2156" alt="2023 5 2_BlinMI-w_Location30_不同评估场景_实验结果_V2_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7bc05290-e3e3-4474-b1a3-5df3f59fc764">
<img width="2156" alt="2023 5 2_BlinMI-w_Location30_不同评估场景_实验结果_V2_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/76b8158a-bbe1-4877-a3e3-0677784a9bb1">
   **(6) Purchase100**:  
   <img width="2156" alt="2023 5 2_BlinMI-w_Purchase100_不同评估场景_实验结果_V2_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c55e1670-68f4-4366-8019-3969ad3f829c">
<img width="2156" alt="2023 5 2_BlinMI-w_Purchase100_不同评估场景_实验结果_V2_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/5a109ddd-8065-4ed7-a34d-93a0ebf3fcbf">
   **(7) Texas100**: 
   <img width="2156" alt="2023 5 2_BlinMI-w_Texas100_不同评估场景_实验结果_V2_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/f84ebc1c-4e35-4366-9f91-ec867b2f7cb7">
<img width="2156" alt="2023 5 2_BlinMI-w_Texas100_不同评估场景_实验结果_V2_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/365bf80b-9a73-478a-9922-5b4e42af0b76">

   **14. BlindMI-without**:  
  
  **(1) CIFAR100**:
  <img width="2156" alt="2023 5 2_BlinMI-without_CIFAR100_不同评估场景_实验结果(1)_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c24ac266-96d5-44ea-b505-0a246092623d">
<img width="2156" alt="2023 5 2_BlinMI-without_CIFAR100_不同评估场景_实验结果(1)_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/f981b126-fe1b-4422-aac0-89a22fee8258">
  **(2) CIFAR10**:
   <img width="2156" alt="2023 5 2_BlinMI-without_CIFAR10_不同评估场景_实验结果(1)_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/60bfb4c4-ad88-4a54-a6ac-c69b4269697e">
<img width="2156" alt="2023 5 2_BlinMI-without_CIFAR10_不同评估场景_实验结果(1)_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/f967a70e-f61f-4257-829c-5f25aaaa80cd">
   **(3) CH_MNIST**:
   <img width="2156" alt="2023 5 2_BlinMI-without_CH_MNST_不同评估场景_实验结果(1)_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4fba66ec-f830-475f-a73f-a0610c3f25a8">
<img width="2156" alt="2023 5 2_BlinMI-without_CH_MNST_不同评估场景_实验结果(1)_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/8141d982-9405-4167-9422-82784956dc01">
   **(4) Location30**:   
<img width="2156" alt="2023 5 2_BlinMI-without_Location30_不同评估场景_实验结果(1)_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/8406478c-684e-4216-8132-1940227f7677">
<img width="2156" alt="2023 5 2_BlinMI-without_Location30_不同评估场景_实验结果(1)_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/976a9a5c-d826-4160-89b1-a523485f3b47">
   **(5) Purchase100**:  
   <img width="2156" alt="2023 5 2_BlinMI-without_Purchase100_不同评估场景_实验结果(1)_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d745c99d-622f-436a-915b-4b3b8b692a65">
<img width="2156" alt="2023 5 2_BlinMI-without_Purchase100_不同评估场景_实验结果(1)_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/0541ae5f-4c1d-4230-beb4-d3b54292b383">
   **(6) Texas100**:  
   <img width="2156" alt="2023 5 2_BlinMI-without_Texas100_不同评估场景_实验结果(1)_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d1499447-1114-4936-8603-de638da366ed">
<img width="2156" alt="2023 5 2_BlinMI-without_Texas100_不同评估场景_实验结果(1)_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4c7f3ec0-ae8d-41bd-80d5-3f3c8c06a7d7">

   **15. Loss-Threshold**:  
  
  **(1) CIFAR100**:
  <img width="2156" alt="2023 5 2_Loss-Threshold_CIFAR100_不同评估场景_实验结果_V2_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e29aac59-bf69-4f56-bec6-f1ec685eee2d">
<img width="2156" alt="2023 5 2_Loss-Threshold_CIFAR100_不同评估场景_实验结果_V2_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/08d7067b-005d-4040-b157-c01e7359b68b">
  **(2) CIFAR10**:
   <img width="2156" alt="2023 5 2_Loss-Threshold_CIFAR10_不同评估场景_实验结果_V2_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/dd2c4bc0-2672-4a3c-b895-6d6d2e115530">
<img width="2156" alt="2023 5 2_Loss-Threshold_CIFAR10_不同评估场景_实验结果_V2_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/442857d1-289e-48c6-8d71-cff029637bd9">
   **(3) CH_MNIST**:
   <img width="2156" alt="2023 5 2_Loss-Threshold_CH_MNST_不同评估场景_实验结果_V2_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/aa33ee40-9da5-4b71-830c-87c53267f0ef">
<img width="2156" alt="2023 5 2_Loss-Threshold_CH_MNST_不同评估场景_实验结果_V2_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/625779da-33be-4ec2-839c-f0fa43389daf">
   **(4) ImageNet**:   
<img width="2156" alt="2023 5 2_Loss-Threshold_ImageNet_不同评估场景_实验结果_V2_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1251bb2a-8403-4d11-a1b0-63c9ab6ea23b">
<img width="2156" alt="2023 5 2_Loss-Threshold_ImageNet_不同评估场景_实验结果_V2_08" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/76d0c1a1-9299-4adb-89b4-b1091a42d8fd">
   **(5) Location30**:  
   <img width="2156" alt="2023 5 2_Loss-Threshold_Location30_不同评估场景_实验结果_V2_09" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4851132f-06b2-46f2-b7be-daf3302ccf65">
<img width="2156" alt="2023 5 2_Loss-Threshold_Location30_不同评估场景_实验结果_V2_10" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/e76d1a17-070a-4fae-800c-e10b2135f792">
   **(6) Purchase100**:  
   <img width="2156" alt="2023 5 2_Loss-Threshold_Purchase100_不同评估场景_实验结果_V2_11" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1c6b41c9-8f25-4a4e-b787-2e0170598c8f">
<img width="2156" alt="2023 5 2_Loss-Threshold_Purchase100_不同评估场景_实验结果_V2_12" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/9ba3d22a-8b7c-49d4-81c6-da877bff9d96">
   **(7) Texas100**: 
   <img width="2141" alt="2023 5 2_Loss-Threshold_Texas100_不同评估场景_实验结果_V2_13" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7373cf47-d96d-4c18-9843-eab87f1cc446">
<img width="2141" alt="2023 5 2_Loss-Threshold_Texas100_不同评估场景_实验结果_V2_14" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7430ced6-0171-479f-90bf-8fc9df37e13e">
   
 * Part II: The Thresholds at maximum MA

**1. Risk score attacks**: 

   **(1) CIFAR100**:
   <img width="2179" alt="CIFAR100_Risk score_不同类别_阈值_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/65d7709c-311c-464b-8d53-65a09ef3068a">
   
  **(2) CH_MNIST**:
  <img width="2179" alt="CH_MNST_Risk score_不同类别_阈值_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/2c55dd65-b7f8-4907-918b-83dbbc02d881">
<img width="2179" alt="CH_MNST_Risk score_不同类别_阈值_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/4e74a126-a298-49e0-9db5-4c905e611a2e">

  **(3) ImageNet**:
  <img width="2179" alt="ImageNet_Risk score_不同类别_阈值_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/47e6809a-5617-4d8e-a0bb-837ec4cb845d">
  
  **(4) Location30**:
  <img width="2179" alt="Location30_Risk score_不同类别_阈值_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d921b4b8-48e4-4d5e-b333-62f9a9518a48">
  
  **(5) Purchase100**:
  <img width="2179" alt="Purchase100_Risk score_不同类别_阈值_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3f31521f-b9dc-46f2-8f85-a7886191ec3b">
  
  **(6) Texas100**:
  <img width="2179" alt="Texas100_Risk score_不同类别_阈值_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1a56bf14-c882-4222-a5ff-96f3fe23570e">
  
  **2. Shapley values attacks**: 

   **(1) CIFAR100**: 
   <img width="2179" alt="CIFAR100_Shapley values_不同类别_阈值_01" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/637f1940-6bfe-4de7-85fd-8e0dca2c5e3e">

   **(2) CIFAR10**:
     <img width="2179" alt="CIFAR10_Shapley values_不同类别_阈值_02" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c7319c78-6897-4304-adce-989f701cbb47">
     <img width="2179" alt="CIFAR10_Shapley values_不同类别_阈值_03" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/65f3bd5a-6797-406c-ac17-c46eb044f64f">

   **(3) CH_MNIST**:
   <img width="2179" alt="CH_MNST_Shapley values_不同类别_阈值_04" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/fa1c5caa-bf38-485e-b890-f2d6d53e57f0">
   <img width="2179" alt="CH_MNST_Shapley values_不同类别_阈值_05" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d09afce2-358b-4ce2-a274-223e37dcd3b0">

   **(4) ImageNet**: 
   <img width="2179" alt="ImageNet_Shapley values_不同类别_阈值_06" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7f8acbed-dbc7-41fa-beee-71fae1df1582">

   **(5) Location30**:
   <img width="2179" alt="Location30_Shapley values_不同类别_阈值_07" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3624de18-c8e7-45ca-84c3-84ab26644c91">

   **(6) Purchase100**:
   ![Purchase100_Shapley values_不同类别_阈值_08](https://github.com/MIBench/MIBench.github.io/assets/124696836/8362b639-b135-44ab-9120-316ffc11e244)

   **(7) Texas100**:
  ![Texas100_Shapley values_不同类别_阈值_09](https://github.com/MIBench/MIBench.github.io/assets/124696836/5f496c0b-81e7-4a7f-9cd7-4563b8202db7)

   * Part III: The Results of 4 Research Questions
   
**(1) CIFAR100**:

**RQ1: Effect of Distance Distribution of Data Samples in the Target Dataset**

ES01: CIFAR100_Normal + 2.893 + 0.085 + 20%                       
ES29: CIFAR100_Uniform + 2.893 + 0.085 + 20%                       
ES57: CIFAR100_Bernoulli + 2.893 +0.085 + 20%
<img width="2156" alt="CIFAR100_RQ1" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3315f074-1c63-4eb3-97db-501ca6ee6a28">
<table>
	<tr>
		<td><center><img src=https://github.com/MIBench/MIBench.github.io/assets/124696836/8f22d996-4ae1-4d1c-9bcb-45c6afbed023 />CIFAR100_N_2 893_d1_20%</center></td>
		<td><center><img src=https://github.com/MIBench/MIBench.github.io/assets/124696836/01e45d8e-f8a7-46a1-a158-63c84b463214 />CIFAR100_U_2 893_d1_20%</center></td>
		<td><center><img src=https://github.com/MIBench/MIBench.github.io/assets/124696836/b24f7fd9-e18d-4e88-aa7e-17322c0c8f25 />CIFAR100_B_2 893_d1_20%</center></td>
        </tr>
</table>
	
**RQ2: Effect of Distance between data samples of the Target Dataset**

ES02: CIFAR100_Normal + 2.893 + 0.085 + 40%                        
ES10: CIFAR100_Normal + 3.813 + 0.085 + 40%                       
ES22: CIFAR100_Normal + 4.325 + 0.085 + 40%
<img width="2156" alt="CIFAR100_RQ2" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/ea1e5692-0d7f-4b3b-8fc9-bd444b8028c2">

**RQ3: Effect of Differential Distances between two datasets**

ES03: CIFAR100_Normal + 2.893 + 0.085 + 45%                       
ES05: CIFAR100_Normal + 2.893 + 0.119 + 45%                       
ES07: CIFAR100_Normal + 2.893 + 0.157 + 45%
<img width="2156" alt="CIFAR100_RQ3" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/0e68ea78-bab2-4656-97e3-5aa5f525ff0d">

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES37: CIFAR100_Uniform + 3.813 + 0.085 + 20%                       
ES38: CIFAR100_Uniform + 3.813 + 0.085 + 40%                       
ES39: CIFAR100_Uniform + 3.813 + 0.085 + 45%                        
ES40: CIFAR100_Uniform + 3.813 + 0.085 + 49%
![CIFAR100_RQ4](https://github.com/MIBench/MIBench.github.io/assets/124696836/51c502c9-9134-4140-a0a2-0ecd5bc86629)

**(2) CIFAR10**:

**RQ1: Effect of Distance Distribution of Data Samples in the Target Dataset**

ES13: CIFAR10_Normal + 2.501 + 0.213 + 20%                       
ES41: CIFAR10_Uniform + 2.501 + 0.213 + 20%                       
ES69: CIFAR10_Bernoulli + 2.501 + 0.213 + 20%
<img width="2144" alt="CIFAR10_RQ1" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/bc622880-a0ae-45e9-91cc-557164919c9f">

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES02: CIFAR10_Normal + 1.908 + 0.155 + 40%                       
ES10: CIFAR10_Normal + 2.501 + 0.155 + 40%                       
ES22: CIFAR10_Normal + 3.472 + 0.155 + 40%
<img width="2144" alt="CIFAR10_RQ2" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/8b8fc162-890e-4e41-b2c0-8a0d5e3367fc">
<table>
	<tr>
		<td><center><img src=https://github.com/MIBench/MIBench.github.io/assets/124696836/9a496dbd-9dc8-4109-9a4e-058a4d4aa746 />CIFAR10_N_1 908_d1_40%</center></td>
		<td><center><img src=https://github.com/MIBench/MIBench.github.io/assets/124696836/efba0d0e-e962-4672-a115-5adb5a8aa776 />CIFAR10_N_2 501_d1_40%</center></td>
		<td><center><img src=https://github.com/MIBench/MIBench.github.io/assets/124696836/44758fa8-79a3-4509-8eab-f2ee1b9df295 />CIFAR10_N_3 472_d1_40%</center></td>
        </tr>
</table>

**RQ3: Effect of Differential Distances between two datasets**

ES51: CIFAR10_Uniform + 3.472 + 0.155 + 45%                       
ES53: CIFAR10_Uniform + 3.472 + 0.213 + 45%                        
ES55: CIFAR10_Uniform + 3.472 + 0.291 + 45%
<img width="2144" alt="CIFAR10_RQ3" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/d13defe4-e623-4f52-b369-5d094b0e512d">

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES57: CIFAR10_Bernoulli + 1.908 +0.155 + 20%                        
ES58: CIFAR10_Bernoulli + 1.908 + 0.155 + 40%                       
ES59: CIFAR10_Bernoulli + 1.908 + 0.155 + 45%                       
ES60: CIFAR10_Bernoulli + 1.908 + 0.155 + 49%
![CIFAR10_RQ4](https://github.com/MIBench/MIBench.github.io/assets/124696836/8bca97cb-3668-4a0f-9954-c3e1954421c7)

**(3) CH_MNIST**:

**RQ1: Effect of Distance Distribution of Data Samples in the Target Dataset**

ES21: CH_MNIST_Normal + 1.720 +0.083 + 20%                       
ES49 : CH_MNIST_Uniform + 1.720 +0.083 + 20%                       
ES77: CH_MNIST_Bernoulli + 1.720 +0.083 + 20%
<img width="2168" alt="CH_MNIST_RQ1" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/3ebffda4-1854-4a03-98fa-6f7cd5152fca">

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES04: CH_MNIST_Uniform + 0.954 + 0.108 + 40%                       
ES14: CH_MNIST_Uniform + 1.355 + 0.108 + 40%                       
ES24: CH_MNIST_Uniform + 1.720 + 0.108 + 40%
<img width="2168" alt="CH_MNIST_RQ2" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6222b0ea-39f7-4e26-9f45-cf581d030ce5">

**RQ3: Effect of Differential Distances between two datasets**

ES03: CH_MNIST_Normal + 0.954 + 0.083 + 45%                                              
ES05: CH_MNIST_Normal + 0.954 + 0.108 + 45%                                              
ES07: CH_MNIST_Normal + 0.954 + 0.133 + 45%
<img width="2168" alt="CH_MNIST_RQ3" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/f809c884-cb44-4d82-a592-d09ef4cbfda3">

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES73: CH_MNIST_Bernoulli + 1.355 + 0.133 + 20%                                              
ES74: CH_MNIST_Bernoulli + 1.355 + 0.133 + 40%                                              
ES75: CH_MNIST_Bernoulli + 1.355 + 0.133 + 45%                                              
ES76: CH_MNIST_Bernoulli + 1.355 + 0.133 + 49%
![CH_MNIST_RQ4](https://github.com/MIBench/MIBench.github.io/assets/124696836/0297774d-4fb5-465d-9e90-b3d443ae5df6)


**(4) ImageNet**:

**RQ1: Effect of Distance Distribution of Data Samples in the Target Dataset**

ES02: ImageNet_Normal + 0.934 + 0.046 + 40%                                              
ES30: ImageNet_Uniform + 0.934 + 0.046 + 40%                                              
ES58: ImageNet_Bernoulli + 0.934 + 0.046 + 40%
<img width="2237" alt="ImageNet_RQ1" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/7430c293-94d1-4a24-805a-205be06b0239">

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES34: ImageNet_Uniform + 0.934 + 0.08 + 49%                                               
ES44: ImageNet_Uniform + 1.130 + 0.08 + 49%                                              
ES54: ImageNet_Uniform + 1.388 + 0.08 + 49%
<img width="2237" alt="ImageNet_RQ2" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/c9fdf48b-c4f1-4cce-8de6-6b4b8d88b9b6">

**RQ3: Effect of Differential Distances between two datasets**

ES79: ImageNet_Bernoulli + 1.388 + 0.046 + 45%                                               
ES81: ImageNet_Bernoulli + 1.388 + 0.080 + 45%                                              
ES83: ImageNet_Bernoulli + 1.388 + 0.145 + 45%
<img width="2237" alt="ImageNet_RQ3" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/1a87713f-7463-4274-ac43-2b5c73b087f4">

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES13: ImageNet_Normal + 1.130 + 0.080 + 20%                       
ES14: ImageNet_Normal + 1.130 + 0.080 + 40%                       
ES15: ImageNet_Normal + 1.130 + 0.080 + 45%                       
ES16: ImageNet_Normal + 1.130 + 0.080 + 49%
![ImageNet_RQ4](https://github.com/MIBench/MIBench.github.io/assets/124696836/22ab6a75-6dba-4c04-8c7d-ef4d7a7f4bf4)

**(5) Location30**:

**RQ1: Effect of Distance Distribution of Data Samples in the Target Dataset**

ES01: Location30_Normal + 0.570 + 0.041 + 4%                       
ES29: Location30_Uniform + 0.570 + 0.041 + 4%                       
ES57: Location30_Bernoulli + 0.570 + 0.041 + 4%
<img width="2171" alt="Location30_RQ1" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/b48a6b8b-b231-43f9-8b10-40b8071eacbd">

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES32: Location30_Uniform + 0.57 + 0.076 + 8%                       
ES42: Location30_Uniform + 0.724 + 0.076 + 8%                       
ES52: Location30_Uniform + 0.801 + 0.076 + 8%
<img width="2171" alt="Location30_RQ2" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/fcc1c793-89b1-4ca4-a75b-26eacba540ef">

**RQ3: Effect of Differential Distances between two datasets**

ES23: Location30_Normal + 0.801 + 0.041 + 12%                       
ES25: Location30_Normal + 0.801 + 0.076 + 12%                       
ES27: Location30_Normal + 0.801 + 0.094 + 12%
<img width="2171" alt="Location30_RQ3" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/6ee00849-b39c-4a69-9c7a-2d54ae7495ac">
         
**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES73: Location30_Bernoulli + 0.724 + 0.094 + 4%                       
ES74: Location30_Bernoulli + 0.724 + 0.094 + 8%                       
ES75: Location30_Bernoulli + 0.724 + 0.094 + 12%                       
ES76: Location30_Bernoulli + 0.724 + 0.094 + 16%
![Location30_RQ4](https://github.com/MIBench/MIBench.github.io/assets/124696836/edd058e2-c6d8-43fc-9a7a-2938c5f6c4b9)

**(6) Purchase100**:

**RQ1: Effect of Distance Distribution of Data Samples in the Target Dataset**

ES01: Purchase100_Normal + 0.550 + 0.087 + 2%                       
ES29: Purchase100_Uniform + 0.550 + 0.087 + 2%                       
ES57: Purchase100_Bernoulli + 0.550 + 0.087 + 2%
<img width="2167" alt="Purchase100_RQ1" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/416b4fee-2aae-42a9-9a95-d2896c27527c">

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES04: Purchase100_Normal + 0.550 + 0.110 + 4%                       
ES14: Purchase100_Normal + 0.625 + 0.110 + 4%                       
ES24: Purchase100_Normal + 0.729 + 0.110 + 4%
<img width="2167" alt="Purchase100_RQ2" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/878c72a0-25ef-4ad6-ac88-50eba804899e">

**RQ3: Effect of Differential Distances between two datasets**

ES51: Purchase100_Uniform + 0.729 + 0.087 + 10%                       
ES53: Purchase100_Uniform + 0.729 + 0.110 + 10%                       
ES55: Purchase100_Uniform + 0.729 + 0.156 + 10%
<img width="2167" alt="Purchase100_RQ3" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/99c0c212-2543-49f8-a6f5-adba8ee67deb">

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES65: Purchase100_Bernoulli + 0.625 + 0.087 + 2%                       
ES66: Purchase100_Bernoulli + 0.625 + 0.087 + 4%                       
ES67: Purchase100_Bernoulli + 0.625 + 0.087 + 10%                       
ES68: Purchase100_Bernoulli + 0.625 + 0.087 + 12%
![Purchase100_RQ4](https://github.com/MIBench/MIBench.github.io/assets/124696836/7439772d-8c5f-4a9b-af29-d08cec2d30dd)

**(7) Texas100**:

**RQ1: Effect of Distance Distribution of Data Samples in the Target Dataset**

ES01: Texas100_Normal + 0.530 + 0.038 + 2%                       
ES29: Texas100_Uniform + 0.530 + 0.038 + 2%                       
ES57: Texas100_Bernoulli + 0.530 + 0.038 + 2%
<img width="2179" alt="Texas100_RQ1" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/57e8bd9c-a6db-4343-bf43-9ea12387fe08">

**RQ2: Effect of Distance between data samples of the Target Dataset**

ES02: Texas100_Normal + 0.530 + 0.038 + 4%                       
ES10: Texas100_Normal + 0.641 + 0.038 + 4%                       
ES22: Texas100_Normal + 0.734 + 0.038 + 4%
<img width="2179" alt="Texas100_RQ2" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/fbd31f91-9d32-4cf7-8dba-ab0431adbf0d">
                
**RQ3: Effect of Differential Distances between two datasets**

ES51: Texas100_Uniform + 0.734 + 0.038 + 10%                       
ES53: Texas100_Uniform + 0.734 + 0.073 + 10%                       
ES55: Texas100_Uniform + 0.734 + 0.107 + 10%
<img width="2179" alt="Texas100_RQ3" src="https://github.com/MIBench/MIBench.github.io/assets/124696836/768ea1e8-9f38-444e-93b3-97df0efedf3a">

**RQ4: Effect of the Ratios of the samples that are made no inferences by an MI attack**

ES65: Texas100_Bernoulli + 0.641 + 0.038 + 2%                       
ES66: Texas100_Bernoulli + 0.641 + 0.038 + 4%                       
ES67: Texas100_Bernoulli + 0.641 + 0.038 + 10%                       
ES68: Texas100_Bernoulli + 0.641 + 0.038 + 12%
![Texas100_RQ4](https://github.com/MIBench/MIBench.github.io/assets/124696836/10b5600b-b37a-499f-af41-ed8085a93354)

**Additional Evaluation Results**




