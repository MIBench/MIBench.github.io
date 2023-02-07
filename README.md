# MIBench.github.io
MIBench: A Comprehensive Benchmark for Membership Inference Attacks

MIBench is a comprehensive benchmark for comparing different MI attacks, which consists not only the evaluation metric module, but also the evaluation scenario module. And we design the evaluation scenarios from four perspectives: data distribution, the distance between members and nonmembers in the target dataset, the differential distance between two datasets (i.e., the target dataset and a generated dataset with only nonmembers), and the ratio of the samples that are made no inferences by an MI attack. The evaluation metric module consists of six typical evaluation metrics (e.g., precision, recall, f1-score, false positive rate (FPR), 
false negative rate (FNR), membership advantage (MA)). We have identified three principles for the proposed “comparing different MI attacks” methodology, and we have designed and implemented the MIBench benchmark with 84 evaluation scenarios for each dataset. In total, we have used our benchmark to fairly and systematically compare 13 state-of-the-art MI attack algorithms across 588 evaluation scenarios, and these evaluation scenarios cover 7 widely used datasets and 7 representative types of models.

	MI attacks:
	NN_attack
	Loss-Threshold
	Lable-only
	Top3-NN attack
	Top1-Threshold
	BlindMI-Diff 
	Top2+True
	Privacy Risk Scores
	Shapley Values
	Positive Predictive Value 
	Calibrated Score
	Distillation-based Thre.
	Likelihood ratio attack
	Datasets: CH_MNIST, CIFAR10, CIFAR100, ImageNet, Location30, 
Purchase100, ImageNet, Texas100

	Models: MLP, StandDNN, VGG16, VGG19, ResNet50, ResNet101, 
DenseNet121


	Requirements:
You can run the following script to configurate necessary environment
sh ./sh/install.sh

	Usage:
   Please first to make a folder for record, all experiment results with save to record folder as default. And make folder for data to put supported datasets.
   XXX  XXX

	Attack:
This is a demo script of running NN_attack on CIFAR100.
python ./attack/NN_attack.py --yaml_path ../config/attack/NN/CIFAR100.yaml --dataset CIFAR100 --dataset_path ../data --save_folder_name CIFAR100_0_1

	Selected attacks:
![Selected attacks](https://user-images.githubusercontent.com/124696836/217288040-c38bb11e-6b8a-48ae-bc1b-1c7639e886c3.png)


