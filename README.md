# MeaKAM (Under Review)
✨✨✨This is the official implementation of A Novel Multi-Graph-Structure Guided Human Lower-Limb Physiological Load Estimation Method Based on Multi-Inertial-Sensors Fusion During Walking.

# Abstract
💻💻💻Wearable technology has advanced sensor-based human physiological information measurement, offering broad application prospects in health monitoring. Knee Adduction Moment (KAM) is always used as a measurement of lower-limb physiological load. Traditional musculoskeletal calculation methods  rely on collecting joint kinematics data from pressure tables and optical motion capture system. In contrast, wearable-based deep-learning methods for KAM estimation offer a more cost-effective solution. However, they still face challenges in achieving high measurement accuracy and stability. In this work, we focus on developing and validating a robust estimator for a key biomechanical marker (KAM) that can serve as a basis for downstream translational studies. We propose a deep-learning KAM measurement method (MeaKAM) based on multi-inertial-sensors fusion, which consists of four core modules. The Multi-Granularity Encoder (MGE) module captures multi-granularity temporal features. The Gait-Guided Graph Learning (GGL) Module constructs three graph structures to explicitly model the dependencies between sensors, orientations, and time steps. Finally, we introduce the Dual-Granularity Denoising Enhancement (DDE) module to improve step-wise diffusion refinement. Meanwhile, three multi-task learning strategies are used to enhance the generalization performance of MeaKAM. Extensive experiments on several real datasets show that our framework MeaKAM achieves state-of-the-art compared with counterpart estimation methods. MeaKAM achieved 0.29% BW×BH and 1.47% BW×BH in mean-square-error (MSE) with Dataset IMU-Phone and IMU-KAM, obtaining a 5.44% improvement over the SOTA, and achieving 1.95×10-4 in Squared Deviation jitter with Dataset IMU-KAM, obtaining a 39.76% improvement over the SOTA. Extensive Experiments including ablation, explainability,  calculation cost, additional lower limb indicator experiment, and configuration experiments demonstrate the wide applications of MeaKAM in lower limb motion analysis scenarios.
![image](docs/pipeline.png)
# Environment Layout Setup
<img src="docs/layout.png" width="48%" style="margin-right:2%">
<img src="docs/environment.png" width="48%">

# Comparison Results
<img src="docs/Performance_1.png" width="32%">
<img src="docs/Performance_2.png" width="32%">
<img src="docs/Performance_3.png" width="32%">

# Ablation Results
<img src="docs/Ablation.png" width="60%" style="margin-left:20%">

# Code Introduction
We provide the script to train and validate the end-to-end MeaKAM for the IMU-KAM dataset. Please modify your dataset path and run:
```
python main.py
```

# Data
🚀🚀🚀 We are currently in the process of requesting permission to publicly release the full dataset. In the meantime, we have provided three sample subjects to illustrate the data construction and format. If you need additional examples, please contact fzh_sjtu@sjtu.edu.cn, and we can provide sample data from six subjects.


# Contact
📩📩📩 For questions or suggestions, please contact [**Zehui Feng**](mailto:fzh_sjtu@sjtu.edu.cn).
