# Obesity-Related Breast Cancer Prediction
This project presents a simple regularised multilayer perceptron trained over the Breast Cancer Coimbra Dataset (Patrício et al., 2018) that is capable of obtaining **more than 98% accuracy** for the lower bound of the 95% confidence interval of the ROCAUC.

In short, the classifier being presented has the capability of reliably detecting breast cancer given simple blood samples. While the multilayer perceptron method is obviously not novel, its application over the chosen dataset is (at the time of writing).

This work took place in fulfilment for the Introduction to Neural Networks module in 2019, taught by Dr. Simon O'Keefe. Said module was a component of my final year studying the Computer Science bachelor's programme at the University of York in the UK. I was awarded a high first class honours score for my work. The full report detailing the findings can be seen under [_report.pdf_](report.pdf).

Dependencies
---
All project dependencies are listed below:
<p align="center">
    <img src="https://user-images.githubusercontent.com/17494044/59552804-502b1800-8f83-11e9-98bf-32da328cc524.png">
</p>
The code may run correctly if packages with more recent or older versions are installed, but no testing has been conducted to guarantee such.

Running
---
In short: run the [_mlp.py_](mlp.py) file to train and test the classifier. Below, we provide a more detailed description of the remaining contents.

The dataset is instantiated directly in code in the form of a dictionary. It can be found in the [_load_data.py_](load_data.py) script.

The [_mlp.py_](mlp.py) script contains the instantiated model and can be further tuned according to personal discretion. Function calls can also be modified to produce graphs and figures (off by default).

By running [_mlp.py_](mlp.py), the classifier is trained and validated with Monte Carlo Cross Validation, as per the paper by Patrício et al., 2018. Within the [*mccv_keras.py*](mccv_keras.py) script, which is called by [_mlp.py_](mlp.py), we implement our cross validation mechanism exactly as per the [_svm.py_](svm.py) file. This is important because by running [_svm.py_](svm.py) we reproduce the original experiment from Patrício et al., 2018 with very similar figures, thus proving that the reimplemented validation mechanism works exactly the same as the original, seeing as the original implementation was carried out in the R programming language.

Again, for more details, read the [report](report.pdf). 

Bibliography
---
M. Patrício et al., “Using Resistin, glucose, age and BMI to predict the presence of breast cancer,” BMC Cancer, vol. 18, no. 1, pp. 1–8, 2018.

J. Crisóstomo et al., “Hyperresistinemia and metabolic dysregulation: a risky crosstalk in obese breast cancer,” Endocrine, vol. 53, no. 2, pp. 433–442, Aug. 2016.

S. O’Keefe, “Learning with an MLP,” INNS. University of York, 2018.

Citation
---
P. H. M. Wigderowitz, “Introduction to Neural Networks Open Assessment, 2018/2019,” Dept. of Comp. Science, Univ. of York, York, UK, 2019.

License
---
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
