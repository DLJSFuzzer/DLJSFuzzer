# Mutation-Based Deep Learning Framework Testing Method in JavaScript Environment

This is the implementation repository of our *ASE'24* paper: **Mutation-Based Deep Learning Framework Testing Method in JavaScript Environment**.



## 1. Description

In recent years, Deep Learning (DL) applications in JavaScript environment have become increasingly popular. As the infrastructure for DL applications, JavaScript DL frameworks play a crucial role in the development and deployment. It is essential to ensure the quality of JavaScript DL frameworks. However, the bottleneck of limited computational resources in the JavaScript environment brings new challenges to framework testing. Specifically, JavaScript DL frameworks equips with various optimization mechanisms (e.g., cache reuse, inference acceleration) to overcome the bottleneck of limited computational resources. These optimization mechanisms are overlooked by existing methods, resulting in many bugs in JavaScript DL frameworks being missed. To address the above challenges, we propose a mutation-based JavaScript DL framework testing method named DLJSFuzzer. DLJSFuzzer designs 13 tensor mutation rules targeting the cache reuse mechanism to generate test input tensors. Besides, DLJSFuzzer designs eight model mutation rules targeting the inference acceleration mechanism to generate test input models. To evaluate the effectiveness of DLJSFuzzer, we conduct experiments on the most widely-used JavaScript DL framework, TensorFlow.js. The experimental results show that DLJSFuzzer outperforms state-of-the-art methods in both effectiveness and efficiency. DLJSFuzzer successfully detects 21 unique crashes and 126 unique NaN \& Inconsistency bugs. All detected crashes have been reported to the open-source community, with 12 of them already confirmed by developers. Additionally, DLJSFuzzer has improved by over 47% in model generation efficiency and over 91% in bug detection efficiency compared to all baselines.



You can access this repository using the following command:

```shell
git clone https://github.com/DLJSFuzzer/DLJSFuzzer.git
```



## 2. Framework version

We use two common DL frameworks (including ***TensorFlow*** and ***PyTorch***) to aid the differential testing. The tested framework is ***TensorFlow.js***. We conduct the experimental environment as follow:

| TensorFlow | PyTorch | TensorFlow.js |
| :--------: | :-----: | :-----------: |
|   2.9.0    | 1.12.0  |    3.19.0     |



## 3. Environment

**Step 0:** Please install ***anaconda*** and ***Node.js***

**Step 1:** Create a conda environment. Run the following commands.

```sh
conda create -n DLMOSA python=3.9
source activate DLMOSA
pip install tensorflow==2.9.0
pip install torch==1.12.0
pip install keras==2.6.0
pip install sqlalchemy==1.4.32
pip install mysql-connector-python
pip install flask==2.2.2
pip install flask-cors==3.0.10
pip install gevent==22.10.2
pip install tensorflow-estimator==2.9.0
pip install tensorflow-hub==0.12.0
pip install tensorflowjs==3.19.0
pip install Werkzeug==2.2.2
```

**Step 2:** Create a mysql database. Run the following command.

```
CREATE DATABASE TFJSHelper;
```

**Step 3:** Connect the database TFJSHelper. Modify the *username* and *password* in the url of **my_mysql.py** in the **Database** folder to your own. Run **db_helper.py** in the **Database** folder.

**Step 4:** Install Microsoft Edge browser and Google Chrome browser. Clear the  browser's cache, and change the download path of the browsers to the  **Chrome_output_storage** and **Edge_output_storage** folders in the folder **TFJS_output_storage** respectively.

**Step 5:** Change the *absolutePath* in **globalConfig.py** to the current path of the project.

**Step 6:** Modify *Command_chrome_start* and *Command_edge_start* in **globalConfig.py** to the address of the corresponding main programs of the browsers.

**Step 7:** Run the following command in your cmd.

```
npm install http-server -global
```



## 4. File structure

This project contains five folders. The **LEMON-master** folder is the downloaded open source code for LEMON. The **Muffin-main** folder is the downloaded open source code for Muffin. The **Gandalf-main** folder is the downloaded open source code for Gandalf. The **DLJSFuzzer** folder is the source code for our method. The **result** folder is the experimental result data. To know the execution methods of our baselines, please refer to the corresponding research papers. In this document, we will introduce how to run the source code for **DLJSFuzzer**.

In the source code for **DLJSFuzzer**, the folders named **DataStruct, Test, and Method** contain the body for the method. The program entry of the method is **main.py**. Run **main.py** to run DLMOSA after installing the experimental environment.

The folder named **result_analysis** is used in the Evaluation section, which will be explained in the following sections.

The folder named  **dataset** is used for preprocessing the dataset. **DLJSFuzzer** supports six dataset, including MNIST, Fashion-MNIST, CIFAR-10, ImageNet, Sine-Wave and Stock-Price. In addition, **DLJSFuzzer** also supports randomly generating test input tensors. The first three ones can be accessed by [Keras API](https://keras.io/api/datasets/)，while the rest can be access from [OneDrive](https://onedrive.live.com/?authkey=%21ANVR8C2wSN1Rb9M&id=34CB15091B189D3E%211909&cid=34CB15091B189D3E)(`dataset.zip`) provided by LEMON. Please set the variable *dataset* in the **globalConfig.py** in the folder **Datastruct** to specify the dataset.

## 5. Experiments

### 5.1 Main Method

**Step 0:** Move to the **TFJS_Model** folder in cmd, and execute the following command:

```
 http-server --cors
```

**Step 1:** Make sure you are now in the ***conda*** visual environment!

Use the following command to run the experiment according to the configuration:

```shell
python app.py
python clear_helper.py
python main.py
```

The testing results will be stored in `./result.csv`.

If you do not want to reproduce the experiment, experimental results are available in the folder **result**. There are three folders in the folder **result**: 1) Folder **crash_logs** for the logs of all detected crashes. 2) Folder **NaN&inconsistency** for the logs of all detected NaN & Inconsistency bugs. 3) Folder **models** for the models generated by all baselines. (The models generated by our method is recorded in the csv in the folder **NaN&inconsistency** )

### 5.2.1 Evaluation

**Step 1:** Set *file_path* and *result_csv* in all files in the folder **result_analysis**.

**Step 2:** Use the following command:

```shell
python DLJSFuzzer_edit_distance.py
python Gandalf_LEMON_edit_distance.py
python Muffin_edit_distance.py
python Muffin_structure.py
```

**Confirmed crash analysis**

|       root cause       | number | description                                                  |
| :--------------------: | :----: | ------------------------------------------------------------ |
|      cache reuse       |   5    | Three bugs originate from missing parameters during model execution. The bugs occur during cache reuse. Two bugs stem from tensor shape mismatches. The abnormal changes in tensor shape also occur during cache reuse. |
|   implementation bug   |   4    | Two bugs originate from the imperfect handling of implicit type  conversion. Specifically, due to JavaScript's weak typing mechanism,  TensorFlow.js can create tensors without explicitly declaring their data types. TensorFlow.js is expected to perform implicit type conversion by detecting the actual data type of the tensor. However, in operators  like BatchNorm, if the tensor's data type is declared in advance, these  operators only support the default data type (e.g., float). This results in data type mismatches, leading to crashes. The other two bugs stem  from the lack of support for certain tensor data types (e.g., bfloat16)  in some operators like ReduceMean when running in a CPU environment. |
| inference acceleration |   2    | These two bugs originate from the imperfect implementation of the  inference acceleration mechanism. Specifically, due to parameter  misalignment, the parameter settings in the accelerated model become  incorrect, leading to framework bugs. |
|         Other          |   1    | This bug originates from garbled text resulting from encoding and  decoding in the browser. When TensorFlow.js transmits tensors to the  browser, they undergo various file formats. Multiple encodings and  decodings between these file formats cause the data to become garbled, making it fail in loading by the  model. |
|         Total          |   12   | -                                                            |


