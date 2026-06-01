\# YouthDepressioninScales



\## 中文说明



本项目用于整理青少年抑郁相关问卷数据、医生病历数据，并生成后续检出率分析、不同方式诊断情况分析，以及机器学习建模所需的数据文件。

\## 1. 文件与目录说明

\### DATA 文件夹

`RAWDATA`  

接收到的未经任何处理的原始数据。

`Rawdata1.xlsx`  

整理命名规则后的问卷数据，是预处理流程的主要输入文件之一。

`Complains\_YoungClients.xls`  

医生原始病历数据，是预处理流程的主要输入文件之一。

\---

\### R / Rmd 文件

`预处理.rmd`  

用于处理 `Rawdata1.xlsx` 与 `Complains\_YoungClients.xls` 的主要预处理代码文件。该文件完成 ID 核查、重复 ID 处理、病历清洗、诊断标签生成、机器学习数据构建、文本版本生成以及症状表述命中统计等流程。

\---



\### OUTPUT 文件夹

`OUTPUT` 文件夹用于保存预处理及后续分析生成的中间文件和最终输出文件，包括：

\- `ID\_mismatch\_check.xlsx`

\- `Raw\_output.xlsx`

\- `Complains\_output.xlsx`

\- `Complains\_output\_depression\_pure.xlsx`

\- `Complains\_output\_non\_depression.xlsx`

\- `Questionnaire\_With\_Diagnosis\_Label\_Pure.xlsx`

\- `Questionnaire\_Depression\_Pure.xlsx`

\- `Questionnaire\_Non\_Depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_pure\_depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_text\_M3\_pure\_depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_text\_M4\_pure\_depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_text\_M5\_pure\_depression.xlsx`

\- `ML\_dataset\_phrase\_hits\_nonoverlap\_pure.xlsx`

\- `Phrase\_hit\_overall\_summary\_nonoverlap\_pure.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M3\_pure.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M4\_pure.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M3\_pure\_with\_fisher.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M4\_pure\_with\_fisher.xlsx`

\- `Sample\_level\_phrase\_hits\_4scales\_pure.xlsx`

\- `Group\_level\_phrase\_hit\_summary\_4scales\_pure.xlsx`

\- `Phrase\_level\_group\_comparison\_4scales\_pure.xlsx`

\- `Phrase\_level\_group\_comparison\_4scales\_pure\_with\_fisher.xlsx`



\---

\## 2. 数据预处理流程



预处理主要针对 `Rawdata1.xlsx` 和 `Complains\_YoungClients.xls` 两个数据文件进行。两个数据表均可能存在重复 ID，因此需要在后续分析前确保每个 ID 只保留一条最合适的记录。

\---

\## 2.1 ID 不匹配核查

在正式合并和筛选前，代码首先对两个数据表中的 ID 进行核查。

具体步骤包括：

1\. 读取 `Rawdata1.xlsx` 和 `Complains\_YoungClients.xls`。

2\. 清理 ID 字段中的多余空格。

3\. 提取 Rawdata1 中的唯一有效 ID。

4\. 提取 Complains 中的唯一有效门诊 ID。

5\. 统计：

&#x20;  - Rawdata1 的唯一有效 ID 数；

&#x20;  - Complains 的唯一有效 ID 数；

&#x20;  - Rawdata1 中存在但 Complains 中不存在的 ID；

&#x20;  - Complains 中存在但 Rawdata1 中不存在的 ID；

&#x20;  - 这些不匹配 ID 对应的原始行数。



输出文件为：

`OUTPUT/ID\_mismatch\_check.xlsx`

该文件用于检查两个数据源之间的 ID 对应情况。

\---

\## 2.2 时间字段解析

为了对重复 ID 进行时间匹配，预处理代码会解析两个数据表中的时间字段。

Rawdata1 使用字段：

`提交时间`

该字段可能包含两种格式：

\- 英文格式，例如：`Wed May 25 14:04:35 2022`

\- 数字格式，例如：`2023-06-17 10:11:28`

代码会将其统一解析为：

`提交时间\_parsed`

Complains 使用字段：

`就诊日期`

该字段会被解析为：

`就诊日期\_parsed`

两个时间字段均以 `Asia/Shanghai` 时区进行处理。

\---

\## 2.3 重复 ID 的匹配与筛选规则

\### 1. 两个数据表均出现的 ID

当某个 ID 同时出现在 `Rawdata1.xlsx` 与 `Complains\_YoungClients.xls` 中，且两个数据表中都可能有多条记录时：

处理规则：

1\. 对该 ID 在 Rawdata1 中的每条记录与 Complains 中的每条记录进行两两组合。

2\. 计算 Raw 提交时间与 Complains 就诊日期之间的绝对时间差。

3\. 保留时间差最小的一对记录。

4\. 删除该 ID 的其他 Raw 记录和其他 Complains 记录。

Raw 表使用字段：

`提交时间\_parsed`

Complains 表使用字段：

`就诊日期\_parsed`

该规则保证共同 ID 在两个数据文件中都只保留时间上最接近、最匹配的一条记录。

\---

\### 2. 仅在 Rawdata1 中出现的 ID

当某个 ID 只出现在 `Rawdata1.xlsx` 中，而不出现在 `Complains\_YoungClients.xls` 中时：

处理规则：

1\. 保留该 ID 在 Rawdata1 中提交时间最早的一条记录。

2\. 删除该 ID 的其他记录。

该规则保证 Rawdata1 独有 ID 也只保留一条代表性记录。

\---

\### 3. 仅在 Complains\_YoungClients 中出现的 ID

当某个 ID 只出现在 `Complains\_YoungClients.xls` 中，而不出现在 `Rawdata1.xlsx` 中时：

处理规则：

1\. 保留该 ID 在 Complains 中就诊日期最早的一条记录。

2\. 删除该 ID 的其他记录。

该规则保证 Complains 独有 ID 也只保留一条代表性记录。

\--

\## 2.4 去重后的主要输出文件



完成 ID 匹配和去重后，代码会分别输出问卷数据和病历数据。



`Raw\_output.xlsx`  

去重后的问卷数据。每个 Raw ID 只保留一条记录。该文件也是后续检出率分析载入的主要数据文件。



`Complains\_output.xlsx`  

去重后的病历数据。每个门诊 ID 只保留一条记录。



\---



\## 3. 病历数据清洗与诊断标签构建



在生成 `Complains\_output.xlsx` 后，代码进一步清洗病历数据，并根据诊断字段构建纯抑郁组和非抑郁组。



\---



\## 3.1 病历清洗规则



清洗规则包括：



1\. 仅保留 `现病史` 非空的记录。

2\. 根据 `出生日期` 和 `就诊日期` 计算年龄。

3\. 仅保留年龄在 7 至 18 岁之间的个体。

4\. 删除主诉中包含 `代诊` 的记录。

5\. 清理并标准化 `诊断` 字段，包括：

&#x20;  - 删除换行符、制表符等特殊字符；

&#x20;  - 合并多余空格；

&#x20;  - 删除诊断字段末尾多余分号。



\---



\## 3.2 纯抑郁组定义



纯抑郁组只保留诊断字段完全等于以下标签的记录：



\- `抑郁状态`

\- `抑郁障碍`



生成文件：



`Complains\_output\_depression\_pure.xlsx`



\---



\## 3.3 非抑郁组定义



非抑郁组保留诊断字段中不包含 `抑郁` 的记录。



生成文件：



`Complains\_output\_non\_depression.xlsx`



\---



\## 4. 问卷数据诊断标签生成



代码将病历数据中的诊断分组标签合并回问卷数据。



生成的标签字段为：



`Depression\_status`



取值包括：



\- `Depression`

\- `Non-depression`



输出文件包括：



`Questionnaire\_With\_Diagnosis\_Label\_Pure.xlsx`  

包含诊断标签的完整问卷数据。



`Questionnaire\_Depression\_Pure.xlsx`  

纯抑郁组对应的问卷数据。



`Questionnaire\_Non\_Depression.xlsx`  

非抑郁组对应的问卷数据。



\---



\## 5. 机器学习数据构建



机器学习数据基于纯抑郁组和非抑郁组构建。



\---



\## 5.1 样本筛选



建模数据仅保留同时存在于病历数据和问卷数据中的个体。



其中：



\- 纯抑郁组标签为 `1`

\- 非抑郁组标签为 `0`



\---



\## 5.2 1:1 平衡抽样



为了构建平衡分类数据集，代码以纯抑郁组样本量为基准，从非抑郁组中随机抽取相同数量的样本。



随机种子为：



`20240211`



\---



\## 5.3 问卷计分



代码计算以下量表总分：



\- DSRSC

\- PHQ

\- CDI

\- DASS depression subscale



其中，部分 CDI 和 DSRSC 条目进行了反向计分。



CDI 反向计分条目包括：



`CDI2`, `CDI5`, `CDI7`, `CDI8`, `CDI10`, `CDI11`, `CDI13`, `CDI15`, `CDI16`, `CDI18`, `CDI21`, `CDI24`, `CDI25`



DSRSC 反向计分条目包括：



`DSRSC3`, `DSRSC5`, `DSRSC6`, `DSRSC10`, `DSRSC14`, `DSRSC15`, `DSRSC17`, `DSRSC18`



DASS depression subscale 条目包括：



`DASS3`, `DASS5`, `DASS10`, `DASS13`, `DASS16`, `DASS17`, `DASS21`



\---



\## 5.4 5-fold 交叉验证划分



代码根据标签生成分层 5-fold 划分，保证每个 fold 中的抑郁组和非抑郁组比例相对均衡。



随机种子为：



`20240211`



最终机器学习数据输出为：



`ML\_dataset\_balanced\_5fold\_pure\_depression.xlsx`



该文件包含：



\- ID

\- history\_text

\- label

\- total\_DSRSC

\- total\_PHQ

\- total\_CDI

\- total\_DASS

\- fold



\---



\## 6. M3、M4、M5 文本版本生成



在机器学习数据基础上，代码进一步生成不同文本删除版本，用于测试不同症状表述删除策略对模型表现的影响。



\---



\## 6.1 M3 文本版本



M3 使用指定的 M3 删除表述列表，对 `history\_text` 中对应表述进行删除。



输入删除表述文件：



`delete\_for\_M3\_augmented\_pure.txt`



输出文件：



`ML\_dataset\_balanced\_5fold\_text\_M3\_pure\_depression.xlsx`



\---



\## 6.2 M4 文本版本



M4 使用指定的 M4 删除表述列表，对 `history\_text` 中对应表述进行删除。



输入删除表述文件：



`delete\_for\_M4\_augmented\_pure.txt`



输出文件：



`ML\_dataset\_balanced\_5fold\_text\_M4\_pure\_depression.xlsx`



\---



\## 6.3 M5 文本版本



M5 删除所有人工整理出的症状表述。



输入删除表述文件：



`delete\_for\_M5\_augmented\_pure.txt`



输出文件：



`ML\_dataset\_balanced\_5fold\_text\_M5\_pure\_depression.xlsx`



\---



\## 6.4 文本清理规则



删除表述后，代码会进一步清理文本，包括：



1\. 删除换行符、制表符和多余空格。

2\. 合并连续标点符号。

3\. 删除文本首尾多余标点。

4\. 如果删除后文本为空，则标记为：



`\[NO\_RELEVANT\_TEXT]`



\---



\## 7. 症状表述命中统计



代码还对不同症状表述列表在病历文本中的命中情况进行统计。



\---



\## 7.1 非重叠命中规则



为避免短表述被长表述重复计数，代码采用非重叠命中策略：



1\. 先按表述长度从长到短排序。

2\. 如果长表述已命中，则用遮罩字符替换该表述。

3\. 后续短表述不会再从已遮罩文本中重复命中。

4\. 每条文本记录：

&#x20;  - 是否命中任意表述；

&#x20;  - 命中多少个不同表述；

&#x20;  - 具体命中的表述。



\---



\## 7.2 M3 / M4 表述命中统计



输出文件包括：



`ML\_dataset\_phrase\_hits\_nonoverlap\_pure.xlsx`  

样本级别的 M3 / M4 命中结果。



`Phrase\_hit\_overall\_summary\_nonoverlap\_pure.xlsx`  

按抑郁组和非抑郁组汇总的整体命中率。



`Phrase\_group\_comparison\_nonoverlap\_M3\_pure.xlsx`  

M3 单个表述在抑郁组和非抑郁组中的命中差异。



`Phrase\_group\_comparison\_nonoverlap\_M4\_pure.xlsx`  

M4 单个表述在抑郁组和非抑郁组中的命中差异。



`Phrase\_group\_comparison\_nonoverlap\_M3\_pure\_with\_fisher.xlsx`  

M3 单个表述的 Fisher 精确检验结果，并进行 BH 校正。



`Phrase\_group\_comparison\_nonoverlap\_M4\_pure\_with\_fisher.xlsx`  

M4 单个表述的 Fisher 精确检验结果，并进行 BH 校正。



\---



\## 7.3 四个问卷已覆盖表述命中统计



代码还统计四个问卷已覆盖症状表述在病历文本中的命中情况。



输入文件：



`phrases\_measured\_by\_4scales\_pure.txt`



输出文件包括：



`Sample\_level\_phrase\_hits\_4scales\_pure.xlsx`  

样本级别的四问卷覆盖表述命中结果。



`Group\_level\_phrase\_hit\_summary\_4scales\_pure.xlsx`  

组水平命中率汇总。



`Phrase\_level\_group\_comparison\_4scales\_pure.xlsx`  

单个表述在抑郁组和非抑郁组之间的命中差异。



`Phrase\_level\_group\_comparison\_4scales\_pure\_with\_fisher.xlsx`  

单个表述的 Fisher 精确检验结果，并进行 BH 校正。



\---



\## 8. 后续分析



去重后的问卷数据文件：



`Raw\_output.xlsx`



可作为后续分析的主要输入文件，包括：



\- 检出率分析；

\- 问卷总分计算；

\- 机器学习相关分析。



机器学习分析主要使用：



`ML\_dataset\_balanced\_5fold\_pure\_depression.xlsx`



以及 M3、M4、M5 三个文本删除版本。



\---



\# English Version



This project is used to organize questionnaire data and clinical complaint records related to adolescent depression. It generates cleaned datasets for detection-rate analysis, Exploratory Graph Analysis (EGA), and machine-learning analyses.



\---



\## 1. Files and Directories



\### DATA folder



`RAWDATA`  

The raw data received without any preprocessing.



`RAWDATA1`  

The data after revising and standardizing variable naming conventions.



`Rawdata1.xlsx`  

The cleaned questionnaire dataset with standardized variable names. This is one of the main input files for preprocessing.



`Complains\_YoungClients.xls`  

The original clinical complaint dataset from doctors. This is another main input file for preprocessing.



\---



\### R / Rmd files



`Detection rate.rmd`  

R Markdown file for detection-rate analysis.



`EGA.rmd`  

R Markdown file for EGA, namely Exploratory Graph Analysis.



`预处理.rmd`  

The main preprocessing R Markdown file for processing `Rawdata1.xlsx` and `Complains\_YoungClients.xls`. This file performs ID checking, duplicate-ID handling, clinical-record cleaning, diagnosis-label construction, machine-learning dataset construction, text-version generation, and symptom-phrase hit analysis.



\---



\### OUTPUT folder



The `OUTPUT` folder stores intermediate and final output files generated during preprocessing and analysis, including:



\- `ID\_mismatch\_check.xlsx`

\- `Raw\_output.xlsx`

\- `Complains\_output.xlsx`

\- `Complains\_output\_depression\_pure.xlsx`

\- `Complains\_output\_non\_depression.xlsx`

\- `Questionnaire\_With\_Diagnosis\_Label\_Pure.xlsx`

\- `Questionnaire\_Depression\_Pure.xlsx`

\- `Questionnaire\_Non\_Depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_pure\_depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_text\_M3\_pure\_depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_text\_M4\_pure\_depression.xlsx`

\- `ML\_dataset\_balanced\_5fold\_text\_M5\_pure\_depression.xlsx`

\- `ML\_dataset\_phrase\_hits\_nonoverlap\_pure.xlsx`

\- `Phrase\_hit\_overall\_summary\_nonoverlap\_pure.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M3\_pure.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M4\_pure.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M3\_pure\_with\_fisher.xlsx`

\- `Phrase\_group\_comparison\_nonoverlap\_M4\_pure\_with\_fisher.xlsx`

\- `Sample\_level\_phrase\_hits\_4scales\_pure.xlsx`

\- `Group\_level\_phrase\_hit\_summary\_4scales\_pure.xlsx`

\- `Phrase\_level\_group\_comparison\_4scales\_pure.xlsx`

\- `Phrase\_level\_group\_comparison\_4scales\_pure\_with\_fisher.xlsx`



\---



\## 2. Data Preprocessing Workflow



The preprocessing workflow mainly processes `Rawdata1.xlsx` and `Complains\_YoungClients.xls`. Both datasets may contain repeated IDs. Therefore, before downstream analyses, each ID is reduced to one most appropriate record.



\---



\## 2.1 ID mismatch check



Before formal merging and filtering, the code first checks the ID correspondence between the two datasets.



The steps include:



1\. Read `Rawdata1.xlsx` and `Complains\_YoungClients.xls`.

2\. Trim extra spaces in ID fields.

3\. Extract unique valid IDs from Rawdata1.

4\. Extract unique valid outpatient IDs from Complains.

5\. Summarize:

&#x20;  - the number of unique valid IDs in Rawdata1;

&#x20;  - the number of unique valid IDs in Complains;

&#x20;  - IDs present in Rawdata1 but absent from Complains;

&#x20;  - IDs present in Complains but absent from Rawdata1;

&#x20;  - the number of original rows corresponding to these unmatched IDs.



Output file:



`OUTPUT/ID\_mismatch\_check.xlsx`



This file is used to inspect the ID correspondence between the two data sources.



\---



\## 2.2 Timestamp parsing



To match repeated IDs by time, the preprocessing code parses timestamp fields from both datasets.



Rawdata1 uses the field:



`提交时间`



This field may contain two formats:



\- English-style format, for example: `Wed May 25 14:04:35 2022`

\- Numeric format, for example: `2023-06-17 10:11:28`



The field is parsed into:



`提交时间\_parsed`



Complains uses the field:



`就诊日期`



This field is parsed into:



`就诊日期\_parsed`



Both timestamp fields are processed using the `Asia/Shanghai` time zone.



\---



\## 2.3 Matching and filtering rules for repeated IDs



\### 1. IDs appearing in both datasets



When an ID appears in both `Rawdata1.xlsx` and `Complains\_YoungClients.xls`, and both datasets may contain multiple records for that ID:



Rules:



1\. Generate all possible pairs between Rawdata1 records and Complains records for that ID.

2\. Calculate the absolute time difference between Raw submission time and Complains visit date.

3\. Retain the pair with the smallest time difference.

4\. Remove all other Raw and Complains records for that ID.



Raw timestamp field:



`提交时间\_parsed`



Complains timestamp field:



`就诊日期\_parsed`



This rule ensures that each common ID is represented by the temporally closest and most appropriate pair of records.



\---



\### 2. IDs appearing only in Rawdata1



When an ID appears only in `Rawdata1.xlsx` but not in `Complains\_YoungClients.xls`:



Rules:



1\. Retain the record with the earliest submission time in Rawdata1.

2\. Remove all other records for that ID.



This rule ensures that each Raw-only ID is represented by one representative record.



\---



\### 3. IDs appearing only in Complains\_YoungClients



When an ID appears only in `Complains\_YoungClients.xls` but not in `Rawdata1.xlsx`:



Rules:



1\. Retain the record with the earliest visit date in Complains.

2\. Remove all other records for that ID.



This rule ensures that each Complains-only ID is represented by one representative record.



\---



\## 2.4 Main output files after deduplication



After ID matching and deduplication, the code outputs cleaned questionnaire and clinical-record datasets.



`Raw\_output.xlsx`  

The deduplicated questionnaire dataset. Each Raw ID is retained only once. This file is also the main input file for later detection-rate analysis and EGA.



`Complains\_output.xlsx`  

The deduplicated clinical complaint dataset. Each outpatient ID is retained only once.



\---



\## 3. Clinical Record Cleaning and Diagnosis Label Construction



After generating `Complains\_output.xlsx`, the code further cleans the clinical records and constructs pure-depression and non-depression groups based on the diagnosis field.



\---



\## 3.1 Clinical record cleaning rules



The cleaning rules include:



1\. Keep only records with non-empty `现病史`.

2\. Calculate age using `出生日期` and `就诊日期`.

3\. Keep only individuals aged 7 to 18 years.

4\. Remove records whose chief complaint contains `代诊`.

5\. Clean and standardize the `诊断` field, including:

&#x20;  - removing line breaks, tabs, and other special characters;

&#x20;  - squishing redundant spaces;

&#x20;  - removing redundant semicolons at the end of the diagnosis field.



\---



\## 3.2 Definition of the pure-depression group



The pure-depression group only includes records whose diagnosis field exactly matches one of the following labels:



\- `抑郁状态`

\- `抑郁障碍`



Output file:



`Complains\_output\_depression\_pure.xlsx`



\---



\## 3.3 Definition of the non-depression group



The non-depression group includes records whose diagnosis field does not contain `抑郁`.



Output file:



`Complains\_output\_non\_depression.xlsx`



\---



\## 4. Diagnosis Labels for Questionnaire Data



The diagnosis labels derived from the clinical data are merged back into the questionnaire data.



The generated label field is:



`Depression\_status`



Possible values include:



\- `Depression`

\- `Non-depression`



Output files include:



`Questionnaire\_With\_Diagnosis\_Label\_Pure.xlsx`  

The complete questionnaire dataset with diagnosis labels.



`Questionnaire\_Depression\_Pure.xlsx`  

Questionnaire data for the pure-depression group.



`Questionnaire\_Non\_Depression.xlsx`  

Questionnaire data for the non-depression group.



\---



\## 5. Machine-Learning Dataset Construction



The machine-learning dataset is constructed using the pure-depression and non-depression groups.



\---



\## 5.1 Sample selection



Only individuals present in both the clinical-record dataset and the questionnaire dataset are retained.



Labels are defined as:



\- pure-depression group: `1`

\- non-depression group: `0`



\---



\## 5.2 1:1 balanced sampling



To construct a balanced classification dataset, the sample size of the pure-depression group is used as the reference. The same number of samples is randomly selected from the non-depression group.



Random seed:



`20240211`



\---



\## 5.3 Questionnaire scoring



The code calculates total scores for the following scales:



\- DSRSC

\- PHQ

\- CDI

\- DASS depression subscale



Some CDI and DSRSC items are reverse-scored.



CDI reverse-scored items:



`CDI2`, `CDI5`, `CDI7`, `CDI8`, `CDI10`, `CDI11`, `CDI13`, `CDI15`, `CDI16`, `CDI18`, `CDI21`, `CDI24`, `CDI25`



DSRSC reverse-scored items:



`DSRSC3`, `DSRSC5`, `DSRSC6`, `DSRSC10`, `DSRSC14`, `DSRSC15`, `DSRSC17`, `DSRSC18`



DASS depression subscale items:



`DASS3`, `DASS5`, `DASS10`, `DASS13`, `DASS16`, `DASS17`, `DASS21`



\---



\## 5.4 5-fold cross-validation split



The code generates stratified 5-fold assignments based on the class labels, so that the proportions of depression and non-depression samples are relatively balanced across folds.



Random seed:



`20240211`



Final machine-learning dataset:



`ML\_dataset\_balanced\_5fold\_pure\_depression.xlsx`



This file contains:



\- ID

\- history\_text

\- label

\- total\_DSRSC

\- total\_PHQ

\- total\_CDI

\- total\_DASS

\- fold



\---



\## 6. Generation of M3, M4, and M5 Text Versions



Based on the machine-learning dataset, the code generates different text-deletion versions to evaluate how different symptom-phrase deletion strategies affect model performance.



\---



\## 6.1 M3 text version



M3 removes phrases listed in the specified M3 deletion list from `history\_text`.



Input deletion list:



`delete\_for\_M3\_augmented\_pure.txt`



Output file:



`ML\_dataset\_balanced\_5fold\_text\_M3\_pure\_depression.xlsx`



\---



\## 6.2 M4 text version



M4 removes phrases listed in the specified M4 deletion list from `history\_text`.



Input deletion list:



`delete\_for\_M4\_augmented\_pure.txt`



Output file:



`ML\_dataset\_balanced\_5fold\_text\_M4\_pure\_depression.xlsx`



\---



\## 6.3 M5 text version



M5 removes all manually curated symptom expressions.



Input deletion list:



`delete\_for\_M5\_augmented\_pure.txt`



Output file:



`ML\_dataset\_balanced\_5fold\_text\_M5\_pure\_depression.xlsx`



\---



\## 6.4 Text cleaning rules



After phrase deletion, the code further cleans the text by:



1\. Removing line breaks, tabs, and redundant spaces.

2\. Merging repeated punctuation marks.

3\. Removing redundant punctuation at the beginning and end of the text.

4\. Marking empty text as:



`\[NO\_RELEVANT\_TEXT]`



\---



\## 7. Symptom-Phrase Hit Analysis



The code also calculates how often different symptom-phrase lists are detected in the clinical-history text.



\---



\## 7.1 Non-overlapping hit rule



To avoid double-counting short phrases that are part of longer phrases, the code uses a non-overlapping hit strategy:



1\. Sort phrases from longest to shortest.

2\. Once a longer phrase is detected, replace it with masking characters.

3\. Shorter phrases are not allowed to match the already masked part.

4\. For each text record, the code records:

&#x20;  - whether any phrase is detected;

&#x20;  - how many distinct phrases are detected;

&#x20;  - which phrases are detected.



\---



\## 7.2 M3 / M4 phrase-hit analysis



Output files include:



`ML\_dataset\_phrase\_hits\_nonoverlap\_pure.xlsx`  

Sample-level M3 / M4 phrase-hit results.



`Phrase\_hit\_overall\_summary\_nonoverlap\_pure.xlsx`  

Overall phrase-hit summary by depression and non-depression groups.



`Phrase\_group\_comparison\_nonoverlap\_M3\_pure.xlsx`  

Group comparison of individual M3 phrases.



`Phrase\_group\_comparison\_nonoverlap\_M4\_pure.xlsx`  

Group comparison of individual M4 phrases.



`Phrase\_group\_comparison\_nonoverlap\_M3\_pure\_with\_fisher.xlsx`  

Fisher's exact test results for individual M3 phrases, with BH correction.



`Phrase\_group\_comparison\_nonoverlap\_M4\_pure\_with\_fisher.xlsx`  

Fisher's exact test results for individual M4 phrases, with BH correction.



\---



\## 7.3 Phrase-hit analysis for symptoms covered by four scales



The code also analyzes the clinical-text hit rates of symptom expressions covered by the four questionnaire scales.



Input file:



`phrases\_measured\_by\_4scales\_pure.txt`



Output files include:



`Sample\_level\_phrase\_hits\_4scales\_pure.xlsx`  

Sample-level phrase-hit results for the four-scale-covered phrases.



`Group\_level\_phrase\_hit\_summary\_4scales\_pure.xlsx`  

Group-level phrase-hit summary.



`Phrase\_level\_group\_comparison\_4scales\_pure.xlsx`  

Group comparison of individual phrases.



`Phrase\_level\_group\_comparison\_4scales\_pure\_with\_fisher.xlsx`  

Fisher's exact test results for individual phrases, with BH correction.



\---



\## 8. Downstream Analysis



The deduplicated questionnaire data file:



`Raw\_output.xlsx`



can be used as the main input for downstream analyses, including:



\- detection-rate analysis;

\- questionnaire total-score calculation;

\- machine-learning-related analyses.



The main machine-learning dataset is:



`ML\_dataset\_balanced\_5fold\_pure\_depression.xlsx`



The M3, M4, and M5 text-deletion versions can be used for additional modeling and sensitivity analyses.

