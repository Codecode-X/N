# NegCLIP 

> DDL: 2025.5.7 🥇 
> *European Conference on Artificial Intelligence*  😻`欧洲小而美`

### 待解决的问题

当前项目主要是解决 **CLIP无法理解”否定“prompt** 的问题，如下图所示：

<img src=".\notes\images\image-20250312163611184.png" alt="image-20250312163611184" style="zoom:33%;" />

---

### 相关工作

NegationCLIP [1] 是通过**增加带有更多否定标注**的数据集去**微调CLIP模型**，但这种方法**成本较高**，且很容易导致CLIP模型**过拟合**，甚至出现“灾难性遗忘”的模型坍缩问题。

我们希望通过 **仅插入一个模块** 就实现让CLIP能够 **理解”否定“prompt** 。

---

### 核心IDEA

CLIP推理过程是基于对比学习的，即找出与当前图片最相似的文本进行配对，或者是找出与当前文本最相似的图片（分类任务）或者图片块（目标检测或者分割）进行匹配。

然而由于CLIP无法正确处理否定的表述，因此当我的**文本prompt**是 `A woman without glasses in kitchen` ，他可能会给我**定位到** `A woman with glasses in kitchen` 。

所以我希望通过一个**否定prompt提取模块**，将原prompt内容中的**否定内容** `without glasses` 给提取出来作为 **negetive prompt**，保留**肯定内容** `A woman in kitchen` 作为 **positive prompt** 。

两种prompt采用相反的匹配方式处理：

* 对于 **positive prompt**，我们按照**原CLIP的匹配度评估方式**（**相似度越大越好-拉近**）。
* 对于 **negetive prompt** ，我们则采用**相反的匹配度评估方式**（**相似度越小越好-推远**）。
* 最后**综合两个正负匹配度得分**，得到匹配结果。



### 模型结构设计

方案1

<img src=".\notes\images\image-20250314172912177.png" alt="image-20250314172912177" style="zoom: 50%;" />

方案2

<img src=".\notes\images\image-20250314172931664.png" alt="image-20250314172931664" style="zoom:50%;" />

方案3

<img src=".\notes\images\image-20250314172938521.png" alt="image-20250314172938521" style="zoom:50%;" />

<img src=".\notes\images\idea.png" alt="image-20250314172938521" style="zoom:50%;" />

---

### 待办计划💪💪💪  

### 🎈🎈🎈冲击 ECAI 2025.5.7  

>  *European Conference on Artificial Intelligence*  😻`欧洲小而美`

`⛳ 2025.3.15-3.21` 

**任务摘要**：完成token分类器的训练**数据采集**，完成一个**base模型的所有编码**。

1. **数据采集工作**：由于Token分类器的训练需要数据，数据内容包括：

   * 一段包含一个或多个否定的描述性语句（prompt）
   * 该句子中包含的肯定部分内容（postive prompt）
   * 该句子包含的否定部分内容（negetive prompt）

   ```csv
   """id, prompt, postive promptm, negetive prompt"""
   1, She wears a coat not a jacket, She wears a coat, a jacket
   2, The park has trees but no flowers, The park has trees, flowers
   3, A woman without glasses in kitchen, A woman in kitchen, glasses
   ```

   该部分数据可利用GPT等LLM工具或其他工具进行采集，一共需要采集 800 条。`by: 奥利奥 & Ete` 

2. **Token分类器代码实现**：实现一个轻量级的Token分类器，对文本编码器输出的token进行处理，得到postive  token和negetive token。三种token的形状应相同。`by: 染红` 

3. **基于CoOp增强的CLIP模型实现**：实现模型的其他部分，包括训练和测试部分。`by: X` 

`🎈 2025.3.22-3.28` 

**任务摘要**：完成Token分类器的训练，跑通base模型。(待补充....💭💭💭)

`🎈 2025.3.22-3.28` 

**任务摘要**：(待补充....💭💭💭)

`🎈 2025.3.29-4.4` 

**任务摘要**：(待补充....💭💭💭)

`🎈 2025.4.5-4.11` 

**任务摘要**：(待补充....💭💭💭)

----

### Reference

[1] Park J, Lee J, Song J, et al. Know" No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP[J]. arXiv preprint arXiv:2501.10913, 2025.
