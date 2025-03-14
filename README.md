# NegCLIP

### 待解决的问题

当前项目主要是解决 **CLIP无法理解”否定“prompt** 的问题，如下图所示：

<img src="\notes\images\image-20250312163611184.png" alt="image-20250312163611184" style="zoom:33%;" />

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

![image-20250314172912177](C:\Users\xiaoj\AppData\Roaming\Typora\typora-user-images\image-20250314172912177.png)

方案2

![image-20250314172931664](C:\Users\xiaoj\AppData\Roaming\Typora\typora-user-images\image-20250314172931664.png)

方案3

![image-20250314172938521](C:\Users\xiaoj\AppData\Roaming\Typora\typora-user-images\image-20250314172938521.png)

---

### Reference

[1] Park J, Lee J, Song J, et al. Know" No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP[J]. arXiv preprint arXiv:2501.10913, 2025.