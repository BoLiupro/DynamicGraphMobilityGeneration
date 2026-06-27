# 1. Dynamic Graph建立
## 背景
我现在已经完成了数据的初步处理,数据在data/shanghai文件夹下，主要是mobiliyt.csv和location.csv.现在我要根据mobility建立dynamic graph。我的dynamic graph是每个时间步下面对应一个graph，graph中有user node和location node,有location-location edge和user-location edge。location-location edge是依据location之间的distance和流通量一起确定的，距离越近同时连通量越大，那么loc-loc edge的权重越大。此外如果当下user处于某个location，那么就存在一个loc-user edge，权重为1.
## 任务
写一个python代码，实现dynamic graph的建立
1. 计算loc-loc edge权重，设计适当的方案计算连通量、distance和综合的边权重.为了保证稀疏性，可以用合理的方式设计一个阈值来控制变的是否存在
2. 统计user-loc edge
3. 建立每个时刻的dynamic graph
## 约束
+ 使用python代码
+ 代码逻辑清楚，注释详细
+ 运行过程要有过程进度打印
## 输出
1. python代码保存在/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/data/shanghai/dyanmic_graph
2. mobility_train.csv的保存在/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/data/shanghai/dyanmic_graph/train,test的也类似
3. 每一个时刻的用dynamic_graph_ddddhh个命名格式来保存。每一天每一个hour的都是一个独立的dynamic graph


# 2. Pattern挖掘
## 背景
我已经完成了数据处理和基于mobility的dynamic graph建立，现在要从train数据集中对所有日期的数据进行统计，挖掘一些关于dynamic graphs演变的pattern，包括每个时间步上各个location node的population，edge对应的flow（每个点在每条边上的flow in 和 flow out），并基于evoving clsutering的社区提取算发提取语义社区、挖掘并统计motifs
## 任务
写一个python，在/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/util目录下，由于为了严格防止数据泄露，只使用0601~0614的train的数据（因为后面测试集打算用0615的数据）
1. 统计每个时刻dynamic graph的population分布情况和flow的分布情况
2. 一句user-loc edge边的变化情况，利用evolving clustering，就是在基于leiden的社区语义提取算法加上time smoothing。提取出communites，一个community应该是包括location nodes和user nodes的。并记录communites的组成情况
3. 给予0601-0614所有train天的数据，在每个community内部来统计mobility行为的motifs，包括每个时间段在某个poi区域停留多长的时间（可以划分bins，例如1h~2h、3~6h,6~12h,12~18h,18~24h）,A-B移动，A-B-A移动，A-B-C移动，A-B-C-B移动，A-B-C-B-A移动这几种。所有的不是按照location_id，而是按照POI类型（每个location最大的3个POI），移动的距离也按照distance bins来划分（例如0~1km，1~2km）。所有偶的bins的划分需要依据数据来合理设定。最终统计一个motifs的转移矩阵。同时统计也统计location0 最开始0h的时候那个motifs概率最大。
## 限制
+ 使用python，处理过程打印进度，代码移动注释清楚
+ motifs的数量和粒度等，要根据数据来确定
+ 写完代码后运行查看结果
## 输出
1. 提取到的结果保存在/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/data/shanghai/dyanmic_graph/extracted_pattern

# 2.1 Pattern挖掘修改
## 背景
已经对dynamic pattern进行初步挖掘，但是有一些问题需要修改
## 任务
修改代码：
1. 不是变成336个时间步，而是不同天的同一个时段的可以统计到一起
2. flow需要具体到每一条边是怎么变的的，包括每个边从每条边怎么flow in flow out的
3. 分析完0601~0614之后，communities数量应该是不变的、一致的
4. motifs中大的stay最多是24h

# 2.2 Pattern挖掘修改2
## 背景与任务
已完成第一步的修改，但是communities的划分还有问题。不是说静态的只跑一次然后动态分配users，而是使用evoving clustering，结合时间中user-location连接的变化来进行time smoothing，这样划分出来的是考虑了时间变化但相对稳定的社区划分，而不会每个时间步社区的数量都不一样。此外400个locations只有6个communites太少了，感觉粒度要再细一点。原来生成的json等结果可以先删除掉。

# 3.1 方法框架建立——user agent初始化
## 背景
我现在已经完成了数据集处理和dynamic graph pattern的提取，现在要准备开始搭建我的方法框架了。我的任务是要做一个human mobility generation的任务，根据从dynamic graph当中提取到的知识先验，利用multi llm-agents的框架来实现生成和refelction迭代完善的过程。具体而言，这个multi-agent框架包含user agents和location agents对应每一个user和每一个location，以及一个特reflection agent。生成开始后按照时间步，先分别从user agent的角度（user根据自己的intent决定是否移动以及下一步前往哪个location）的location agent的角度（location根据知识先验判断当前人数是否太少或者太多，以及依据flow先验从哪些其他location吸引什么样的user过来）生成两套mobility方案和对应的generated dynamic graph，然后把生成的dynamic graph交给reflection agent整理总结并给出调整signal，再进行refine,多次迭代后形成最终稳定的mobility，然后进入下一个时刻，直到完成一天的时间步。现在需要一步一步搭建这个framework，首先先开始搭建user agent，包括user初始化、daily plan生成、
## 任务
构建framework的代码，现在先完成user的相关部分
1. 首先所有的user都从0h开始生成并且每个user有一个start location。这个start location就是从mobility_test里面得到的，我们对test里面的user进行生成，但是我们只是知道他们的start location。
2. 在之前的先验知识中，train当中的user应该已经被分配了communites，我需要根据这些user的communites和start location的对应概率关系。基于目标user的start location，为每个需要生成的user分配一个community
3. 对于每个user，利用其community中的motifs transition关系，生成daily plan
4. 在同一个community当中，利用daily plan的相似度，设定一个阈值来为每个user寻找co-mobility user，即有伴随或者相同出行节奏关系的其他users
## 限制
+ 抵用python，代码逻辑清晰，注释清晰
+ 框架流程都基于LangchainLangGraph来控制
+ 所有user的plan的时间长度都等于24h
+ 数据结果用json格式保留
## 输出
1. user agent代码都输出在/model文件夹下
2. 主流程控制的代码可以写一个main.py
3. 生成的user profile等等初始化信息，放到/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/output/user_init

