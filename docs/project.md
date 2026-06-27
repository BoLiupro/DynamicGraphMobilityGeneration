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

# 3.2 方法框架建立——user agent推理
## 背景
我现在已经基本完成user agent的初始化了，现在要进一步完善user agent的mobility推理功能。这个推理的逻辑大概是这样的，到了每个时间步，每个user都要依据自己的daily plan来决定是move还是stay，如果move的话就需要结合spatial graivty scores（基于距离和poi的吸引力分数）、daily plan purpose(daily plan当中移动是为了到一个什么样的区域)、当前location的flow out（flow out更大对应的候选location有很多的可能去）。这些因素最终都用一个LLM来实现推理和最终决策。
## 任务
1. 我觉得现在的user_agent.py更像是user_init.py，而user_graph.py更像是user_agent.py。改一下文件命名
2. 改完了之后，继续完善user_agent的流程
3. 写一个spatial_graivity_search函数在/util/common中，这个函数的输入是distance和purpose，distance就是user如果move行为distance label，poi对应move的poi目标。利用引力模型，依据train数据得到一个合适的参数，计算distance限定范围内的所有location对于user的吸引力，取分最高的k个做为候选。k是一个超参，写到config中以便后续修改。
4. config当中配置langgraph调用llm的api，key="sk-g1F9WeWL1ovJ6ZuHcLCl5nNQRbPELOYTii2n1pBpT6a8ZIW4",域名接口="https://api.openai-proxy.org"以及模型选择也配置在config
5. 所有的prompt模板在/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/util/prompt.py。写一个prompt模板，给定身份、任务、input（attractive scores, daily plan,候选位置的flow out【可以为0】）、输出格式要求（json，包括决定next location id和reason，长度不要过长）
6. 写一个解析llm推理结果的函数在/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/util/parser.py
## 限制
+ 抵用python，代码逻辑清晰，注释清晰
+ 框架流程都基于LangchainLangGraph来控制
+ 提供一个暂时的单个user的测试接口，而不是在还没稳定的时候就跑所有user的
+ 关键的配置、参数都要写到config里面来进行控制
+ 过程要有信息打印
## 输出
1. user_agent.py代码完善
2. prompt.py, parser.py
3. 测试接口代码/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/test.py

# 3.3 方法框架建立——location agent推理
## 背景
我在搭建我的方法框架，目前已经基本完成了user agent的部分，准备开始弄location agent了。location agent是从“location能够吸引什么样的人来"这个角度出发，结合population的分布、flow in的分布情况、，推理location应该有什么user来。最终也会每个时间步形成一个mobility grapn方案，最终user agent形成的和location agent形成的一起交给refelction agent进行后续处理。现在先开始搭建location agent的流程。
## 任务
写location agent的相关代码
1. 每个location有一个agent,记录该location的poi信息、edge信息、以及各种pattern知识先验。
2. 每个时间步，首先要确定哪些location agent需要进行推理。因为很多时候只需要和move行为相关的或者周边的location进行推理就可以。通过知识先验中的flow in来确定哪些location这个时刻对于user的吸引是最有影响的或者说哪些flow模式是主要的显著的。总而言之，每个时间步根据flow in来找到top m个最有影响的location进行推理。m是重要参数，要写在configs中。
3. 被选中的location需要推理哪些user到这个位置来。需要的信息除了locationpoi、当前location的该时刻population和flow in先验（大概会吸引多少比重的user来，通过什么purpose的users，并且分别从哪些location吸引过来），还包括和这个location有边关联的其他location上的所有的user的plan（因为可能有的user个人intent是不移动的，但是站在location的角度移动过来的太少了所以需要移动）、co-mobility users信息（如果有相邻的location上有都要移动的user并且他们是co-mobility user，那么很可能他们是要在同一个location汇合）。
4. 把这些context输入整理成prompt，交给llm进行推理，最终llm给出一个当前时间步移动到该location的user集合
5. 当所有被选中的location agent完成了推理之后，对mobility进行整理，确定所有的stay和move，得到location agent角度下的mobility dynamic graph。
## 约束
+ 抵用python，代码逻辑清晰，注释清晰
+ 框架流程都基于LangchainLangGraph来控制
+ 提供一个暂时的单个user的测试接口，而不是在还没稳定的时候就跑所有user的
+ 关键的配置、参数都要写到config里面来进行控制
+ 过程要有信息打印
## 输出
1. location_agent.py
2. prompt.py和parser.py完善
3. 测试接口代码/Users/liubo/Desktop/HNU/Research/KDD2026/code/DynamicGraphMobilityGeneration/test.py
4. 其他的代码（如果需要）

# 3.3.2 Location agent修改
## 背景和任务
现在已经完成了初步的location agent的搭建，现在要做一些修改：
1. 挑选top m 个location不是按照总天数中的edge_flow_mean，而是应该是这个时间步下的flow的数量来选择；
2. location保留最主要的多个poi来选择相应的user with purpose，而不是仅仅只看location的第一个poi
3. test测试时得到的打印信息中，[1/1] loc 135 (Sports & Fitness)
    pop=8.1  flow_in=0.714  neighbors=2  co_groups=0这里，我觉得neighber应说明是上一时间步处于该location的user还是相连的location。

# 3.3.3 Location Agent修改
## 背景和任务
现在已经完成了初步的location agent的搭建，但是在最终node_compile_mobility_graph中可能会有一些冲突的问题，现在还要做一些修改：
1. 如果最后合并graph的时候出现了冲突（例如同一个user可能被多个location吸引），按照下面的方式解决：尽量让co-mobility user汇合、优先满足flow in吸引力更大的一方、相等的情况则随机分配