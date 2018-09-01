# SSD的MXNet代码分析

参考的代码是：[DSSD-github](git@github.com:MTCloudVision/mxnet-dssd.git)

## MXNet中的一些用于目标检测的工具函数

### 生成锚框先验

函数：*contrib.MultiBoxPrior(data, sizes, ratios, clip, steps, offsets, name, attr, out, **kwargs)*

其中,data是一个Layout为BCHW的思维矩阵，就是用于生成Anchors的Feature map。sizes为锚框的尺寸(Scale)，假设包含n个不同的大小，ratios为锚框的Aspect ratio，假设有m个不同的比例，那么一共生成$wh(n+m-1)$。假设输入图像的高为h宽为w，那么大小为s，比例为r的锚框的尺寸为：

$$\left( ws\sqrt{r}, \frac{ws}{\sqrt{r}}  \right)$$       (1)

在SSD中，每个锚框的位置还有个0.5偏置，用于得到中心。注意，参数sizes中的数值是相对于输入图像的大小而言的，而不是Feature map的大小。

剩下的参数：clip表示是否丢弃out-of-boundary的boxes；steps: Anchor在y和x方向上的步长，类似卷积操作中的Stride，如果是-1那么就自动计算；offsets，候选框的中心位置的offset，如在SSD中，这个值就等于0.5。name表示返回的symbol。

函数的返回值：返回一个(批量大小，锚框个数，4)的矩阵。锚框个数由公式(1)得到，最后一维的4代表Anchor的中心坐标以及长宽。

### 为每个锚框生成标签

函数：*contirb.MultiBoxTarget(anchor, label, cls_pred, overlap_threshold, ignore_label, negative_mining_ratio, minimum_negative_mining_thresh, minimum_negative_samples, variances, name)*

这一步的目的是对每个锚框生成标签，也就是每个框是pos/neg，遮掩neg的ndarray，锚框与真实Bounding box的偏移。

输入的参数中，anchor为上一步生成的Anchors的结果，维度为(批量大小，锚框个数，4)。label为GroundTruth BBox的信息，每个GT Box的label为5-vector，第一个数表示类别，后面四个数表示GT Box的真实位置(中心坐标+长宽)，所以最后数据的形状为(批量大小，BBox个数，5)。BBox个数远大于锚框个数。cls_pred为对每个锚框的分类预测结果。overlap_threshold，当锚框与GT Box的IoU超过这个阈值时才认为这个锚框与GT是匹配的，默认为0.5。negative_mining_ratio表示negative和positive样本之间的最大比值，默认为-1也就是不进行调整。negative_mining_thresh为Negative Mining的阈值。minimum_negative_sample表示最小的Negative样本数。Variance表示Bounding box regression中的权重，用于修改网络预测的BBox相对于Anchor偏移量的作用。也就是[面试必备SSD模型-知乎](https://zhuanlan.zhihu.com/p/43459011)中计算BBox的公式中引入了权重，示例如下(以x为例)：

$$x' = x + variance[0] * dx * Aw$$		(2)

其中，$x'$为BBox的值，x为Anchor的x值，dx为网络BBox Regression的输出，Aw为Anchor的宽。

MXNet中这个函数一共返回三个NDarray，第三个NDArray的形状为(批量大小，锚框个数)分别表示每个锚框的类别，要么是背景0要么是其它类别结果；当为0时，表示该锚框与GT Box的IoU小于0.5。返回的第二个NDArray表示用来遮掩不需要的负类边框，形状为(批量大小，锚框数*4)，乘以4的原因在于每个锚框对应四个位置预测分量。返回的第一个NDArray就表示锚框与真是边界框的偏移量了，只有正类锚框有非0值。

再补充一点GLUON中关于这个函数的解释。

在训练时，每个锚框都表示成一个样本。对每个样本我们需要预测它是否包含我们感兴趣的物体，以及如果包含，那么预测它的真实边界框。在训练前我们首先需要为每个锚框生成标签。这里标签包含两类，第一类对应的真实物体的标号。一个常用的构造方法就是对每个真实的边界框，我们选取一个或多个与其相似的的锚框赋予它们这个真实边界框里的物体标号。具体来说，对一个训练数据中提供的真实边界框，假设其对应物体标号为$i$，我们选取所有与其IoU大于某个阈值的锚框，如果没有这样的锚框，那么我们就选取IoU最大的那个。然后将选中的锚框的物体标号设计成$i+1$。如果一个锚框没有被任何真实边界框选中，即不与任何训练数据中的物体足够重合，那么将赋予标号0，代表只包含背景。称后者为负类锚框，其余则称之为正类。**这一步其实就是对应上面的输出的第三个NDArray，也就是Anchor与GT Box的匹配问题。**

对于正类锚框，我们还需要构造第二类标号，即他们与真实边界框的距离。一个简单的方法是它与真实边界框的坐标差。但因为有图像边缘的限制，这些差值都在-1到1之间，而且分布差异很大，这使得模型预测变得复杂。我们通常会将其进行非线性变换来使得数值上更加均匀来方便模型预测。

### 实际预测

函数：*MultiBoxDetection(cls_prob, loc_pred, anchor, clip, threshold, background_id, num_threshold, force_suppress, variances, num_topk, name, attrr, out, \*\*kwargs)*

这一步跟上一步中的训练过程类似，都是对每个锚框预测其包含的物体类别和与真实边界框的位移。

其输入参数：cls_prob为网络预测的分类结果，loc_pred为偏移量的网络输出结果，anchor为MultiBoxPrior输出的Anchor数据，clip是否去掉out-of-boundary boxes，threshold为判断为正类的阈值，background_id为背景类的id，默认为0；num_threshod为进行NMS的阈值；force_suppress、variances意义与上面相同；nms_topk在进行NMS之前只考虑最大的K个Anchor的预测。

函数的输出数据的形状为(批量大小，锚框个数，6)。也就是每个锚框对应6个数值，第一个数值表示预测的类别，如果是-1表示要么是背景要么是由NMS移除的BBox，第二个数值表示物体属于此类的概率，也就是用于NMS的值了，后面四个数值表示预测的边界框。

简单说一下NMS的过程。对于每个物体类别(不包含背景)，我们先获取每个预测边界框里包含这个类别的概率(也就是上面输出的第二个值)。然后我们找到概率最大的那个边界框，如果其置信度大于某个阈值(threshold参数)，那么保留它到输出，接下来移除所有跟这个边界框的IoU大于某个阈值的边界框。在剩下的边界框里面继续找出**预测概率**最大的边界框，一直重复前面的移除过程(下一个类别)，直到我们遍历保留或这移除每个边界框。

## SSD模型

### SSD中的Head部分

函数：*legacy_conv_act_layer(from_layer, name, num_filter, kernel, pad, stride, act_type, use_batchnorm)*

也就是封装了：Convolution，Activation、BN的代码块，同时返回激活前结果(conv)、激活后结果(relu)。

### 生成所有层的Prior Box及其位置、类别预测结果

函数：*multibox_layer(from_layers, num_filters, topdown_layers, use_predict_module, num_classes, size, ratio, normalizatin, num_channels, clip, interm_layer, steps, use_tdm)*

本部分是SSD头部代码中的重点。用到的MXNet函数：

* *mx.sym.L2Normalization(data, eps, mode, name, attr)*

  对输入的数据进行L2正则化。mode可选['channel', instance, spatial]三种。

  对一维的Normalization，计算过程如下：

  $$out = data / sqrt(sum(data ** 2) + eps)$$

  * mode = channel

    ```Python
    for i in 0 ... N:
        out[:, i, :, ..., :] = data[:, i, :, ..., :] / sqrt(sum(data[:, i, :, ..., :] ** 2) + eps)
    ```

  * mode = instance

    ```Python
    for i in 0...N:
        out[i, :, ..., :] = data[i, :, ..., :] / sqrt(sum(data[i, :, ... , :] ** 2) +  eps)
    ```

    ​

  * mode = spatial

    ```Python
    for dim in 2 ... N:
        for i in 0 ... N:
            out[..., i, ...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)
    ```

* *mx.sym.broadcast_mul()*

函数的输入包括：

* from_layers 表示所有用于生成Prior Box的Feature layers

  等。

函数的输出为：

list of outputs: [loc_preds, cls_preds, anchor_boxes]

* loc_preds: localization regression prediction
* cls_preds: classification prediction
* anchor_boxes: generated anchor boxes

#### 预测Anchor的位置

代码如下：

```Python
	# num_anchors表示每个像素位置的Anchor的数量，也就是aspect + ratio - 1
	num_loc_pred = num_anchors * 4     # 输出的位置，Channel的个数为每个像素的Anchor的数量 * 4(表示4个偏移量)
    bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
    init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
    loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
    stride=(1,1), pad=(1,1), num_filter=num_loc_pred, \           # 经过一次卷积运算，输出的Channel的数量为num_loc_pred，也就是每个像素Anchor的个数 * 4
    name="{}_loc_pred_conv".format(from_name))   # 这一步的输出loc_pred的shape为：(batch_size, num_loc_pred, h, w)， 其中w,h分别对应产生Anchor的Feature Map的宽和长。
    loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))   # 此时loc_pred的shape为(batch_size, h, w, num_loc_pred)，对应的是模型示意图中的Permute操作
    loc_pred = mx.symbol.Flatten(data=loc_pred)   # Flatten得到二维结果，loc_pred现在的shape为(batch_size, h * w * num_loc_pred)，对应的三个步骤中的Flatten操作
    loc_pred_layers.append(loc_pred)   # 每层Feature Map的二维结果作为list的一个元素
```

#### 预测Anchor的分类

```Python
# create class prediction layer
        num_cls_pred = num_anchors * num_classes   # 输出的Channel的个数为：num_anchors * num_classes，也就是每个Anchors产生num_classes个预测，然后每个位置包含num_anchors个Anchors
        bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})   # 设置lr_mult=2.0
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_cls_pred, \
            name="{}_cls_pred_conv".format(from_name))   # 经过一次卷积后，输出的cls_pred的大小为：
        # (batch_size, num_cls_pred, w, h)其中w,h对应当前Feature Map的宽和长
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))   # 同样经历一次Permute,得到的结果的shape为(batch_size, h, w, num_cls_pred)
        cls_pred = mx.symbol.Flatten(data=cls_pred)   # 对应三步操作中的第二步，也就是Flatten
        cls_pred_layers.append(cls_pred)    # 每一层Feature的二维结果作为一个list的元素
```

#### 产生Anchor Prior

这部分对应的是示意图中的第三条路，代码如下：

```Python
 # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])   # Step表示在输入图像(300, 300)上相邻两个Anchor的步长
        else:
            step = '(-1.0, -1.0)'
        anchors = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str,
                                                  clip=clip, name="{}_anchors".format(from_name),
                                                  steps=step)   # 使用MultiBoxPrior生成Anchor框,得到的anchors的大小为:(batch_size, num_anchors, 4), 这里的num_anchors为wh(m+n-1)
        anchors = mx.symbol.Flatten(data=anchors)    # 这里直接Flatten成一个二维数据
        anchor_layers.append(anchors)               # 每层Feature Map生成Anchors作为一个list的element
```

代码如下：

```Python
# create class prediction layer
        num_cls_pred = num_anchors * num_classes
        bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_cls_pred, \
            name="{}_cls_pred_conv".format(from_name))
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)
```

#### 得到预测结果

上一步之后得到三个list:

* list of anchor locations

  形状为(Feature Map的个数，批量大小，fm_i * anchors_num_fm_i * 4 )，其中，后面一项的加和的个数等于Feature Map Level的个数，anchors_num_fm_i表示第fm_i层Feature Map的anchor的个数，也就是该层的长宽w，h乘上每个像素位置的产生的(m+n-1)个Anchors，后面的4表示每个Anchors预测4个偏移量。

* list of anchor predictions

  k

* list of prior anchors

代码如下：

```Python
   loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers), \
        dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")
    anchor_boxes = mx.symbol.Concat(*anchor_layers, \
        num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]
```



### 生成Anchors的Target

在上一步生成Anchor的先验数据后，先验数据包含：list of anchor locations, list of anchor categories predictions, list of prior anchors。

这一步

代码注释详情：

```Python
def multibox_layer(from_layers, num_filters, topdown_layers, use_perdict_module, num_classes, sizes=[.2, .95], ratios=[1], normalization=-1, num_channels=[],
                    clip=False, interm_layer=0, steps=[],use_tdm = False):
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0, \    # 
        "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers), \
        "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers), \
        "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)

    # sum(...) = 1, len(num_channels) = 1
    assert sum(x > 0 for x in normalization) <= len(num_channels), \
        "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"
	### 上面的代码都是对输入的检查，也就是说各个输入的数量要与from_layer中包含的层数一致。
        
    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0

    if use_tdm:
        from_layers = dt.construct_topdown_upsample_layer(from_layers, num_filters)

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer, \
                mode="channel", name="{}_norm".format(from_name))  # mode = channel
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_channels.pop(0), 1, 1),
                init=mx.init.Constant(normalization[k]),    # 初始化为20
                attr={'__wd_mult__': '0.1'})
            # lhs.shape = 1, rhs.shape = from_layer.shape = (batch, 512, 38, 38)
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
            from_layers[k] = from_layer  # 相当于对conv4_3的输出扩大了20倍！？

    from_layers = dt.construct_dssd_deconv_layer(from_layers, num_filters, topdown_layers, use_perdict_module)   # 

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3), \
                stride=(1,1), pad=(1,1), num_filter=interm_layer, \
                name="{}_inter_conv".format(from_name))
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu", \
                name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) -1 + len(ratio)   # 每个像素位置的Anchor的数量是:aspect + ratio - 1

        # create location prediction layer
        num_loc_pred = num_anchors * 4
        bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_loc_pred, \
            name="{}_loc_pred_conv".format(from_name))
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        # create class prediction layer
        num_cls_pred = num_anchors * num_classes
        bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3), \
            stride=(1,1), pad=(1,1), num_filter=num_cls_pred, \
            name="{}_cls_pred_conv".format(from_name))
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)

        # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'
        anchors = mx.contrib.symbol.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str,
                                                  clip=clip, name="{}_anchors".format(from_name),
                                                  steps=step)
        anchors = mx.symbol.Flatten(data=anchors)
        anchor_layers.append(anchors)

    loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers), \
        dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers), \
        dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")
    anchor_boxes = mx.symbol.Concat(*anchor_layers, \
        num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]
```





### 完整的带注释代码

```python
# 输入的类别数是20，nms的阈值是0.5，force_suppress=False不同的类别之间不进行Suppression，nms_topk对前400个Box进行NMS
def get_symbol_train(num_classes=20, nms_thresh=0.5, force_suppress=False,
                     nms_topk=400, **kwargs):
    data = mx.symbol.Variable(name="data")      # 跟label是输入的两个Symbol数据
    label = mx.symbol.Variable(name="label")
     # group 1， Input: 300 * 300 * 3
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2, subsample 2 now, 64 * 150 * 150
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3, subsample 4 now, 128 * 75 * 75
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
        pooling_convention="full", name="pool3")   # Pooling的convention为full，会选择ceil而不是floor计算输出的大小
    # group 4, subsample 8 now, 38 * 38 * 256，为什么不是37*37*256，见上面pooling_convention选项
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")     # Subsample 8, 38 * 38 * 512
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5, subsample 16 now, 19 * 19 * 512
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(3, 3), stride=(1, 1),   #这里stride为1
        pad=(1,1), name="pool5")
    # group 6, subsample 16 now, 19 * 19 * 512
    conv6 = mx.symbol.Convolution(    # Dilate output shape: o = \left\rfloor \frac{i + 2p - k - (k-1)(d-1)}{s} + 1 = (19 + 12 - 3 - 2 * 5) / 1 + 1 = 19
        data=pool5, kernel=(3, 3), pad=(6, 6), dilate=(6, 6),
        num_filter=1024, name="conv6")
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7, input shape: 1024 * 19 * 19
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="conv7")
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu", name="relu7")
    # 此时，输出的relu7大小是：1024 * 19 * 19
    ### ssd extra layers ###   输入大小为1024 * 19 * 19
    conv8_1, relu8_1 = legacy_conv_act_layer(relu7, "8_1", 256, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv8_2, relu8_2 = legacy_conv_act_layer(relu8_1, "8_2", 512, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)    # 输出大小：512 * 10 * 10
    conv9_1, relu9_1 = legacy_conv_act_layer(relu8_2, "9_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv9_2, relu9_2 = legacy_conv_act_layer(relu9_1, "9_2", 256, kernel=(3,3), pad=(1,1), \
        stride=(2,2), act_type="relu", use_batchnorm=False)    # 输出大小：256 * 5 * 5
    conv10_1, relu10_1 = legacy_conv_act_layer(relu9_2, "10_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv10_2, relu10_2 = legacy_conv_act_layer(relu10_1, "10_2", 256, kernel=(3,3), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)    # 输出大小为：256 × 3 × 3
    conv11_1, relu11_1 = legacy_conv_act_layer(relu10_2, "11_1", 128, kernel=(1,1), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)
    conv11_2, relu11_2 = legacy_conv_act_layer(relu11_1, "11_2", 256, kernel=(3,3), pad=(0,0), \
        stride=(1,1), act_type="relu", use_batchnorm=False)    # 最后输出大小为: 256 × 1 × 1
    
    # 设置一些必要的参数：
    # specific parameters for VGG16 network
    from_layers = [relu4_3, relu7, relu8_2, relu9_2, relu10_2, relu11_2]   # 选择多个Feature Level，生成Prior Boxes
    sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]   # 不同层的Scale
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \          # 不同层的Ratios
        [1,2,.5], [1,2,.5]]                 # Anchor的个数为:size的个数 + Ratio的个数 - 1，为了减少Anchor的个数
    normalizations = [20, -1, -1, -1, -1, -1]     # 只对relu4_3进行normalization
    steps = [ x / 300.0 for x in [8, 16, 32, 64, 100, 300]]   # 相对于输入图像(300, 300)而言的相邻Prior Box之间的步长！       
    num_channels = [512]
    
    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_channels, clip=False, interm_layer=0, steps=steps)  # clip = False，不去除out-of-boundary boxes。
    
    # mx.symbol.contrib.MultiBoxTarget
    # 原型参见MXNet文档。用于生成Bounding Box的训练目标数据，
    tmp = mx.symbol.contrib.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    
     loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.symbol.contrib.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out                       # 返回网络输出，包括cls_prob, loc_loss, cls_label, det
```



