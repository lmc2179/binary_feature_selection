# binary_feature_selection
information theoretic feature selection for classifiers with binary features

# The metrics

From the summary of the literature in [Tang, Kay, He 2016](https://arxiv.org/pdf/1602.02850.pdf); more detailed citations forthcoming

## Mutual Information

![img](https://latex.codecogs.com/gif.latex?MI(t_k,c_i)=log\left(\frac{\mathbb{P}(t_k,c_i)}{\mathbb{P}(t_k)\mathbb{P}(c_i)}\right)) 

## Cross entropy for text

![img](https://latex.codecogs.com/gif.latex?CET(t_k,c_i)=\mathbb{P}(t_k,c_i)log\left(\frac{\mathbb{P}(t_k,c_i)}{\mathbb{P}(t_k)\mathbb{P}(c_i)}\right))

## Information Gain

![img](https://latex.codecogs.com/gif.latex?IG(t_k,c_i)=CET(t_k,c_i)+CET(\overline{t_k},c_i))

## Chi-square statistic

![img](https://latex.codecogs.com/gif.latex?\chi^2(t_k,c_i)=\frac{[\mathbb{P}(t_k,c_i)\mathbb{P}(\overline{t_k},\overline{c_i})-\mathbb{P}(t_k,\overline{c_i})\mathbb{P}(\overline{t_k},c_i)]^2}{\mathbb{P}(t_k,c_i)\mathbb{P}(\overline{t_k},\overline{c_i})\mathbb{P}(t_k,\overline{c_i})\mathbb{P}(\overline{t_k},c_i)})

## GSS Coefficient

![img](https://latex.codecogs.com/gif.latex?GSS(t_k,c_i)=\mathbb{P}(t_k,c_i)\mathbb{P}(\overline{t_k},\overline{c_i})-\mathbb{P}(t_k,\overline{c_i})\mathbb{P}(\overline{t_k},c_i))