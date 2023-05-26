
## SAMScore: A Semantic Structural Similarity Metric for Image Translation Evaluation

[Yunxiang Li](https://www.yunxiangli.top/), Meixu Chen, Wenxuan Yang, Kai Wang, [Jun Ma](https://scholar.google.com/citations?hl=zh-CN&user=bW1UV4IAAAAJ), [Alan C. Bovik](https://www.ece.utexas.edu/people/faculty/alan-bovik), [You Zhang](https://profiles.utsouthwestern.edu/profile/161901/you-zhang.html). 

<div>
    <a href="https://arxiv.org/pdf/2305.15367.pdf"><img src="https://info.arxiv.org/brand/images/brand-logo-primary.jpg" alt="Arxiv"></a>
    <a href="https://colab.research.google.com/github/Kent0n-Li/SAMScore/blob/main/SAMScore.ipynb#scrollTo=mCidlfXu88UY"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </div>
  <br>
  
  
<img src='imgs/overview.jpg' width=1200>

### Quick start

Run `pip install samscore`. The following Python code is all you need.

```python
import argparse
import samscore

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p1','--path_source', type=str, default='./imgs/n02381460_20_real.png')
parser.add_argument('-p0','--path_generated', type=str, default='./imgs/n02381460_20_fake.png')
parser.add_argument('-v','--version', type=str, default='1.0')

opt = parser.parse_args()

## Initializing the model
SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_l" )
samscore_result = SAMScore_Evaluation(source_image_path=opt.path_source,  generated_image_path=opt.path_generated)

print('SAMScore: %.4f'%samscore_result)
```


## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{zhang2018perceptual,
  title={SAMScore: A Semantic Structural Similarity Metric for Image Translation Evaluation},
  author={Yunxiang Li, Meixu Chen, Wenxuan Yang, Kai Wang, Jun Ma, Alan C. Bovik, You Zhang},
  booktitle={arxiv},
  year={2023}
}
```


