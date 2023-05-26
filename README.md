
## SAMScore: A Semantic Structural Similarity Metric for Image Translation Evaluation

[Yunxiang Li](https://www.yunxiangli.top/), Meixu Chen, Wenxuan Yang, Kai Wang, [Jun Ma](https://scholar.google.com/citations?hl=zh-CN&user=bW1UV4IAAAAJ), [Alan C. Bovik](https://www.ece.utexas.edu/people/faculty/alan-bovik), [You Zhang](https://profiles.utsouthwestern.edu/profile/161901/you-zhang.html). 

<div>
    <a href="https://arxiv.org/pdf/2305.15367.pdf"><img src="https://info.arxiv.org/brand/images/brand-logo-primary.jpg" alt="Arxiv" width=85></a> 
      <br>
    <a href="https://colab.research.google.com/github/Kent0n-Li/SAMScore/blob/main/SAMScore.ipynb#scrollTo=mCidlfXu88UY"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </div>

  
  
<img src='imgs/overview.jpg' width=1200>

### Quick start

Run `pip install samscore`.
```python
pip install samscore
pip install git+https://github.com/facebookresearch/segment-anything.git
```

The following Python code is all you need.
```python
import requests
import os
import samscore

def download_image(url, save_path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful

    with open(save_path, 'wb') as file:
        file.write(response.content)
os.makedirs('imgs', exist_ok=True)
# Example usage
image_url = 'https://i.ibb.co/yFFg5pn/n02381460-20-real.png'
save_location = 'imgs/real.png'
download_image(image_url, save_location)

image_url = 'https://i.ibb.co/GCQ2jQy/n02381460-20-fake.png'
save_location = 'imgs/fake.png'
download_image(image_url, save_location)

## Initializing the model
SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_b" )
samscore_result = SAMScore_Evaluation.evaluation_from_path(source_image_path='imgs/real.png',  generated_image_path='imgs/fake.png')

print('SAMScore: %.4f'%samscore_result)
```


## Citation

If you find this repository useful for your research, please use the following.

```
@inproceedings{li2023samscore,
  title={SAMScore: A Semantic Structural Similarity Metric for Image Translation Evaluation},
  author={Yunxiang Li, Meixu Chen, Wenxuan Yang, Kai Wang, Jun Ma, Alan C. Bovik, You Zhang},
  booktitle={arxiv},
  year={2023}
}
```


