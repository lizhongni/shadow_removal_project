## Introduction
[DC-ShadowNet: Single-Image Hard and Soft Shadow Removal]

## Prerequisites
```
git clone https://github.com/Dabai93/shadow_removal_project.git
cd shadow_removal_project
conda create -n shadow python=3.7
conda activate shadow
pip3 install -r requirements.txt
```

## Datasets
1. SRD [Train](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view)|[BaiduPan](https://pan.baidu.com/s/1mj3BoRQ), [Test](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view).
[Shadow Masks](https://github.com/vinthony/ghost-free-shadow-removal)

2. AISTD|ISTD+ [[link]](https://github.com/cvlab-stonybrook/SID)

3. ISTD [[link]](https://drive.google.com/file/d/1I0qw-65KBA6np8vIZzO6oeiOvcDBttAY/view)

4. USR: Unpaired Shadow Removal Dataset [[link]](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view)

5. LRSS: Soft Shadow Dataset [[link]](http://visual.cs.ucl.ac.uk/pubs/softshadows/)<br>
   The LRSS dataset contains 134 shadow images (62 pairs of shadow and shadow-free images). <br>
   We use 34 pairs for testing and 100 shadow images for training. <br>
   For shadow-free training images, 28 from LRSS and 72 randomly selected from the USR dataset.<br>
   |[[Dropbox]](https://www.dropbox.com/scl/fo/3dt75e23riozwa6uczeqd/ABNkIZKaP8jFarfNrUUjpVg?rlkey=eyfjn7dhd9pbz6rh247ylbt0c&st=01lh80r8&dl=0)|[[BaiduPan(code:t9c7)]](https://pan.baidu.com/s/1c_VsDVC92WnvI92v8cldsg?pwd=t9c7)|
   | :-----------: | :-----------: |


## Pre-trained Models: [[Dropbox]](https://www.dropbox.com/sh/346iirg55e1qnir/AADqxEu8vyj4KfKR0wOfELjKa?dl=0) | [[BaiduPan(code:gr59)]](https://pan.baidu.com/s/1EyYvjeu6AnJuY3wEuJS74A?pwd=gr59) 
| Dataset  | Model Dropbox | Model BaiduPan | Model Put in Path| Results Dropbox | Results BaiduPan |
| :----: | :-----------: | :----------: |:---------------: |  :----------: |:---------------: | 
| SRD |[[BaiduPan(code:zhd2)]](https://pan.baidu.com/s/1CV1wQkSMR9OOw9ROAdY-pg?pwd=zhd2) |`results/SRD/model/`| 
| AISTD/ISTD+ |[[BaiduPan(code:cfn9)]](https://pan.baidu.com/s/1wuAZ9ACx6w_2v2MbzrYY7Q?pwd=cfn9)  |`results/AISTD/model/`| 
| ISTD |[[BaiduPan(code:b8o0)]](https://pan.baidu.com/s/1qtC0PtCqS5drYRi1-Ta2gw?pwd=b8o0) |
| USR | [BaiduPan(code:e0a8)](https://pan.baidu.com/s/16MYozQ3QYT3bAhE-eTehXA?pwd=e0a8)  |`results/USR/model/`| 
| LRSS| [[BaiduPan(code:bbns)]](https://pan.baidu.com/s/1yLxFKLH7QJr_f75ITUCRMQ?pwd=bbns) |

## Single Image Test
1. Download the pre-trained SRD model | [[BaiduPan(code:zhd2)]](https://pan.baidu.com/s/1CV1wQkSMR9OOw9ROAdY-pg?pwd=zhd2), put in `results/SRD/model/`
2. Put the test images in `./samples`, results in: `results/(dataset)/output/` <br>
```
${DCShadowNet}
|-- samples           
|-- results
    |-- SRD 
      |--input_output
      |--output           
```

```
> bash test.sh
> python test.py --dataset 'SRD' --device 'cpu' --result_dir 'results' --samplepath './samples'
```
## Real Time Test
Download the pre-trained SRD model | [[BaiduPan(code:zhd2)]](https://pan.baidu.com/s/1CV1wQkSMR9OOw9ROAdY-pg?pwd=zhd2), put in `results/SRD/model/`
```
> bash test_realtime.sh
> python test_realtime.py --device 'cpu' --dataset 'SRD' --samplepath './samples' --cameraID 0 --optimize

```
## Optimization module

Include CPU infernce optimization, Model pruning and Model quantization operations
Improve CPU inference performance

## Train
```
${DCShadowNet}
|-- dataset
    |-- SRD
      |-- trainA ## Shadow 
      |-- trainB ## Shadow-free 
      |-- testA  ## Shadow 
      |-- testB  ## Shadow-free 
    |-LRSS
      |-- trainA ## Shadow 
      |-- trainB ## Shadow-free 
      |-- testM  ## shadow-mask
      |-- testA  ## Shadow 
      |-- testB  ## Shadow-free 
```
```
> bash train.sh
> python train.py --dataset 'SRD' --datasetpath '/home/luni/shadow_removal_project/dataset/SRD' --iteration 20 --batch_size 1 --lr 0.0001 --device 'cpu'
```


