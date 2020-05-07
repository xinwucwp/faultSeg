# FaultSeg3D: using synthetic datasets to train an end-to-end convolutional neural network for 3D seismic fault segmentation

**This is a [Keras](https://keras.io/) version of FaultSeg implemented by [Xinming Wu](http://www.jsg.utexas.edu/wu/) for 3D fault segmentation**

As described in **FaultSeg: using synthetic datasets to train an end-to-end convolutional neural network for 3D fault segmentation** by [Xinming Wu](http://www.jsg.utexas.edu/wu/)<sup>1</sup>, 
[Luming Liang](https://sites.google.com/site/lumingliangshomepage/)<sup>2</sup>, 
[Yunzhi Shi](https://scholar.google.com/citations?user=t9dQMgIAAAAJ&hl=en)<sup>1</sup> and 
[Sergey Fomel](https://www.jsg.utexas.edu/researcher/sergey_fomel/)<sup>1</sup>.
<sup>1</sup>BEG, UT Austin; <sup>2</sup>Uber.

## Getting Started with Example Model for fault prediction

If you would just like to try out a pretrained example model, then you can download the [pretrained model](https://drive.google.com/drive/folders/1q8sAoLJgbhYHRubzyqMi9KkTeZWXWtNd) and use the 'prediction.ipynb' script to run a demo.

On 05/07/2020, new models are uploaded for testing. Please run predNew.ipynb to test the new models.

### Dataset

**To train our CNN network, we automatically created 200 pairs of synthetic seismic and corresponding fault volumes, which were shown 
to be sufficient to train a good fault segmentation network.** 

**The training and validation datasets can be downloaded [here](https://drive.google.com/drive/folders/1FcykAxpqiy2NpLP1icdatrrSQgLRXLP8)**

### Training

Run train.py to start training a new faultSeg model by using the 200 synthetic datasets

## Publications

If you find this work helpful in your research, please cite:

    @article{wu2019faultSeg,
        author = {Xinming Wu and Luming Liang and Yunzhi Shi and Sergey Fomel},
        title = {Fault{S}eg3{D}: using synthetic datasets to train an end-to-end convolutional neural network for 3{D} seismic fault segmentation},
        journal = {GEOPHYSICS},
        volume = {84},
        number = {3},
        pages = {IM35-IM45},
        year = {2019},
    }

---
## Validation on a synthetic example
Fault detections are computed on a syntehtic seismic image by using 8 methods of C3 (Gersztenkorn and Marfurt, 1999),
C2 (Marfurt et al., 1999), planarity (Hale, 2009), structure-oriented linearity (Wu, 2017), structure-oriented semblance (Hale, 2009), fault likelihood (Hale, 2013; [Wu and Hale, 2016](https://library.seg.org/doi/abs/10.1190/geo2015-0380.1), [code](https://github.com/dhale/ipf)), optimal surface voting ([Wu and Fomel, 2018](https://library.seg.org/doi/abs/10.1190/geo2018-0115.1), [code](https://github.com/xinwucwp/osv)) and our CNN-based segmentation.
![results/comparison.jpeg](results/comparison.jpeg)

**To quantitatively evaluate the fault detection methods, we further calculate the precision-recall (Martin et al., 2004) and receiver operator characteristic (ROC) (Provost et al., 1998) curves shown below. From the precision-recall curves, we can clearly observe that our CNN method (red curve) provides the highest precisions for all the choices of recall.**
![results/PR_and_ROC_curves.jpeg](results/PR_and_ROC_curves.jpeg)

---
## Validation on multiple field examples

Although trained by only synthetic datasets, the CNN model works well in 
predicting faults in field datasets that are acquired at totally different surveys. 


### Example of Netherlands off-shore F3 (provided by the Dutch Government through TNO and dGB Earth Sciences)

compare the CNN fault probability (top right) with fault likelihood (bottom)
![results/f3CnnFaultByWu.png](results/f3CnnFaultByWu.png)

---
### Example of Clyde (provided by Clyde through Paradigm)

compare the CNN fault probability (middle) with fault likelihood (right)
![results/clydeCnnFaultByWu.png](results/clydeCnnFaultByWu.png)

---
### Example of Costa Rica (acquired in the subduction zone, Costa Rica Margin, provided by Nathan Bangs)

compare the CNN fault probability (left column) with fault likelihood (right column)
![results/crfCnnFaultByWu.png](results/crfCnnFaultByWu.png)

---
### Example of Campos (acquired at the Campos Basin, offshore Brazil, provided by Michacle Hudec)
![results/camposCnnFaultByWu.png](results/camposCnnFaultByWu.png)

---
### Example of [Kerry-3D](https://wiki.seg.org/wiki/Kerry-3D) (The fault features have been thinned in this example)
![results/kerryCnnFaultByWu.png](results/kerryCnnFaultByWu.png)

---
### Example of [Opunake-3D](https://wiki.seg.org/wiki/Opunake-3D) (The fault features have been thinned in this example)
![results/opunakeCnnFaultByWu.png](results/opunakeCnnFaultByWu.png)

## License

This extension to the Keras library is released under a creative commons license which allows for personal and research use only. 
For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
