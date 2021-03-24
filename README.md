# ibug.face_alignment
This branch is to show that I am getting differences in my tensor outputs between the stable build and the nightly build

## Prerequisites
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [OpenCV](https://opencv.org/): `$pip3 install opencv-python`
* Build pytorch `using conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge` and it should work
* If you build with `conda install pytorch torchvision cudatoolkit=11.1 -c pytorch-nightly -c conda-forge` instead, it will fail and the results will be clear


## How to Test
with two different environments env1 and env2, each with a different pytorch version:
```
conda activate env1
python batch_norm_test.py

conda activate env2
python batch_norm_test.py
```

Then compare the two out files in [batch_norm_test_resources](batch_norm_test_resources/)
You can already see with the current one, that there is a difference at the output of "Convblock: b3_1.conv3" 


## References
\[1\] Bulat, Adrian, and Georgios Tzimiropoulos. "[How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks).](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf)" In _Proceedings of the IEEE International Conference on Computer Vision_, pp. 1021-1030. 2017.
