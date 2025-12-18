# The realization of paper 'HD-CNN: Hierarchical Deep Convolutional Neural Networks for Large Scale Visual Recognition'

## HD-CNN
* remember to chage the data address: 'root_path' in dataset.py
* Change: 'Determine which dataset to be used ================ cifar-10/cifar-100'
* Change: 'The number of cluster centers ======================= 2/9'
* Change: 'The number of pre-train steps ================== when final test, make it bigger'
* Change: 'The number of test batch steps in a epoch ================= when final test, make it bigger'
* Change: 'The number of fine classes ======================= 10/100'
* Change (optional): two parameters: 'u_t= 1/(args.num_superclass*5)' in main.py and 'lam=20' in loss.py

### Possible problems
* I am not sure if cuda works
* I am not sure if fine-tune part have problem as the accuracy looks a little weird _I am trying to find why_

This repository contains source code from an early-stage undergraduate research project. It is not actively maintained, and the included implementations reflect the practices at the time of the study rather than current standards.
