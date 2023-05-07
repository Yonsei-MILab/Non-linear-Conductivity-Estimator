## Non-linear Conductivity Estimator

Source code for the non-linear conductivity estimator with ANN.

## Submitted Paper

The paper entitled : "Data-driven Electrical Conductivity Neuroimaging using 3T MRI" was submitted to Human Brain Mapping (Kyu-Jin Jung et al., 2023).


## Usage

* train.py : Code for training the network.

* test.py : Code for testing the network.

## Example File

As an example training and test datasets, numerical simulations were performed in MATLAB for cylindrical phantoms: 2D-complex B1+ & B1- maps were obtained by the Bessel-boundary-matching method (Bob van den Bergen et al., 2009), which enables electromagnetic field simulations for concentric models.
Both single & double-layered cylindrical models were composed with various radii and various conductivity values, resulting in a total of 58 example cylinder models for the training dataset.
The test dataset is a double-layered cylinder model, which has never been included for the radii in the training dataset (for both inner & outer cylinders' radii).

* Example_Nonlinear_Conductivity_Estimator_Cylinder : Code example for training & testing the network with cylinder models.

* Cylinder_Example_Epoch__150.pth : Network weight example, which was trained with cylinder models.

* Cylinder_Training_Dataset.mat : Training dataset with cylinder models.

* Cylinder_Test_Dataset.mat : Test dataset with a cylinder model.

# Reference
Jung, K.J., Mandija, S., Cui, C., Kim, J.H., Al-masni, M.A., Meerbothe, T.G., Park, M., van den Berg, C.A.T., and Kim, D.H. (2023), Data-driven Electrical Conductivity Neuroimaging using 3T MRI. Human Brain Mapping, Under Review.

van den Bergen, B., Stolk, C.C., van den Berg, J.B., Lagendijk, J.J.W., and Van den Berg, C.A.T. (2009), Ultra fast electromagnetic field computations for RF multi-transmit techniques in high field MRI. Physics in Medicine & Biology, 54(5), 1253-1264.
