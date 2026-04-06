Hello,

While studying the book "Deep Learning with JavaScript" by Shanqing Cai, Stanley Bileschi, Eric D. Nielsen, and François Chollet,

I was not entirely satisfied with the examples and results obtained using the TensorFlow.js library. 

As a result, I decided to rewrite the examples from scratch using different strategies to see if better results could be achieved.

You need to install tensoflow.js on your machine

```bash
npm i
```

or directly

```bash
npm i @tensorflow/tfjs
```

(c) Alexandre Brillant

# example1.js

## Data

The data directory contains a local version of the Boston Housing dataset (all with the CSV format), which includes 12 features and 333 samples.

I choose to normalize the data using the formula: (value − min_value) / (max_value − min_value). I don't use the book normalize function with the mean.

## Goal

I train a non-linear model with 2 layers using various strategies. 

The goal is to estimate the price for house using 12 features (crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,lstat).

## Run the example

```bash
npm run ex1
```

or 

```bash
node src/example1.js
```

## Book result

 Final loss inside the Book at 23 with 50 units for the first layer ?

## Stategies

All my strategies are configurable via this table:

```javascript
const strategies = [
    { maxUnits : 1, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "sgd" },  // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "sgd" },  // Good
    { maxUnits : 5, maxEpochs : 100, loss : "meanSquaredError", activation: "relu", optimizer : "sgd" }, // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "sigmoid", optimizer : "sgd" }, // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "adam" },  // Bad
];
```

It uses 5 units only.

## Result

The result for this data is:

Loss Result with {"maxUnits":1,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"relu","optimizer":"sgd"} .... :
**11.703289031982422**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"relu","optimizer":"sgd"} .... :
**3.623922109603882**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanSquaredError","activation":"relu","optimizer":"sgd"} .... :
**19.283775329589844**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"sigmoid","optimizer":"sgd"} .... :
**6.683498382568359**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"relu","optimizer":"adam"} .... :
**9.521827697753906**

The best loss achieved is 3.6, but it is possible to go below 2 by increasing the number of epochs.

## Comment

I do not understand why, in the aforementioned book, the final loss is so high (23 with 50 units !), nor why the authors rely on external libraries to parse a simple .csv file ??
