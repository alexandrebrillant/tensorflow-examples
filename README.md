Hello,

While studying the book "Deep Learning with JavaScript" by Shanqing Cai, Stanley Bileschi, Eric D. Nielsen, and François Chollet,

![TensorFlow.js book](images/book.jpg)

I was not entirely satisfied with the examples and results obtained using the TensorFlow.js library. 

As a result, I decided to rewrite the examples from scratch using different strategies to see if better results could be achieved. I replaced too commonjs modules by a modern usage of the ES6 modules. I removed all unnecessary dependencies (like the one for CSV format).

You need to install tensoflow.js (I used the simple one tfjs) on your machine

Using my package :

```bash
npm i
```

or directly :

```bash
npm i @tensorflow/tfjs
```

(c) [Alexandre Brillant](https://www.alexandrebrillant.com)

# Prediction : example1.js

## Data

The data directory contains a local version of the Boston Housing dataset (all with the CSV format), which includes 12 features and 333 samples. I wrote myself the CSV parsing inside
the **loadData** method.

I choose to normalize the data using the formula: (value − min_value) / (max_value − min_value). I don't use the book normalize function with the mean. This is a general normalizer for any tensor2d object. 

I added a second parameter colValues containing the max/min value for each column of the train data set to be sure to have the same normalization space.

```javascript
function normalizer( tensor2d, colValues ) {
    const shape = tensor2d.shape;
    const colCount = shape[1];
    const normalisees = [];
    const lastColValues = [];

    for ( let i = 0; i < colCount; i++ ) {
        const col = tensor2d.slice( [ 0, i ], [-1, 1 ] );
        const minValue = colValues ? colValues[ i ].minValue : col.min();
        const maxValue = colValues ? colValues[ i ].maxValue : col.max();
        const delta = maxValue.sub( minValue );
        const colNorm = ( col.sub( minValue ) ).div( delta );
        normalisees.push( colNorm );
        lastColValues.push( {
            maxValue,
            minValue
        } );
    }

    return {
        tensor:tf.concat( normalisees, 1),
        colValues:lastColValues
    }
}

```

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

All my strategies (the hyperparameters) are configurable via this table:

```javascript
const strategies = [
    { maxUnits : 1, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "sgd" },  // Good
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "sgd" },  // Good
    { maxUnits : 5, maxEpochs : 100, loss : "meanSquaredError", activation: "relu", optimizer : "sgd" }, // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "sigmoid", optimizer : "sgd" }, // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "adam" },  // Bad
];
```

It uses 5 units only and the basic "sgd" optimizer for the best result.

## Result

The result for this data is:

Loss Result with {"maxUnits":1,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"relu","optimizer":"sgd"} .... :
**3.983289031982422**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"relu","optimizer":"sgd"} .... :
**3.623922109603882**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanSquaredError","activation":"relu","optimizer":"sgd"} .... :
**19.283775329589844**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"sigmoid","optimizer":"sgd"} .... :
**6.683498382568359**

Loss Result with {"maxUnits":5,"maxEpochs":100,"loss":"meanAbsoluteError","activation":"relu","optimizer":"adam"} .... :
**9.521827697753906**

The best loss achieved is 3.6, but it is possible to go below 2 by increasing the number of epochs.

## Conclusion

Definitely, the choices proposed in the book were not optimal. Using a much more computationally expensive strategy, they achieved a loss of 23, which is double the worst performance of my example.

It is possible that the normalization technique has a significant impact ?.


# Binary classification : example2.js

## Data

The data directory contains a local version of the Phishing web site dataset (all with the CSV format), which includes 30 features and about 5000 samples. I wrote myself the CSV parsing inside the **loadData** method.

We needn't to normalize data as this is a set of 0 or 1.

## Goal

I train a non-linear model with 2 layers using various strategies. 

The goal is to classify a phishing or not web site using 30 features. We considere a label to 1 for "Yes this is a phishing
web site" and the label to 0 for "No, this is not a phishing web site".

## Run the example

```bash
npm run ex2
```

or 

```bash
node src/example2.js
```

## Stategies

```javascript
const strategies = [
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.5 },
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.6 },
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.7 },
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.8 }
];
```

We use only the binaryCrossentropy which adds good or bad score depending on the prediction rate.

When a result probability > theshold, then it means this is a label 1 and else this is a label 0. So each time we
increase the threshold, we improve the precision of the prediction because we ask a very high probability.

## Result

The result are only for the Label 1 (phishing detection). Good prediction is a rate of quality when the
model predicts a label 1. The Miss prediction is for the case the model predicts a label 0 for a label 1.

- Label 1 : Good prediction (97.64791025872988%) - Missed prediction (4.776551474579338%)
- Label 1 : Good prediction (97.90121223086665%) - Missed prediction (5.210783426813824%)
- Label 1 : Good prediction (98.91442011941378%) - Missed prediction (8.576081056631084%)
- Label 1 : Good prediction (100%) - Missed prediction (43.495567215487604%)
- Label 1 : Good prediction (99.6924190338339%) - Missed prediction (11.072914781979373%)

=> High Precision (100% at threshold 0.7):

The model only predicts "phishing" when it is almost certain.
Pro: No false alarms (all predicted "phishing" sites are truly phishing).
Con: Many actual phishing sites are missed (high missed prediction rate, e.g., 43%).

Low Missed Predictions (Lower Threshold):

The model predicts "phishing" more often, catching more actual phishing sites.
Pro: Fewer missed phishing sites. Con: More false alarms (lower precision).

There's no universal solution ! The choice depends on your priority

# Multi-class classification : example3.js

## Data

The data directory contains a dataset for the iris flowers.

There're 4 features for detecting the following flowers :

Iris-setosa
Iris-versicolor
Iris-virginica

Each flower is used as an array with 3 columns :

Iris-setosa : [1,0,0]
Iris-versicolor : [0,1,0]
Iris-virginica : [0,0,1]

Each column is a probability (so 1 is for 100%).

## Goal

Be able to detect a flower from 4 features.

## Run the example

```bash
npm run ex3
```

or 

```bash
node src/example3.js
```

## Stategies

```javascript
const strategies = [
    { maxUnits : 100, maxEpochs : 500, loss : "categoricalCrossentropy", activation: "sigmoid", optimizer : "adam" },
    { maxUnits : 10, maxEpochs : 250, loss : "categoricalCrossentropy", activation: "sigmoid", optimizer : "adam" },
    { maxUnits : 100, maxEpochs : 250, loss : "categoricalCrossentropy", activation: "relu", optimizer : "adam" },
    { maxUnits : 10, maxEpochs : 500, loss : "categoricalCrossentropy", activation: "relu", optimizer : "adam" }
];
```

The "categoricalCrossentropy" is required for a multi-class problem. Here we have 3 labels for 3 flowers.

# Result

We displayed both the total accuracy and the accuracy by flower for each strategy. You may run several times for comparing the results.

```javascript
{"maxUnits":100,"maxEpochs":500,"loss":"categoricalCrossentropy","activation":"sigmoid","optimizer":"adam"}
Total Accuracy =98%
- Flower Iris-setosa Accuracy = 100 %
- Flower Iris-versicolor Accuracy = 100 %
- Flower Iris-virginica Accuracy = 96 %
--------------------------------------------
{"maxUnits":10,"maxEpochs":250,"loss":"categoricalCrossentropy","activation":"sigmoid","optimizer":"adam"}
Total Accuracy =68%
- Flower Iris-setosa Accuracy = 100 %
- Flower Iris-versicolor Accuracy = 100 %
- Flower Iris-virginica Accuracy = 65 %
--------------------------------------------
{"maxUnits":100,"maxEpochs":250,"loss":"categoricalCrossentropy","activation":"relu","optimizer":"adam"}
Total Accuracy =98%
- Flower Iris-setosa Accuracy = 100 %
- Flower Iris-versicolor Accuracy = 100 %
- Flower Iris-virginica Accuracy = 73 %
--------------------------------------------
{"maxUnits":10,"maxEpochs":500,"loss":"categoricalCrossentropy","activation":"relu","optimizer":"adam"}
Total Accuracy =97%
- Flower Iris-setosa Accuracy = 100 %
- Flower Iris-versicolor Accuracy = 100 %
- Flower Iris-virginica Accuracy = 77 %
```

The relu activation for the first layer is good enough. The sigmoid usage from the book is not necessary.
10 neurons is enough too for the first layer.


