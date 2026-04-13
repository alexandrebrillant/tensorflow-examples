Hello,

While studying the book "Deep Learning with JavaScript" by Shanqing Cai, Stanley Bileschi, Eric D. Nielsen, and François Chollet,

I was not entirely satisfied with the examples and results obtained using the TensorFlow.js library. 

As a result, I decided to rewrite the examples from scratch using different strategies to see if better results could be achieved. I replaced too commonjs modules by a modern usage of the ES6 module. I removed all unnecessary dependencies (like the one for CSV format).

You need to install tensoflow.js (I used the simple one tfjs) on your machine

Using this package :

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

It is possible that the normalization technique ha a significant impact ??.


