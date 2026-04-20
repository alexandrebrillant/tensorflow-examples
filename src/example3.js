/**
 * example3.js
 * Multi-class classification for iris flowers
 * It uses iris data from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
 * (c) Alexandre Brillant
 * https://github.com/AlexandreBrillant/
 * https://www.alexandrebrillant.com
 */

import { readFileSync } from "fs";
import tf from '@tensorflow/tfjs';

// List of the flowers from the data
// Could be extracted automatically too from the data
const IRIS_NAMES = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]

// Map index to a name
const NAMES_INDEX = [];

// Compute each real label for the IA network with a probability per flower [1,0,0],[0,1,0],[0,0,1]
const LABELS = {};

IRIS_NAMES.forEach( 
    ( flower, index ) => {
        LABELS[ flower ] = ( new Array( IRIS_NAMES.length ).fill( 0.0 ) );
        LABELS[ flower ][ index ] = 1.0;
        NAMES_INDEX[ index ] = { name:flower };
    } 
);

/**
 * Load/Parse local Iris data
 * @returns a {data,shape} objet for training/testing data */
function loadData() {

    const content = readFileSync( "./data/iris/iris.csv", "UTF-8" );
    // Random array
    const tab = content.split( "\n" ).sort( () => Math.random() - 0.5 );
    const firstCols = tab[0].split( ",");
    const shape = {
        "train-data.csv" : [ tab.length / 2, firstCols.length - 1 ],
        "train-target.csv" : [ tab.length / 2, 3 ],
        "test-data.csv" : [ tab.length / 2, firstCols.length - 1 ],
        "test-target.csv" : [ tab.length / 2, 3 ],            
    };

    // Extract data and shape
    return tab.reduce( ( acc, line, index ) => {
        const cols = line.split( "," );
        let namePrefix = "train";
        if ( index >= tab.length / 2 )
            namePrefix = "test";

        // Convert string to float
        acc.data[ namePrefix + "-data.csv" ] = acc.data[ namePrefix + "-data.csv" ] || [];
        acc.data[ namePrefix + "-data.csv" ].push( ...cols.filter( (value, index ) => index < cols.length - 1 ).map( ( value ) => parseFloat( value ) ) );

        // Convert label to unique value [1,0,0],[0,1,0],[0,0,1]
        acc.data[ namePrefix + "-target.csv" ] = acc.data[ namePrefix + "-target.csv" ] || [];
        acc.data[ namePrefix + "-target.csv" ].push( ...cols.filter( ( value, index ) => index == cols.length - 1 ).map( ( value ) => LABELS[ value ] ) );

        return acc;

    }, { data : {}, shape } );

}

const irisData = loadData();

const trainTensors = {
    data : tf.tensor2d( irisData.data[ "train-data.csv" ], irisData.shape[ "train-data.csv"] ),
    target : tf.tensor2d( irisData.data[ "train-target.csv" ], irisData.shape[ "train-target.csv" ] )
};

const testTensors = {
    data : tf.tensor2d( irisData.data[ "test-data.csv" ], irisData.shape[ "test-data.csv"] ),
    target : tf.tensor2d( irisData.data[ "test-target.csv" ], irisData.shape[ "test-target.csv"] ),
}


const features = irisData.shape[ "train-data.csv"][1];
const rows = irisData.shape[ "train-data.csv"][0];

// Display training/testing data

const dataTypes = [ "train", "test" ];
dataTypes.forEach( 
    (dataType) => {

        console.log( `** ${dataType} data **` )
        console.log( rows + " samples with " + features + " features" );
        const label_0 = irisData.data[ dataType + "-target.csv" ].filter( item => item[0]==1 ).length;
        const label_1 = irisData.data[ dataType + "-target.csv" ].filter( item => item[1]==1 ).length;
        const label_2 = irisData.data[ dataType + "-target.csv" ].filter( item => item[2]==1 ).length;
        console.log( "- Labels to [1,0,0] :" + label_0 + " items" );
        console.log( "- Labels to [0,1,0] :" + label_1 + " items" );
        console.log( "- Labels to [0,0,1] :" + label_2 + " items" );

    } );

console.log( "-------------------------------------" )
console.log( "Running multi-class classification..." )

const tensorFlowRuntime = async ( strategy, logMode = false ) => {
    const model = tf.sequential();
    model.add( tf.layers.dense( { inputShape : [ features ], units : strategy.maxUnits, activation: strategy.activation }) );
    model.add( tf.layers.dense( { units : Object.keys( LABELS ).length, activation : "softmax" } ) );

    model.compile( 
        { 
            optimizer: strategy.optimizer, 
            loss: strategy.loss,
            metrics : [ "accuracy" ]
        }
    );

    await model.fit( 
        trainTensors.data,
        trainTensors.target, { 
            epochs: strategy.maxEpochs,
            callbacks: {
                onEpochEnd: async( epoch, logs ) => {
                    const accuracy = logs.accuracy;
                    if ( logMode ) {
                        console.log( epoch + " : " + accuracy );
                    }
                }
            }
    } );

    const predictions = await model.predict(testTensors.data);
    const result = predictions.arraySync();
    const goodTarget = irisData.data[ "test-target.csv" ];

    let total_good_prediction = 0;
    let total_prediction = 0;
    result.forEach( 
        ( prediction, index ) => {
            const goodLabel = goodTarget[ index ];
            // Compute the predicion index for the label
            const maxValue = Math.max( ...prediction );
            const maxIndex = prediction.indexOf( maxValue );
            if ( goodLabel[ maxIndex ] == 1 ) {    
                total_good_prediction++;
                NAMES_INDEX[ maxIndex ].goodPrediction = ( NAMES_INDEX[ maxIndex ].goodPrediction || 0 ) + 1;
            }
            NAMES_INDEX[ maxIndex ].totalPrediction = ( NAMES_INDEX[ maxIndex ].totalPrediction || 0 ) + 1;
            total_prediction++;
        } );
    // Display the total result
    const accuracy = Math.floor( ( total_good_prediction / total_prediction ) * 100 );
    console.log( "--------------------------------------------" );
    console.log( JSON.stringify( strategy ) );
    console.log( "Total Accuracy =" + accuracy + "%" );
    // Display the result by flower
    for ( const value of Object.values( NAMES_INDEX ) ) {
        const flowerAccuracy = Math.floor( ( value.goodPrediction / value.totalPrediction ) * 100 );
        console.log( "- Flower " + value.name + " Accuracy = " + flowerAccuracy + " %" );
    }
};

const strategies = [
    { maxUnits : 100, maxEpochs : 500, loss : "categoricalCrossentropy", activation: "sigmoid", optimizer : "adam" },
    { maxUnits : 10, maxEpochs : 250, loss : "categoricalCrossentropy", activation: "sigmoid", optimizer : "adam" },
    { maxUnits : 100, maxEpochs : 250, loss : "categoricalCrossentropy", activation: "relu", optimizer : "adam" },
    { maxUnits : 10, maxEpochs : 500, loss : "categoricalCrossentropy", activation: "relu", optimizer : "adam" },
    { maxUnits : 10, maxEpochs : 500, loss : "categoricalCrossentropy", activation: "tanh", optimizer : "adam" }
];

strategies.forEach( 
    ( strategy ) => {    
        tensorFlowRuntime( strategy )
} );

