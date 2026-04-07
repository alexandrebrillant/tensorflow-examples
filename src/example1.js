/**
 * Multiple strategies using TensorFlow.js
 * It uses local Boston House data files with 12 features * 333 sample => 3996 data
 * (c) Alexandre Brillant
 */

import { readFileSync } from "fs";
import tf from '@tensorflow/tfjs';

/**
 * Load/Parse local Boston data
 * @returns 
 */
function loadData() {

    // data columns
    // crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,lstat

    // target columns
    // medv
    const files = [
        "test-data.csv",
        "test-target.csv",
        "train-data.csv",
        "train-target.csv"
    ]

    const data = {
    };
    const shape = {
    }

    files.forEach( 
        ( file ) => {
            const content = readFileSync( "./data/boston/" + file, "UTF-8" );
            const tab = content.split( "\n" );
            data[ file ] = [];
            let col;
            let row;
            for ( let i = 1; i < tab.length; i++ ) {
                const lineTab = tab[ i ].split( "," );
                !col && ( col = lineTab.length );
                if ( lineTab.length != col ) {
                    throw "Invalid row " + ( i + 1 ) + " should be " + col + " columns for file [" + file + "]";
                }
                data[ file ].push( ...lineTab );
            }
            row = tab.length - 1;

            data[ file ] = data[ file ].map( item => parseFloat( item ) );
            shape[ file ] = [ row, col ];
        } 
    );

    return { data, shape };
}

// normalize_value = (value − min_value) / (max_value − min_value)
function normalizer( tensor2d ) {
    const shape = tensor2d.shape;
    const colCount = shape[1];
    const normalisees = [];
    for ( let i = 0; i < colCount; i++ ) {
        const col = tensor2d.slice( [ 0, i ], [-1, 1 ] );
        const minValue = col.min();
        const maxValue = col.max();
        const colNorm = ( col.sub( minValue ) ).div( maxValue.sub( minValue ) );
        normalisees.push( colNorm );
    }
    return tf.concat( normalisees, 1);
}

//////////////////////////////////////////////////////////////////////////////////////

const tensorFlowRuntime = ( async ( strategy, logMode = false ) => {
    const bostonData = loadData();
    const model = tf.sequential();

    model.add( tf.layers.dense( { inputShape : [12], units : strategy.maxUnits, activation: strategy.activation }) );
    model.add( tf.layers.dense( { units : 1 }) );
    model.compile( { optimizer: strategy.optimizer, loss: strategy.loss } );

    const trainTensors = {
        data : tf.tensor2d( bostonData.data[ "train-data.csv" ], bostonData.shape[ "train-data.csv"] ),
        target : tf.tensor2d( bostonData.data[ "train-target.csv" ], bostonData.shape[ "train-target.csv" ] )
    };

    trainTensors.data = normalizer( trainTensors.data );

    await model.fit( 
        trainTensors.data,
        trainTensors.target, { 
            epochs: strategy.maxEpochs,
            callbacks: {
                onEpochEnd: async( epoch, logs ) => {
                    const trainLoss = logs.loss;
                    if ( logMode )
                        console.log( epoch + " : " + trainLoss );
                }
            }
    } );

    const testTensors = {
        data : tf.tensor2d( bostonData.data[ "test-data.csv" ], bostonData.shape[ "test-data.csv"] ),
        target : tf.tensor2d( bostonData.data[ "test-target.csv" ], bostonData.shape[ "test-target.csv"] ),
    }

    testTensors.data = normalizer( testTensors.data );
    const loss = model.evaluate( testTensors.data, testTensors.target ).dataSync()[ 0 ];

    console.log( "Loss Result with " + JSON.stringify( strategy ) + " .... :" + loss.toString() );

} );

const strategies = [
    { maxUnits : 1, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "sgd" },  // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "sgd" },  // Good
    { maxUnits : 5, maxEpochs : 100, loss : "meanSquaredError", activation: "relu", optimizer : "sgd" }, // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "sigmoid", optimizer : "sgd" }, // Bad
    { maxUnits : 5, maxEpochs : 100, loss : "meanAbsoluteError", activation: "relu", optimizer : "adam" },  // Bad
];

strategies.forEach( 
    ( strategy ) => {
        tensorFlowRuntime( strategy )
} );
