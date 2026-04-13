/**
 * example2.js
 * Binary classification for phishing site web data detection
 * It uses local phishing detection data files with 30 features * ~5000 samples => ~150000 data
 * (c) Alexandre Brillant
 * https://github.com/AlexandreBrillant/
 * https://www.alexandrebrillant.com
 */

import { readFileSync } from "fs";
import tf from '@tensorflow/tfjs';

/**
 * Load/Parse local Phishing data
 * @returns 
 */
function loadData() {

    // data columns
    // HAVING_IP_ADDRESS,URL_LENGTH,SHORTINING_SERVICE,HAVING_AT_SYMBOL,DOUBLE_SLASH_REDIRECTING,PREFIX_SUFFIX,HAVING_SUB_DOMAIN,SSLFINAL_STATE,DOMAIN_REGISTERATION_LENGTH,FAVICON,PORT,HTTPS_TOKEN,REQUEST_URL,URL_OF_ANCHOR,LINKS_IN_TAGS,SFH,SUBMITTING_TO_EMAIL,ABNORMAL_URL,REDIRECT,ON_MOUSEOVER,RIGHTCLICK,POPUPWIDNOW,IFRAME,AGE_OF_DOMAIN,DNSRECORD,WEB_TRAFFIC,PAGE_RANK,GOOGLE_INDEX,LINKS_POINTING_TO_PAGE,STATISTICAL_REPORT

    // target columns
    // RESULT
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
            const content = readFileSync( "./data/phishing/" + file, "UTF-8" );
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

            data[ file ] = data[ file ].map( item => parseInt( item ) );
            shape[ file ] = [ row, col ];
        } 
    );

    return { data, shape };
}

const phishingData = loadData();

const trainTensors = {
    data : tf.tensor2d( phishingData.data[ "train-data.csv" ], phishingData.shape[ "train-data.csv"] ),
    target : tf.tensor2d( phishingData.data[ "train-target.csv" ], phishingData.shape[ "train-target.csv" ] )
};

const testTensors = {
    data : tf.tensor2d( phishingData.data[ "test-data.csv" ], phishingData.shape[ "test-data.csv"] ),
    target : tf.tensor2d( phishingData.data[ "test-target.csv" ], phishingData.shape[ "test-target.csv"] ),
}

const features = phishingData.shape[ "train-data.csv"][1];
const rows = phishingData.shape[ "train-data.csv"][0];

console.log( "- Training data :" )
console.log( rows + " samples with " + features + " features" );
const label_1 = phishingData.data[ "train-target.csv" ].filter( item => item==1 ).length;
const label_0 = phishingData.data[ "train-target.csv" ].filter( item => item==0 ).length;
console.log( "Labels to 0 :" + label_0 + " items" );
console.log( "Labels to 1 :" + label_1 + " items" );

const features2 = phishingData.shape[ "test-data.csv"][1];
const rows2 = phishingData.shape[ "test-data.csv"][0];

console.log( "- Testing data :" )
console.log( rows2 + " samples with " + features2 + " features" );
const label_12 = phishingData.data[ "test-target.csv" ].filter( item => item==1 ).length;
const label_02 = phishingData.data[ "test-target.csv" ].filter( item => item==0 ).length;
console.log( "Labels to 0 :" + label_02 + " items" );
console.log( "Labels to 1 :" + label_12 + " items" );

console.log( "-------------------------------------" )
console.log( "Running binary classification..." )

const tensorFlowRuntime = async ( strategy, logMode = false ) => {

    const model = tf.sequential();

    model.add( tf.layers.dense( { inputShape : [ features ], units : strategy.maxUnits, activation: strategy.activation }) );
    model.add( tf.layers.dense( { units : 1 }) );
    model.compile( 
        { 
            optimizer: strategy.optimizer, 
            loss: strategy.loss,
            metrics : [ "accuracy", "precision" ]
        }
    );

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

    // Convert the test data to a prediction with a label 0 or 1

    const predictions = model.predict(testTensors.data).greater(strategy.threshold).cast( "int32" );
    const predictionsArray = predictions.dataSync();
    const targetArray = testTensors.target.dataSync()

    let totalWrongPrediction = 0;
    let totalMissPrediction = 0;

    for (let i = 0; i < predictionsArray.length; i++) {
        if ( predictionsArray[i] == 1 ) {
            if ( predictionsArray[i]!= targetArray[i] )
                totalWrongPrediction++;
        }

        if ( targetArray[i] ==1 ) {
            if ( predictionsArray[i] != 1 )
                totalMissPrediction++;
        }
    }

    const goodPrediction = ( 1 - ( totalWrongPrediction / predictionsArray.length ) ) * 100;
    const missPrediction = ( totalMissPrediction  / predictionsArray.length ) * 100;

    console.log( "Label 1 : Good prediction (" + goodPrediction + "%) - Miss prediction (" + missPrediction + "%)" );

};

const strategies = [
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.5 },
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.6 },
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.7 },
    { maxUnits : 10, maxEpochs : 100, loss : "binaryCrossentropy", activation: "sigmoid", optimizer : "adam", threshold:0.8 }
];

strategies.forEach( 
    ( strategy ) => {
        tensorFlowRuntime( strategy )
} );
