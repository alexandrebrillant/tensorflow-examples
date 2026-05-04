/** 
* example4.js
* MNIST image classification
* All the images and labels are stored in a specific ubyte format inside the data/mnist folder
* 
* For ARM, you must use @tensorflow/tfjs
* For x86, you must use @tensorflow/tfjs-node
*
* (c) Alexandre Brillant
* https://github.com/AlexandreBrillant/
* https://www.alexandrebrillant.com
*/

import { readFileSync } from "fs";
import tf from '@tensorflow/tfjs';

// try these better solution (need to install it before)
// import tf from '@tensorflow/tfjs-node';
// import tf from '@tensorflow/tfjs-node-gpu';

// Images data

// Limit the number of images for testing
// Use the following code
// const limitSize = Number.MAX_SAFE_INTEGER;
const limitSize = 1000;

function imagesToTensor4d( path ) {
    const res = [];
    const buffer = readFileSync( path );
    const magicNumber = buffer.readUInt32BE(0);
    const count = buffer.readUInt32BE(4);
    const height = buffer.readUInt32BE(8);
    const width = buffer.readUInt32BE(12);
    const imageData = buffer.subarray(16);

    const maxImages = Math.min( limitSize, count );

    for (let i = 0; i < maxImages; i++) {
        const offset = i * height * width;
        const image = imageData.subarray(offset, offset + height * width );

        const image4d = [];
        for ( let y = 0; y < height; y++ ) {
            const row = [];
            for ( let x = 0; x < width; x++ ) {
                row.push( [ image[ y * width + x ] / 255 ] );
            }
            image4d.push( row );
        }

        res.push( image4d );
    }

    const tensor = tf.tensor4d( 
        res, [ maxImages, width, height, 1 ] );

    return { 
        height,
        width,
        tensor
    };
}

function readLabels( path ) {
    const res = [];
    const buffer = readFileSync( path );
    const magicNumber = buffer.readUInt32BE(0);
    const count = buffer.readUInt32BE(4);
    const labelsData = buffer.subarray(8);
    
    const maxLabels = Math.min( count, limitSize );

    for (let i = 0; i < maxLabels; i++) {
        const label = labelsData[i];
        const array = new Array(10);
        array.fill( 0 );
        array[ label ] = 1;
        res.push( array );
    }

    return tf.tensor2d( res, [ maxLabels, 10 ] );
}

function readMnist() {
    const images = {
        "train" : { source : "train-images-idx3-ubyte" },
        "test" : { source : "t10k-images-idx3-ubyte" }
    };
    const labels = {
        "train" : { source : "train-labels-idx1-ubyte" },
        "test" : { source : "t10k-labels-idx1-ubyte" }
    };

    for ( const cat of [ 'train', 'test' ] ) {
        const imagesPath = "./data/mnist/" + images[ cat ].source;
        const labelsPath = "./data/mnist/" + labels[ cat ].source;
        images[ cat ] = imagesToTensor4d( imagesPath );
        labels[ cat ] = readLabels( labelsPath );
    }

    return {
        "train" : {
            labels : labels[ "train" ],
            images : images[ "train" ].tensor,
            width : images[ "train" ].width,
            height:  images[ "train" ].height,

        },
        "test" : {
            labels : labels[ "test" ],
            images : images[ "test" ].tensor,
            width : images[ "test" ].width,
            height : images[ "test" ].height
        }
    };
}

console.log( "Reading data..." );

const mnist = readMnist();

///////////////////////// Model

const tensorFlowRuntime = async ( strategy, logMode = false ) => {

    const kernelSize = strategy.kernelSize || 3;
    const filters = strategy.filters || 16;

    const model = tf.sequential();

    model.add( tf.layers.conv2d(
        {
            inputShape : [ mnist.train.height, mnist.train.width, 1 ],
            kernelSize,
            filters,
            activation : "relu"
        } ) );

    model.add( tf.layers.maxPooling2d( { poolSize:2, strides : 2 } ) );

    /*
    model.add( tf.layers.conv2d(
        {
            kernelSize,
            filters,
            activation : "relu"
        } ) );

    model.add( tf.layers.maxPooling2d( { poolSize:2, strides : 2 } ) );
    */

    model.add( tf.layers.flatten() );

    const units = strategy.units || 64;

    model.add( tf.layers.dense( { units, activation : "relu" } ) );
    model.add( tf.layers.dense( { units : 10, activation : "softmax" } ) );

    const optimizer = "rmsprop";

    model.compile( 
        { 
            optimizer,
            loss : "categoricalCrossentropy",
            metrics : [ "accuracy" ]
        } );

    console.log( `Training... ${JSON.stringify( strategy ) }` );

    const batchSize = strategy.bachSize || 200;

    await model.fit( 
        mnist.train.images, 
        mnist.train.labels,
        {
            // epochs : 100,
            validationSplit : 0.15,
            batchSize,
            callbacks: {
                onEpochEnd: async( epoch, logs ) => {
                }
            }
        }
    );

    console.log( "Evaluating..." );

    const res = model.evaluate( mnist.test.images, mnist.test.labels );

    const loss = res[0].dataSync()[0];
    const acc = res[1].dataSync()[0];

    console.log( "================================" );
    console.log( JSON.stringify( strategy ) );
    console.log( `Loss : ${loss} / Accuracy : ${(acc * 100).toFixed(2)}%` );
}

const strategies = [
    { 
        kernelSize:2,
        filters:8,
        units:32
    },
    { 
        kernelSize:3,
        filters:16,
        units:64
    },
    { 
        kernelSize:3,
        filters:32,
        units:64
    },
    { 
        kernelSize:3,
        filters:32,
        units:128
    },
    { 
        kernelSize:3,
        filters:64,
        units:64
    },    
    { 
        kernelSize:2,
        filters:32,
        units:64
    },       
    { 
        kernelSize:4,
        filters:32,
        units:64
    }
];

strategies.forEach( strategy => tensorFlowRuntime( strategy ) );
