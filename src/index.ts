import { Model, pca_20_svm } from "./models/pca_20_svm"
import numeric from "./utils/numeric"
import { webglFilter } from "./svmfilter/webglfilter"

// const WIDTH = 0
// const HEIGHT = 1

// const convergenceThreshold = 0.5

// const sobelInit = false
// const lbpInit = false

// const currentPositions = []
// const previousParameters = []
// const previousPositions = []

// const patches = []
// const responses = []

// const responseMode = "single"
// const responseList = ["raw"]
// const responseIndex = 0

/*
It's possible to experiment with the sequence of variances used for the finding the maximum in the KDE.
This sequence is pretty arbitrary, but was found to be okay using some manual testing.
*/
// const varianceSeq = [10, 5, 1]
// var varianceSeq = [3,1.5,0.75];
// var varianceSeq = [6,3,0.75];
// const PDMVariance = 0.7

// const relaxation = 0.1

// const first = true
// const detectingFace = false

// const convergenceLimit = 0.01

// const updatePosition = new Float64Array(2)
// const vecpos = new Float64Array(2)

// const facecheck_count = 0

// const scoringHistory = []
// const meanscore = 0

export const Tracker = ({
    // constantVelocity = true,
    searchWindow = 11,
    // scoreThreshold = 0.5,
    // stopOnConvergence = false,
    // sharpenResponse = false,
    weightPoints
}: // faceDetection = {},
// maxIterationsPerAnimFrame = 3
TrackerParams) => {
    const init = (model: Model = pca_20_svm) => {
        // default model is pca 20 svm model

        // load from model
        const { patchType, numPatches, canvasSize: modelSize, weights, bias } = model.patchModel
        const patchSize = model.patchModel.patchSize[0]
        const { numEvalues: numParameters, eigenValues, nonRegularizedVectors } = model.shapeModel
        // if (patchType === "MOSSE") searchWindow = patchSize

        // // set up canvas to work on
        // sketchCanvas = document.createElement("canvas")
        // sketchCC = sketchCanvas.getContext("2d")

        // sketchW = sketchCanvas.width = modelWidth + (searchWindow - 1) + patchSize - 1
        // sketchH = sketchCanvas.height = modelHeight + (searchWindow - 1) + patchSize - 1

        // load eigenvectors
        const eigenVectors = numeric.rep([numPatches * 2, numParameters], 0.0)
        for (let i = 0; i < numPatches * 2; i++) {
            for (let j = 0; j < numParameters; j++) {
                eigenVectors[i][j] = model.shapeModel.eigenVectors[i][j]
            }
        }

        // load mean shape
        const meanShape = []
        for (let i = 0; i < numPatches; i++) {
            meanShape[i] = [model.shapeModel.meanShape[i][0], model.shapeModel.meanShape[i][1]]
        }

        // get max and mins, width and height of meanshape
        const msMeta = {
            x: {
                min: 1000000,
                max: 0
            },
            y: {
                min: 1000000,
                max: 0
            },
            width: 0,
            height: 0
        }
        for (let i = 0; i < numPatches; i++) {
            if (meanShape[i][0] < msMeta.x.min) msMeta.x.min = meanShape[i][0]
            if (meanShape[i][1] < msMeta.y.min) msMeta.y.min = meanShape[i][1]
            if (meanShape[i][0] > msMeta.x.max) msMeta.x.max = meanShape[i][0]
            if (meanShape[i][1] > msMeta.y.max) msMeta.y.max = meanShape[i][1]
        }
        msMeta.width = msMeta.x.max - msMeta.x.min
        msMeta.height = msMeta.y.max - msMeta.y.min

        // get scoringweights if they exist
        // const scoringWeights = new Float64Array(model.scoring.coef)
        // const scoringBias = model.scoring.bias
        // const scoringCanvas = { width: 0, height: 0 }
        // scoringCanvas.width = model.scoring.size[0]
        // scoringCanvas.height = model.scoring.size[1]

        // precalculate gaussianPriorDiagonal
        const gaussianPD = numeric.rep([numParameters + 4, numParameters + 4], 0)
        // set values and append manual inverse
        for (let i = 0; i < numParameters; i++) {
            gaussianPD[i + 4][i + 4] = nonRegularizedVectors.indexOf(i) >= 0 ? 1 / 10000000 : 1 / eigenValues[i]
        }

        const currentParameters = Array(numParameters + 4).fill(0)
        // for (let i = 0; i < numParameters + 4; i++) {
        // currentParameters[i] = 0
        // }
        const pw = patchType === "SVM" ? patchSize + searchWindow - 1 : searchWindow
        const pl = patchType === "SVM" ? patchSize + searchWindow - 1 : searchWindow

        if (patchType === "SVM") {
            const filter = webglFilter()
            try {
                filter.init(
                    weights,
                    bias,
                    numPatches,
                    pw, // searchWindow + patchSize - 1,
                    pl, // searchWindow + patchSize - 1,
                    patchSize,
                    patchSize
                )
                // if ("lbp" in weights) lbpInit = true
                // if ("sobel" in weights) sobelInit = true
            } catch (err) {
                console.error(err, "webgl fail")
            }
        }
        // } else if (patchType === "MOSSE") {
        //     mosseCalc = new mosseFilterResponses()
        //     mosseCalc.init(weights, numPatches, patchSize, patchSize)
        // }

        const pdataLength = pw * pl
        const halfSearchWindow = (searchWindow - 1) / 2
        const responsePixels = searchWindow * searchWindow

        const vecProbs = new Float64Array(responsePixels)
        const patches = []
        for (let i = 0; i < numPatches; i++) {
            patches[i] = new Float64Array(pdataLength)
        }

        if (weightPoints) {
            // weighting of points
            let pointWeights = []
            for (let i = 0; i < numPatches; i++) {
                if (i in weightPoints) {
                    pointWeights[i * 2] = weightPoints[i]
                    pointWeights[i * 2 + 1] = weightPoints[i]
                } else {
                    pointWeights[i * 2] = 1
                    pointWeights[i * 2 + 1] = 1
                }
            }
            pointWeights = numeric.diag(pointWeights)
        }

        // faceDetector = new faceDetection(model, params.faceDetection)
    }

    // const start = (element, box) => {
    //     // check if model is initalized, else return false
    //     if (typeof model === "undefined") {
    //         console.log("tracker needs to be initalized before starting to track.")
    //         return false
    //     }
    //     // check if a runnerelement already exists, if not, use passed parameters
    //     if (typeof runnerElement === "undefined") {
    //         runnerElement = element
    //         runnerBox = box
    //     }

    //     faceDetector.init(element)

    //     // start named timeout function
    //     runnerTimeout = requestAnimationFrame(runnerFunction)
    // }

    // const stop = () => {
    //     // stop the running tracker if any exists
    //     cancelAnimationFrame(runnerTimeout)
    // }

    // const track = (element, box) => {
    //     emitEvent("clmtrackrBeforeTrack", params.eventDispatcher)

    //     let scaling, translateX, translateY, rotation
    //     let ptch, px, py

    //     if (first) {
    //         if (!detectingFace) {
    //             detectingFace = true

    //             // this returns a Promise
    //             faceDetector
    //                 .getInitialPosition(box)
    //                 .then(function(result) {
    //                     scaling = result[0]
    //                     rotation = result[1]
    //                     translateX = result[2]
    //                     translateY = result[3]

    //                     currentParameters[0] = scaling * Math.cos(rotation) - 1
    //                     currentParameters[1] = scaling * Math.sin(rotation)
    //                     currentParameters[2] = translateX
    //                     currentParameters[3] = translateY

    //                     currentPositions = calculatePositions(currentParameters, true)

    //                     first = false
    //                     detectingFace = false
    //                 })
    //                 .catch(function(error) {
    //                     // send an event on no face found
    //                     emitEvent("clmtrackrNotFound", params.eventDispatcher)

    //                     detectingFace = false
    //                 })
    //         }

    //         return false
    //     } else {
    //         facecheck_count += 1

    //         if (params.constantVelocity) {
    //             // calculate where to get patches via constant velocity prediction
    //             if (previousParameters.length >= 2) {
    //                 for (let i = 0; i < currentParameters.length; i++) {
    //                     currentParameters[i] =
    //                         relaxation * previousParameters[1][i] +
    //                         (1 - relaxation) * (2 * previousParameters[1][i] - previousParameters[0][i])
    //                     // currentParameters[i] = (3*previousParameters[2][i])
    //                          - (3*previousParameters[1][i]) + previousParameters[0][i];
    //                 }
    //             }
    //         }

    //         // change translation, rotation and scale parameters
    //         rotation = halfPI - Math.atan((currentParameters[0] + 1) / currentParameters[1])
    //         if (rotation > halfPI) {
    //             rotation -= Math.PI
    //         }
    //         scaling = currentParameters[1] / Math.sin(rotation)
    //         translateX = currentParameters[2]
    //         translateY = currentParameters[3]
    //     }

    //     // copy canvas to a new dirty canvas
    //     sketchCC.save()

    //     // clear canvas
    //     sketchCC.clearRect(0, 0, sketchW, sketchH)

    //     sketchCC.scale(1 / scaling, 1 / scaling)
    //     sketchCC.rotate(-rotation)
    //     sketchCC.translate(-translateX, -translateY)

    //     sketchCC.drawImage(element, 0, 0, element.width, element.height)

    //     sketchCC.restore()
    //     // 	get cropped images around new points based on model parameters (not scaled and translated)
    //     const patchPositions = calculatePositions(currentParameters, false)

    //     // check whether tracking is ok
    //     if (scoringWeights && facecheck_count % 10 === 0) {
    //         if (!checkTracking()) {
    //             // reset all parameters
    //             resetParameters()

    //             // send event to signal that tracking was lost
    //             emitEvent("clmtrackrLost", params.eventDispatcher)

    //             return false
    //         }
    //     }

    //     let pdata, pmatrix, grayscaleColor
    //     for (let i = 0; i < numPatches; i++) {
    //         px = patchPositions[i][0] - pw / 2
    //         py = patchPositions[i][1] - pl / 2
    //         ptch = sketchCC.getImageData(Math.round(px), Math.round(py), pw, pl)
    //         pdata = ptch.data

    //         // convert to grayscale
    //         pmatrix = patches[i]
    //         for (let j = 0; j < pdataLength; j++) {
    //             grayscaleColor = pdata[j * 4] * 0.3 + pdata[j * 4 + 1] * 0.59 + pdata[j * 4 + 2] * 0.11
    //             pmatrix[j] = grayscaleColor
    //         }
    //     }

    //     // draw weights for debugging
    //     // drawPatches(sketchCC, weights, patchSize, patchPositions, function(x) {return x*2000+127});

    //     // draw patches for debugging
    //     // drawPatches(sketchCC, patches, pw, patchPositions, false, [27,32,44,50]);

    //     if (patchType === "SVM") {
    //         if (typeof webglFi !== "undefined") {
    //             responses = getWebGLResponses(patches)
    //         } else if (typeof svmFi !== "undefined") {
    //             responses = svmFi.getResponses(patches)
    //         } else {
    //             throw new Error("SVM-filters do not seem to be initiated properly.")
    //         }
    //     } else if (patchType === "MOSSE") {
    //         responses = mosseCalc.getResponses(patches)
    //     }

    //     // option to increase sharpness of responses
    //     if (params.sharpenResponse) {
    //         for (let i = 0; i < numPatches; i++) {
    //             for (let j = 0; j < responses[i].length; j++) {
    //                 responses[i][j] = Math.pow(responses[i][j], params.sharpenResponse)
    //             }
    //         }
    //     }

    //     // draw responses for debugging
    //     // drawPatches(sketchCC, responses, searchWindow, patchPositions, function(x) {return x*255});

    //     // iterate until convergence or max 10, 20 iterations?:
    //     const originalPositions = currentPositions
    //     let jac
    //     const meanshiftVectors = []

    //     for (let i = 0; i < varianceSeq.length; i++) {
    //         // calculate jacobian
    //         jac = createJacobian(currentParameters, eigenVectors)

    //         // for debugging
    //         // var debugMVs = [];

    //         let opj0, opj1

    //         for (let j = 0; j < numPatches; j++) {
    //             opj0 = originalPositions[j][0] - ((searchWindow - 1) * scaling) / 2
    //             opj1 = originalPositions[j][1] - ((searchWindow - 1) * scaling) / 2

    //             // calculate PI x gaussians
    //             const vpsum = gpopt(
    //                 searchWindow,
    //                 currentPositions[j],
    //                 updatePosition,
    //                 vecProbs,
    //                 responses,
    //                 opj0,
    //                 opj1,
    //                 j,
    //                 varianceSeq[i],
    //                 scaling
    //             )

    //             // calculate meanshift-vector
    //             gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1, scaling)
    //             // var debugMatrixMV = gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1);

    //             meanshiftVectors[j] = [vecpos[0] - currentPositions[j][0], vecpos[1] - currentPositions[j][1]]

    //             // debugMVs[j] = debugMatrixMV;
    //         }

    //         // draw meanshiftVector for debugging
    //         // drawPatches(sketchCC, debugMVs, searchWindow, patchPositions, function(x) {return x*255*500});

    //         const meanShiftVector = numeric.rep([numPatches * 2, 1], 0.0)
    //         for (let k = 0; k < numPatches; k++) {
    //             meanShiftVector[k * 2][0] = meanshiftVectors[k][0]
    //             meanShiftVector[k * 2 + 1][0] = meanshiftVectors[k][1]
    //         }

    //         // compute pdm parameter update
    //         // var prior = numeric.mul(gaussianPD, PDMVariance);
    //         const prior = numeric.mul(gaussianPD, varianceSeq[i])
    //         let jtj
    //         if (params.weightPoints) {
    //             jtj = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, jac))
    //         } else {
    //             jtj = numeric.dot(numeric.transpose(jac), jac)
    //         }
    //         const cpMatrix = numeric.rep([numParameters + 4, 1], 0.0)
    //         for (let l = 0; l < numParameters + 4; l++) {
    //             cpMatrix[l][0] = currentParameters[l]
    //         }
    //         const priorP = numeric.dot(prior, cpMatrix)
    //         let jtv
    //         if (params.weightPoints) {
    //             jtv = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, meanShiftVector))
    //         } else {
    //             jtv = numeric.dot(numeric.transpose(jac), meanShiftVector)
    //         }
    //         const paramUpdateLeft = numeric.add(prior, jtj)
    //         const paramUpdateRight = numeric.sub(priorP, jtv)

    //         const paramUpdate = numeric.dot(numeric.inv(paramUpdateLeft), paramUpdateRight)
    //         // var paramUpdate = numeric.solve(paramUpdateLeft, paramUpdateRight, true);

    //         const oldPositions = currentPositions

    //         // update estimated parameters
    //         for (let k = 0; k < numParameters + 4; k++) {
    //             currentParameters[k] -= paramUpdate[k]
    //         }

    //         // clipping of parameters if they're too high
    //         let clip
    //         for (let k = 0; k < numParameters; k++) {
    //             clip = Math.abs(3 * Math.sqrt(eigenValues[k]))
    //             if (Math.abs(currentParameters[k + 4]) > clip) {
    //                 if (currentParameters[k + 4] > 0) {
    //                     currentParameters[k + 4] = clip
    //                 } else {
    //                     currentParameters[k + 4] = -clip
    //                 }
    //             }
    //         }

    //         // update current coordinates
    //         currentPositions = calculatePositions(currentParameters, true)

    //         // check if converged
    //         // calculate norm of parameterdifference
    //         let positionNorm = 0
    //         let pnsq_x, pnsq_y
    //         for (let k = 0; k < currentPositions.length; k++) {
    //             pnsq_x = currentPositions[k][0] - oldPositions[k][0]
    //             pnsq_y = currentPositions[k][1] - oldPositions[k][1]
    //             positionNorm += pnsq_x * pnsq_x + pnsq_y * pnsq_y
    //         }

    //         // if norm < limit, then break
    //         if (positionNorm < convergenceLimit) {
    //             break
    //         }
    //     }

    //     if (params.constantVelocity) {
    //         // add current parameter to array of previous parameters
    //         previousParameters.push(currentParameters.slice())
    //         if (previousParameters.length === 3) {
    //             previousParameters.shift()
    //         }
    //     }

    //     // store positions, for checking convergence
    //     if (previousPositions.length === 10) {
    //         previousPositions.shift()
    //     }
    //     previousPositions.push(currentPositions.slice(0))

    //     // send an event on each iteration
    //     emitEvent("clmtrackrIteration", params.eventDispatcher)

    //     // we must get a score before we can say we've converged
    //     if (scoringHistory.length >= 5 && this.getConvergence() < convergenceThreshold) {
    //         if (params.stopOnConvergence) {
    //             this.stop()
    //         }

    //         emitEvent("clmtrackrConverged", params.eventDispatcher)
    //     }

    //     // return new points
    //     return currentPositions
    // }

    // const getCurrentPosition = () => {
    //     if (first) {
    //         return false
    //     } else {
    //         return currentPositions
    //     }
    // }

    return {
        init
    }
}
