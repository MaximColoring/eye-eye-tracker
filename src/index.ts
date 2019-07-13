import { Model, pca_20_svm } from "./models"
import numeric from "./utils/numeric"
import { webglFilter } from "./svmfilter/webglfilter"
import { FaceDetector, faceDetector } from "./facedetector/faceDetection"
import { createJacobian, gpopt, gpopt2 } from "./util"

const WIDTH = 0
const HEIGHT = 1

const convergenceThreshold = 0.5

// const sobelInit = false
// const lbpInit = false

// const currentPositions = []
// const prevParameters = []
// const prevPositions = []

// const patches = []
// const responses = []

/*
It's possible to experiment with the sequence of variances used for the finding the maximum in the KDE.
This sequence is pretty arbitrary, but was found to be okay using some manual testing.
*/
const varianceSeq = [10, 5, 1]
// var varianceSeq = [3,1.5,0.75];
// var varianceSeq = [6,3,0.75];
// const PDMVariance = 0.7

const relaxation = 0.1

// const first = true
// const detectingFace = false

const convergenceLimit = 0.01

// const facecheck_count = 0

// const scoringHistory = []

export const Tracker = ({
    searchWindow = 11,
    scoreThreshold = 0.5,
    stopOnConvergence = false,
    sharpenResponse = 0,
    maxIterationsPerAnimFrame = 3,
    weightPoints
}: TrackerParams) => {
    let _initialized = false
    let detector: FaceDetector
    let model: Model
    let runnerElement: HTMLCanvasElement
    let runnerBox: number[] | undefined
    let runnerTimeout: number
    let detecting: boolean = false
    let first: boolean = true
    let currentParameters: number[] = []
    let prevParameters: number[][] = []
    const updatePosition = new Float64Array(2)
    let currentPositions: number[][] = []
    let prevPositions: number[][][] = []
    let vecProbs: Float64Array
    const vecpos = new Float64Array(2)

    const responseMode = "single"
    const responseList: Array<keyof FiltersTypes> = ["raw"]
    let responseIndex = 0

    let pw: number
    let pl: number
    let pdataLength: number
    let patches: number[][]
    let responses: number[][]
    let gaussianPD: number[][]
    let numPatches: number
    let numParameters: number
    let patchType: "SVM" | "MOSSE"
    let meanShape: number[][]
    let facecheck_count = 0

    let eigenVectors: number[][]
    let eigenValues: number[]

    let sketchCanvas: HTMLCanvasElement
    let sketchCC: CanvasRenderingContext2D
    let sketchH: number
    let sketchW: number

    let scoringCanvas: HTMLCanvasElement
    let scoringContext: CanvasRenderingContext2D
    let scoringWeights: Float64Array
    let scoringBias: number
    let scoringHistory: number[] = []

    let filter: ReturnType<typeof webglFilter>

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

    const init = (pmodel: Model = pca_20_svm) => {
        if (_initialized) return
        // default model is pca 20 svm model
        model = pmodel
        numPatches = model.patchModel.numPatches
        patchType = model.patchModel.patchType
        numParameters = model.shapeModel.numEvalues
        eigenValues = model.shapeModel.eigenValues
        const { canvasSize: modelSize, weights, bias } = model.patchModel
        const patchSize = model.patchModel.patchSize[0]
        const { nonRegularizedVectors } = model.shapeModel
        // if (patchType === "MOSSE") searchWindow = patchSize

        // // set up canvas to work on
        sketchCanvas = document.createElement("canvas")
        sketchCC = sketchCanvas.getContext("2d")!
        scoringCanvas = document.createElement("canvas")
        scoringContext = scoringCanvas.getContext("2d")!

        sketchW = sketchCanvas.width = modelSize[WIDTH] + (searchWindow - 1) + patchSize - 1
        sketchH = sketchCanvas.height = modelSize[HEIGHT] + (searchWindow - 1) + patchSize - 1

        // load eigenvectors
        eigenVectors = numeric.rep([numPatches * 2, numParameters], 0.0) as number[][]
        for (let i = 0; i < numPatches * 2; i++) {
            for (let j = 0; j < numParameters; j++) {
                eigenVectors[i][j] = model.shapeModel.eigenVectors[i][j]
            }
        }

        // load mean shape
        meanShape = []
        for (let i = 0; i < numPatches; i++) {
            meanShape[i] = [model.shapeModel.meanShape[i][0], model.shapeModel.meanShape[i][1]]
        }

        // get max and mins, width and height of meanshape

        for (let i = 0; i < numPatches; i++) {
            if (meanShape[i][0] < msMeta.x.min) msMeta.x.min = meanShape[i][0]
            if (meanShape[i][1] < msMeta.y.min) msMeta.y.min = meanShape[i][1]
            if (meanShape[i][0] > msMeta.x.max) msMeta.x.max = meanShape[i][0]
            if (meanShape[i][1] > msMeta.y.max) msMeta.y.max = meanShape[i][1]
        }
        msMeta.width = msMeta.x.max - msMeta.x.min
        msMeta.height = msMeta.y.max - msMeta.y.min

        // get scoringweights if they exist
        if (model.scoring) {
            scoringWeights = new Float64Array(model.scoring.coef)
            scoringBias = model.scoring.bias
            scoringCanvas.width = model.scoring.size[0]
            scoringCanvas.height = model.scoring.size[1]
        }

        // precalculate gaussianPriorDiagonal
        gaussianPD = numeric.rep([numParameters + 4, numParameters + 4], 0) as number[][]
        // set values and append manual inverse
        for (let i = 0; i < numParameters; i++) {
            gaussianPD[i + 4][i + 4] = nonRegularizedVectors.indexOf(i) >= 0 ? 1 / 10000000 : 1 / eigenValues[i]
        }

        currentParameters = Array(numParameters + 4).fill(0)

        pw = patchType === "SVM" ? patchSize + searchWindow - 1 : searchWindow
        pl = patchType === "SVM" ? patchSize + searchWindow - 1 : searchWindow

        if (patchType === "SVM") {
            filter = webglFilter()
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

        pdataLength = pw * pl
        const responsePixels = searchWindow * searchWindow

        vecProbs = new Float64Array(responsePixels)
        patches = []
        for (let i = 0; i < numPatches; i++) {
            patches[i] = new Float64Array(pdataLength) as any
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

        detector = faceDetector(model, {})
        _initialized = true
    }

    const checkTracking = () => {
        const trackingImgW = 20
        const trackingImgH = 22

        scoringContext.drawImage(
            sketchCanvas,
            Math.round(msMeta.x.min + msMeta.width / 4.5),
            Math.round(msMeta.y.min - msMeta.height / 12),
            Math.round(msMeta.width - (msMeta.width * 2) / 4.5),
            Math.round(msMeta.height - msMeta.height / 12),
            0,
            0,
            trackingImgW,
            trackingImgH
        )
        // getImageData of canvas
        const imgData = scoringContext.getImageData(0, 0, trackingImgW, trackingImgH)
        // convert data to grayscale
        const trackingImgSize = trackingImgW * trackingImgH
        const scoringData = new Array(trackingImgSize)
        const scdata = imgData.data
        let scmax = 0
        for (let i = 0; i < trackingImgSize; i++) {
            scoringData[i] = scdata[i * 4] * 0.3 + scdata[i * 4 + 1] * 0.59 + scdata[i * 4 + 2] * 0.11
            scoringData[i] = Math.log(scoringData[i] + 1)
            if (scoringData[i] > scmax) scmax = scoringData[i]
        }

        if (scmax > 0) {
            // normalize & multiply by svmFilter
            let mean = 0
            for (let i = 0; i < trackingImgSize; i++) {
                mean += scoringData[i]
            }
            mean /= trackingImgSize
            let sd = 0
            for (let i = 0; i < trackingImgSize; i++) {
                sd += (scoringData[i] - mean) * (scoringData[i] - mean)
            }
            sd /= trackingImgSize
            sd = Math.sqrt(sd)

            let score = 0
            for (let i = 0; i < trackingImgSize; i++) {
                scoringData[i] = (scoringData[i] - mean) / sd
                score += scoringData[i] * scoringWeights[i]
            }
            score += scoringBias
            score = 1 / (1 + Math.exp(-score))

            if (scoringHistory.length === 5) {
                scoringHistory.shift()
            }
            scoringHistory.push(score)

            if (scoringHistory.length > 4) {
                // get average
                let meanscore = 0
                for (let i = 0; i < 5; i++) {
                    meanscore += scoringHistory[i]
                }
                meanscore /= 5
                // if below threshold, then reset (return false)
                if (meanscore < scoreThreshold) return false
            }
        }
        return true
    }
    const getConvergence = () => {
        if (prevPositions.length < 10) return 999999

        let prevX = 0.0
        let prevY = 0.0
        let currX = 0.0
        let currY = 0.0

        // average 5 previous positions
        for (let i = 0; i < 5; i++) {
            for (let j = 0; j < numPatches; j++) {
                prevX += prevPositions[i][j][0]
                prevY += prevPositions[i][j][1]
            }
        }
        prevX /= 5
        prevY /= 5

        // average 5 positions before that
        for (let i = 5; i < 10; i++) {
            for (let j = 0; j < numPatches; j++) {
                currX += prevPositions[i][j][0]
                currY += prevPositions[i][j][1]
            }
        }
        currX /= 5
        currY /= 5

        // calculate difference
        const diffX = currX - prevX
        const diffY = currY - prevY
        let msavg = diffX * diffX + diffY * diffY
        msavg /= prevPositions.length
        return msavg
    }

    const start = (element: HTMLCanvasElement, box?: number[]) => {
        // check if model is initalized, else return false
        if (!model) {
            console.log("tracker needs to be initalized before starting to track.")
            return false
        }
        // check if a runnerelement already exists, if not, use passed parameters
        if (!runnerElement) {
            runnerElement = element
            runnerBox = box
        }
        detector.init(element)
        // start named timeout function
        runnerTimeout = requestAnimationFrame(runnerFunction)
    }

    const runnerFunction = () => {
        runnerTimeout = requestAnimationFrame(runnerFunction)
        // schedule as many iterations as we can during each request
        const startTime = new Date().getTime()
        let run_counter = 0

        while (new Date().getTime() - startTime < 16 && run_counter < maxIterationsPerAnimFrame) {
            const tracking = track(runnerElement, runnerBox)
            if (!tracking) break
            run_counter++
        }
    }

    const stop = () => {
        // stop the running tracker if any exists
        cancelAnimationFrame(runnerTimeout)
    }

    const calculatePositions = (parameters: number[], useTransforms: boolean) => {
        let x
        let y
        let a
        let b
        const positions = []
        for (let i = 0; i < numPatches; i++) {
            x = meanShape[i][0]
            y = meanShape[i][1]
            for (let j = 0; j < parameters.length - 4; j++) {
                x += model.shapeModel.eigenVectors[i * 2][j] * parameters[j + 4]
                y += model.shapeModel.eigenVectors[i * 2 + 1][j] * parameters[j + 4]
            }
            if (useTransforms) {
                a = parameters[0] * x - parameters[1] * y + parameters[2]
                b = parameters[0] * y + parameters[1] * x + parameters[3]
                x += a
                y += b
            }
            positions[i] = [x, y]
        }

        return positions
    }

    const resetParameters = () => {
        first = true
        scoringHistory = []
        prevParameters = []
        currentPositions = []
        prevPositions = []
        for (let i = 0; i < currentParameters.length; i++) {
            currentParameters[i] = 0
        }
    }

    const getWebGLResponsesType = (type: keyof FiltersTypes) => {
        switch (type) {
            case "lbp":
                return filter.getLBPResponses(patches)
            case "raw":
                return filter.getRawResponses(patches)
            case "sobel":
                return filter.getSobelResponses(patches)
        }
    }

    const getWebGLResponses = (): number[][] => {
        if (responseMode === "single") {
            return getWebGLResponsesType(responseList[0])
        } else if (responseMode === "cycle") {
            const response = getWebGLResponsesType(responseList[responseIndex])
            responseIndex++
            if (responseIndex >= responseList.length) responseIndex = 0
            return response
        } else {
            // blend
            const res = []
            for (let i = 0; i < responseList.length; i++) {
                res[i] = getWebGLResponsesType(responseList[i])
            }
            const blendedResponses = []
            const searchWindowSize = searchWindow * searchWindow
            for (let i = 0; i < numPatches; i++) {
                const response = Array(searchWindowSize)
                for (let k = 0; k < searchWindowSize; k++) response[k] = 0
                for (let j = 0; j < responseList.length; j++) {
                    for (let k = 0; k < searchWindowSize; k++) {
                        response[k] += res[j][i][k] / responseList.length
                    }
                }
                blendedResponses[i] = response
            }
            return blendedResponses
        }
    }

    const track = (element: CanvasImageSource, box?: number[]) => {
        let scaling: number
        let translateX: number
        let translateY: number
        let rotation: number
        let ptch: ImageData
        let px: number
        let py: number

        if (first) {
            if (!detecting) {
                detecting = true

                // this returns a Promise
                detector
                    .getInitialPosition(box)
                    .then(result => {
                        scaling = result[0]
                        rotation = result[1]
                        translateX = result[2]
                        translateY = result[3]

                        currentParameters[0] = scaling * Math.cos(rotation) - 1
                        currentParameters[1] = scaling * Math.sin(rotation)
                        currentParameters[2] = translateX
                        currentParameters[3] = translateY

                        currentPositions = calculatePositions(currentParameters, true)

                        first = false
                    })
                    .catch(e => {
                        console.error("error in track", e)
                    })
                    .finally(() => {
                        detecting = false
                    })
            }

            return false
        } else {
            facecheck_count += 1

            // calculate where to get patches via constant velocity prediction
            if (prevParameters.length >= 2) {
                for (let i = 0; i < currentParameters.length; i++) {
                    currentParameters[i] =
                        relaxation * prevParameters[1][i] +
                        (1 - relaxation) * (2 * prevParameters[1][i] - prevParameters[0][i])
                }
            }

            // change translation, rotation and scale parameters
            rotation = Math.PI / 2 - Math.atan((currentParameters[0] + 1) / currentParameters[1])
            if (rotation > Math.PI / 2) {
                rotation -= Math.PI
            }
            scaling = currentParameters[1] / Math.sin(rotation)
            translateX = currentParameters[2]
            translateY = currentParameters[3]
        }

        // copy canvas to a new dirty canvas
        sketchCC.save()

        // clear canvas
        sketchCC.clearRect(0, 0, sketchW, sketchH)

        sketchCC.scale(1 / scaling, 1 / scaling)
        sketchCC.rotate(-rotation)
        sketchCC.translate(-translateX, -translateY)

        sketchCC.drawImage(element, 0, 0, element.width as number, element.height as number)

        sketchCC.restore()
        // 	get cropped images around new points based on model parameters (not scaled and translated)
        const patchPositions = calculatePositions(currentParameters, false)

        // // check whether tracking is ok
        if (scoringWeights && facecheck_count % 10 === 0) {
            if (!checkTracking()) {
                resetParameters()
                console.log("resetting params")
                return false
            }
        }

        let pdata: Uint8ClampedArray
        let pmatrix
        let grayscaleColor
        for (let i = 0; i < numPatches; i++) {
            px = patchPositions[i][0] - pw / 2
            py = patchPositions[i][1] - pl / 2
            ptch = sketchCC.getImageData(Math.round(px), Math.round(py), pw, pl)
            pdata = ptch.data

            // convert to grayscale
            pmatrix = patches[i]
            for (let j = 0; j < pdataLength; j++) {
                grayscaleColor = pdata[j * 4] * 0.3 + pdata[j * 4 + 1] * 0.59 + pdata[j * 4 + 2] * 0.11
                pmatrix[j] = grayscaleColor
            }
        }

        // draw weights for debugging
        // drawPatches(sketchCC, weights, patchSize, patchPositions, function(x) {return x*2000+127});

        // draw patches for debugging
        // drawPatches(sketchCC, patches, pw, patchPositions, false, [27,32,44,50]);

        // if (patchType === "SVM") {
        responses = getWebGLResponses()
        // } else if (patchType === "MOSSE") {
        // responses = mosseCalc.getResponses(patches)
        // }

        // option to increase sharpness of responses
        if (sharpenResponse) {
            for (let i = 0; i < numPatches; i++) {
                for (let j = 0; j < responses[i].length; j++) {
                    responses[i][j] = Math.pow(responses[i][j], sharpenResponse)
                }
            }
        }

        // draw responses for debugging
        // drawPatches(sketchCC, responses, searchWindow, patchPositions, function(x) {return x*255});

        // iterate until convergence or max 10, 20 iterations?:
        const originalPositions = currentPositions
        let jac
        const meanshiftVectors = []

        for (const seq of varianceSeq) {
            // calculate jacobian
            jac = createJacobian(meanShape, numPatches, numParameters, currentParameters, eigenVectors) as number[][]

            let opj0
            let opj1

            for (let j = 0; j < numPatches; j++) {
                opj0 = originalPositions[j][0] - ((searchWindow - 1) * scaling) / 2
                opj1 = originalPositions[j][1] - ((searchWindow - 1) * scaling) / 2

                // calculate PI x gaussians
                const vpsum = gpopt(
                    searchWindow,
                    currentPositions[j],
                    updatePosition,
                    vecProbs,
                    responses,
                    opj0,
                    opj1,
                    j,
                    seq,
                    scaling
                )

                // calculate meanshift-vector
                gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1, scaling)
                // var debugMatrixMV = gpopt2(searchWindow, vecpos, updatePosition, vecProbs, vpsum, opj0, opj1);

                meanshiftVectors[j] = [vecpos[0] - currentPositions[j][0], vecpos[1] - currentPositions[j][1]]

                // debugMVs[j] = debugMatrixMV;
            }

            // draw meanshiftVector for debugging
            // drawPatches(sketchCC, debugMVs, searchWindow, patchPositions, function(x) {return x*255*500});

            const meanShiftVector = numeric.rep([numPatches * 2, 1], 0.0) as number[][]
            for (let k = 0; k < numPatches; k++) {
                meanShiftVector[k * 2][0] = meanshiftVectors[k][0]
                meanShiftVector[k * 2 + 1][0] = meanshiftVectors[k][1]
            }

            // compute pdm parameter update
            // var prior = numeric.mul(gaussianPD, PDMVariance);
            const prior = numeric.mul(gaussianPD, seq)
            const jtj = numeric.dot(numeric.transpose(jac), jac) as number[][]
            // if (weightPoints) {
            //     jtj = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, jac))
            // } else {
            // jtj = numeric.dot(numeric.transpose(jac), jac)
            // }
            const cpMatrix = numeric.rep([numParameters + 4, 1], 0.0) as number[][]
            for (let l = 0; l < numParameters + 4; l++) {
                cpMatrix[l][0] = currentParameters[l]
            }
            const priorP = numeric.dot(prior, cpMatrix) as number[][]
            const jtv = numeric.dot(numeric.transpose(jac), meanShiftVector) as number[][]
            // if (params.weightPoints) {
            //     jtv = numeric.dot(numeric.transpose(jac), numeric.dot(pointWeights, meanShiftVector))
            // } else {
            //     jtv =
            // }
            const paramUpdateLeft = numeric.add(prior, jtj)
            const paramUpdateRight = numeric.sub(priorP, jtv)

            const paramUpdate = numeric.dot(numeric.inv(paramUpdateLeft), paramUpdateRight) as number[]
            // var paramUpdate = numeric.solve(paramUpdateLeft, paramUpdateRight, true);

            const oldPositions = currentPositions

            // update estimated parameters
            for (let k = 0; k < numParameters + 4; k++) {
                currentParameters[k] -= paramUpdate[k]
            }

            // clipping of parameters if they're too high
            let clip
            for (let k = 0; k < numParameters; k++) {
                clip = Math.abs(3 * Math.sqrt(eigenValues[k]))
                if (Math.abs(currentParameters[k + 4]) > clip) {
                    const cp = currentParameters[k + 4]
                    currentParameters[k + 4] = cp > 0 ? clip : -clip
                }
            }

            // update current coordinates
            currentPositions = calculatePositions(currentParameters, true)

            // check if converged
            // calculate norm of parameterdifference
            let positionNorm = 0
            let pnsq_x
            let pnsq_y
            for (let k = 0; k < currentPositions.length; k++) {
                pnsq_x = currentPositions[k][0] - oldPositions[k][0]
                pnsq_y = currentPositions[k][1] - oldPositions[k][1]
                positionNorm += pnsq_x * pnsq_x + pnsq_y * pnsq_y
            }

            // if norm < limit, then break
            if (positionNorm < convergenceLimit) {
                break
            }
        }

        // add current parameter to array of previous parameters
        prevParameters.push(currentParameters.slice())
        if (prevParameters.length === 3) {
            prevParameters.shift()
        }

        // store positions, for checking convergence
        if (prevPositions.length === 10) {
            prevPositions.shift()
        }
        prevPositions.push([...currentPositions])

        // we must get a score before we can say we've converged
        if (scoringHistory.length >= 5 && getConvergence() < convergenceThreshold) {
            if (stopOnConvergence) {
                stop()
            }
        }

        // return new points
        return currentPositions
    }

    const getCurrentPosition = () => (first ? false : currentPositions)

    return {
        init,
        start,
        stop,
        getCurrentPosition
    }
}
