type mSize = [number, number]

export type Model = {
    scoring: {
        size: mSize
        bias: number
        coef: number[]
    }
    path: {
        normal: number[][]
        vertices: number[][]
    }
    patchModel: {
        patchType: "SVM" | "MOSSE"
        bias: {
            raw: number[]
            sobel: number[]
            lbp: number[]
        }
        weights: {
            raw: number[][]
            sobel: number[][]
            lbp: number[][]
        }
        numPatches: number
        patchSize: mSize
        canvasSize: mSize
    }
    shapeModel: {
        eigenVectors: number[][]
        numEvalues: number
        eigenValues: number[]
        numPtsPerSample: number
        nonRegularizedVectors: number[]
        meanShape: number[][]
    }
    hints: {
        rightEye: [number, number]
        leftEye: [number, number]
        nose: [number, number]
    }
}

export const pca_20_svm: Model = require("./pca_20_svm.json")
