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

import pca_20_svmModel from "./pca_20_svm"
// tslint:disable-next-line:no-var-requires
export const pca_20_svm: Model = pca_20_svmModel as Model
