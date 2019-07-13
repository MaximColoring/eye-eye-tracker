import numeric from "./utils/numeric"

export const createJacobian = (
    meanShape: number[][],
    numPatches: number,
    numParameters: number,
    parameters: number[],
    eigenVectors: number[][]
) => {
    const jacobian = numeric.rep([2 * numPatches, numParameters + 4], 0.0)
    let j0
    let j1
    for (let i = 0; i < numPatches; i++) {
        // 1
        j0 = meanShape[i][0]
        j1 = meanShape[i][1]
        for (let p = 0; p < numParameters; p++) {
            j0 += parameters[p + 4] * eigenVectors[i * 2][p]
            j1 += parameters[p + 4] * eigenVectors[i * 2 + 1][p]
        }
        jacobian[i * 2][0] = j0
        jacobian[i * 2 + 1][0] = j1
        // 2
        j0 = meanShape[i][1]
        j1 = meanShape[i][0]
        for (let p = 0; p < numParameters; p++) {
            j0 += parameters[p + 4] * eigenVectors[i * 2 + 1][p]
            j1 += parameters[p + 4] * eigenVectors[i * 2][p]
        }
        jacobian[i * 2][1] = -j0
        jacobian[i * 2 + 1][1] = j1
        // 3
        jacobian[i * 2][2] = 1
        jacobian[i * 2 + 1][2] = 0
        // 4
        jacobian[i * 2][3] = 0
        jacobian[i * 2 + 1][3] = 1
        // the rest
        for (let j = 0; j < numParameters; j++) {
            j0 =
                parameters[0] * eigenVectors[i * 2][j] -
                parameters[1] * eigenVectors[i * 2 + 1][j] +
                eigenVectors[i * 2][j]
            j1 =
                parameters[0] * eigenVectors[i * 2 + 1][j] +
                parameters[1] * eigenVectors[i * 2][j] +
                eigenVectors[i * 2 + 1][j]
            jacobian[i * 2][j + 4] = j0
            jacobian[i * 2 + 1][j + 4] = j1
        }
    }

    return jacobian
}

export const gpopt = (
    responseWidth: number,
    currentPositionsj: number[],
    updatePosition: Float64Array,
    vecProbs: Float64Array,
    responses: number[][],
    opj0: number,
    opj1: number,
    j: number,
    variance: number,
    scaling: number
) => {
    let pos_idx = 0
    let vpsum = 0
    let dx
    let dy
    for (let k = 0; k < responseWidth; k++) {
        updatePosition[1] = opj1 + k * scaling
        for (let l = 0; l < responseWidth; l++) {
            updatePosition[0] = opj0 + l * scaling

            dx = currentPositionsj[0] - updatePosition[0]
            dy = currentPositionsj[1] - updatePosition[1]
            vecProbs[pos_idx] = responses[j][pos_idx] * Math.exp((-0.5 * (dx * dx + dy * dy)) / (variance * scaling))

            vpsum += vecProbs[pos_idx]
            pos_idx++
        }
    }

    return vpsum
}

export const gpopt2 = (
    responseWidth: number,
    vecpos: Float64Array,
    updatePosition: Float64Array,
    vecProbs: Float64Array,
    vpsum: number,
    opj0: number,
    opj1: number,
    scaling: number
) => {
    let pos_idx = 0
    let vecsum = 0
    vecpos[0] = 0
    vecpos[1] = 0
    for (let k = 0; k < responseWidth; k++) {
        updatePosition[1] = opj1 + k * scaling
        for (let l = 0; l < responseWidth; l++) {
            updatePosition[0] = opj0 + l * scaling
            vecsum = vecProbs[pos_idx] / vpsum
            vecpos[0] += vecsum * updatePosition[0]
            vecpos[1] += vecsum * updatePosition[1]
            pos_idx++
        }
    }
    // return vecmatrix;
}
