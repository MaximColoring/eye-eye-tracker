type TrackerParams = {
    /** whether to use constant velocity model when fitting (default is true) */
    constantVelocity?: boolean
    /** the size of the searchwindow around each point (default is 11) */
    searchWindow?: number
    /** threshold for when to assume we've lost tracking (default is 0.50) */
    scoreThreshold?: number
    /** whether to stop tracking when the fitting has converged (default is false) */
    stopOnConvergence?: boolean
    /** object with parameters for facedetection : */

    weightPoints?: number[]

    sharpenResponse?: number

    maxIterationsPerAnimFrame?: number
}

type FiltersTypes<T = number[][]> = {
    raw: T
    sobel: T
    lbp: T
}

type Box = {
    x: number
    y: number
    width: number
    height: number
}

declare module "jsfeat" {
    const a: any
    export default a
}

declare module "mosse" {
    const a: any
    export default a
}