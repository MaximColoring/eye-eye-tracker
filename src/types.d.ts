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
    faceDetection?: {
        /** whether to use web workers for face detection (default is true) */
        useWebWorkers?: boolean
    }

    weightPoints?: number[]

    sharpenResponse?: boolean

    maxIterationsPerAnimFrame?: number
}

type FiltersTypes<T = number[][]> = {
    raw: T
    sobel: T
    lbp: T
}
