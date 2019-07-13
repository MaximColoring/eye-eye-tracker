type mSize = [number, number]
type Model = {
  scoring: {
    size: mSize
    bias: number;
    coef: number[];
  };
  path: {
    normal: number[][];
    vertices: number[][];
  };
  patchModel: {
    patchType: "SVM" | "MOSSE";
    bias: {
      raw: number[];
      sobel: number[];
      lbp: number[];
    };
    weights: {
      raw: number[][];
      sobel: number[][];
      lbp: number[][];
    };
    numPatches: number;
    patchSize: mSize;
    canvasSize: mSize;
  };
  shapeModel: {
    eigenVectors: number[][];
    numEvalues: number;
    eigenValues: number[];
    numPtsPerSample: number;
    nonRegularizedVectors: number[];
    meanShape: number[][];
  };
  hints: {
    rightEye: [number, number];
    leftEye: [number, number];
    nose: [number, number];
  };
};

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

export const Tracker: ({ searchWindow, scoreThreshold, stopOnConvergence, sharpenResponse, maxIterationsPerAnimFrame, weightPoints }: TrackerParams) => {
  init: (pmodel?: Model) => void;
  start: (element: HTMLCanvasElement, box?: number[]) => false | undefined;
  stop: () => void;
  getCurrentPosition: () => false | number[][];
};